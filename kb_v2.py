"""Strict, authenticated multi-corpus KB read plane.

This module deliberately does not import the legacy FastAPI application.  The
caller injects the shared embedding and reranker dependencies, which keeps the
v1 route/model surface unchanged while allowing one process to serve both APIs.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import re
import sqlite3
import stat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal, Optional
from urllib.parse import urlparse

import httpx
import yaml
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AnyHttpUrl, AnyUrl, BaseModel, ConfigDict, Field, StrictBool, field_validator


CorpusName = Literal["homelab", "ai"]
ScopeName = Literal["homelab", "ai", "both", "auto"]
SelectedScope = Literal["homelab", "ai", "both", "none"]

CHROMA_BASE = "http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database"
CLIENT_CONFIG_PATH = "/opt/kb/v2-clients.yml"
ROUTER_CONFIG_PATH = "/opt/kb/corpus-router.yml"
PRECALIBRATION_ROUTER_VERSION = "corpus-router-v1-precalibration"
HOMELAB_DECAY_HALF_LIFE = 540.0
HOMELAB_DECAY_FLOOR = 0.30
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIMENSION = 768
COLLECTION_SCHEMA_VERSION = 1

CORPUS_REGISTRY = {
    "homelab": {
        "db_path": "/opt/kb/kb.db",
        "collection": "kb_collection",
    },
    "ai": {
        "db_path": "/opt/ai-kb/ai-kb.db",
        "collection": "ai_kb_collection",
    },
}

_collection_ids: dict[str, str] = {}
_bearer = HTTPBearer(auto_error=False, scheme_name="bearerAuth")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchRequestV2(StrictModel):
    query: str = Field(min_length=1, max_length=1500)
    scope: ScopeName
    top_k: int = Field(default=5, ge=1, le=5, strict=True)
    allow_degraded: StrictBool = False

    @field_validator("query")
    @classmethod
    def trim_nonempty_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain a non-whitespace character")
        return value


class SearchResultV2(StrictModel):
    corpus: CorpusName
    entry_id: int
    ref: str = Field(pattern=r"^(homelab|ai):[1-9][0-9]*$")
    title: str
    content: Optional[str]
    tags: Optional[str]
    public_source_url: Optional[AnyHttpUrl]
    link: AnyUrl
    distance: float
    relevance: float
    final_score: float


class CorpusResultsV2(StrictModel):
    searched: bool
    available: bool
    count: int = Field(ge=0, le=5)
    results: list[SearchResultV2] = Field(max_length=5)


class CorporaV2(StrictModel):
    homelab: CorpusResultsV2
    ai: CorpusResultsV2


class SearchResponseV2(StrictModel):
    query: str
    requested_scope: ScopeName
    selected_scope: SelectedScope
    routing_mode: Literal["explicit", "auto"]
    routing_reason: str
    needs_clarification: bool
    router_version: Optional[str]
    degraded_corpora: list[CorpusName]
    total_count: int = Field(ge=0, le=10)
    corpora: CorporaV2


class CorpusHealthV2(StrictModel):
    ready: bool
    collection: str
    reason: Optional[str] = None


class HealthResponseV2(StrictModel):
    status: Literal["ok", "degraded"]
    corpora: dict[CorpusName, CorpusHealthV2]
    auto_routing_enabled: bool = False


@dataclass(frozen=True)
class AuthorizedClient:
    name: str
    allowed_corpora: frozenset[str]
    allowed_scopes: frozenset[str]


@dataclass(frozen=True)
class RouterConfig:
    router_version: str
    accept_thresholds: dict[str, float]
    reject_threshold: float
    both_margin: float
    dead_zone_lower: float
    dead_zone_upper: float
    candidate_k: int
    max_distance: dict[str, float]
    ai_decay_mode: str
    ai_decay_half_life_days: Optional[float] = None
    ai_decay_floor: Optional[float] = None


@dataclass
class Candidate:
    corpus: str
    entry_id: int
    title: str
    content: Optional[str]
    summary: Optional[str]
    tags: Optional[str]
    source: Optional[str]
    date: Optional[str]
    distance: float
    relevance: float = 0.0
    final_score: float = 0.0


def _config_path() -> str:
    return os.getenv("KB_V2_CLIENTS_CONFIG", CLIENT_CONFIG_PATH)


def _router_config_path() -> str:
    return os.getenv("KB_CORPUS_ROUTER_CONFIG", ROUTER_CONFIG_PATH)


def _number(value, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"router config {name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum or numeric > maximum:
        raise RuntimeError(f"router config {name} is outside [{minimum}, {maximum}]")
    return numeric


def _load_router_config() -> RouterConfig:
    path = _router_config_path()
    try:
        file_stat = os.stat(path)
        if stat.S_IMODE(file_stat.st_mode) != 0o600 or file_stat.st_uid != os.geteuid():
            raise RuntimeError("router config ownership or mode is invalid")
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError("router config is unavailable") from exc
    required = {
        "router_version",
        "accept_thresholds",
        "reject_threshold",
        "both_margin",
        "dead_zone",
        "candidate_k",
        "max_distance",
        "ai_decay",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise RuntimeError("router config shape is invalid")
    version = document["router_version"]
    if not isinstance(version, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", version):
        raise RuntimeError("router config version is invalid")

    def corpus_numbers(field: str, minimum: float, maximum: float) -> dict[str, float]:
        values = document[field]
        if not isinstance(values, dict) or set(values) != set(CORPUS_REGISTRY):
            raise RuntimeError(f"router config {field} shape is invalid")
        return {
            corpus: _number(values[corpus], f"{field}.{corpus}", minimum, maximum)
            for corpus in CORPUS_REGISTRY
        }

    accept = corpus_numbers("accept_thresholds", 0.0, 1.0)
    max_distance = corpus_numbers("max_distance", 0.0, 2.0)
    reject = _number(document["reject_threshold"], "reject_threshold", 0.0, 1.0)
    margin = _number(document["both_margin"], "both_margin", 0.0, 1.0)
    dead_zone = document["dead_zone"]
    if not isinstance(dead_zone, dict) or set(dead_zone) != {"lower", "upper"}:
        raise RuntimeError("router config dead_zone shape is invalid")
    dead_lower = _number(dead_zone["lower"], "dead_zone.lower", 0.0, 1.0)
    dead_upper = _number(dead_zone["upper"], "dead_zone.upper", 0.0, 1.0)
    if reject > dead_lower or dead_lower >= dead_upper or dead_upper > min(accept.values()):
        raise RuntimeError("router config thresholds are inconsistent")
    candidate_k = document["candidate_k"]
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or not 1 <= candidate_k <= 100:
        raise RuntimeError("router config candidate_k must be an integer in [1, 100]")

    decay = document["ai_decay"]
    if not isinstance(decay, dict) or "mode" not in decay:
        raise RuntimeError("router config ai_decay shape is invalid")
    mode = decay["mode"]
    half_life = floor = None
    if mode == "disabled":
        if set(decay) != {"mode"}:
            raise RuntimeError("disabled AI decay cannot have parameters")
    elif mode == "rational":
        if set(decay) != {"mode", "half_life_days", "floor"}:
            raise RuntimeError("rational AI decay parameters are incomplete")
        half_life = _number(decay["half_life_days"], "ai_decay.half_life_days", 1.0, 36500.0)
        floor = _number(decay["floor"], "ai_decay.floor", 0.0, 1.0)
    else:
        raise RuntimeError("router config AI decay mode is invalid")
    config = RouterConfig(
        router_version=version,
        accept_thresholds=accept,
        reject_threshold=reject,
        both_margin=margin,
        dead_zone_lower=dead_lower,
        dead_zone_upper=dead_upper,
        candidate_k=candidate_k,
        max_distance=max_distance,
        ai_decay_mode=mode,
        ai_decay_half_life_days=half_life,
        ai_decay_floor=floor,
    )
    expected_precalibration = {
        "accept_thresholds": {"homelab": 0.60, "ai": 0.60},
        "reject_threshold": 0.40,
        "both_margin": 0.05,
        "dead_zone_lower": 0.40,
        "dead_zone_upper": 0.60,
        "candidate_k": 25,
        "max_distance": {"homelab": 0.60, "ai": 0.60},
        "ai_decay_mode": "disabled",
    }
    if config.router_version != PRECALIBRATION_ROUTER_VERSION or any(
        getattr(config, key) != value for key, value in expected_precalibration.items()
    ):
        raise RuntimeError("router version is not bound to the approved effective values")
    return config


def _load_clients() -> list[tuple[AuthorizedClient, str]]:
    """Load and fully validate the token-name allowlist; token values stay in env."""
    try:
        path = _config_path()
        file_stat = os.stat(path)
        if stat.S_IMODE(file_stat.st_mode) != 0o600 or file_stat.st_uid != os.geteuid():
            raise RuntimeError("client config ownership or mode is invalid")
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except (OSError, RuntimeError, UnicodeError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=503, detail="V2 client authorization is unavailable") from exc

    if not isinstance(document, dict) or set(document) != {"clients"}:
        raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
    clients = document["clients"]
    if not isinstance(clients, dict) or not clients:
        raise HTTPException(status_code=503, detail="V2 client authorization is invalid")

    resolved: list[tuple[AuthorizedClient, str]] = []
    seen_tokens: set[str] = set()
    valid_corpora = set(CORPUS_REGISTRY)
    valid_scopes = {"homelab", "ai", "both", "auto"}
    for name, raw in clients.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        if set(raw) != {"token_env", "allowed_corpora", "allowed_scopes"}:
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        token_env = raw["token_env"]
        corpora = raw["allowed_corpora"]
        scopes = raw["allowed_scopes"]
        if (
            not isinstance(token_env, str)
            or not re.fullmatch(r"KB_V2_TOKEN_[A-Z0-9_]{3,64}", token_env)
            or not isinstance(corpora, list)
            or not corpora
            or not all(isinstance(value, str) for value in corpora)
            or len(corpora) != len(set(corpora))
            or not set(corpora) <= valid_corpora
            or not isinstance(scopes, list)
            or not scopes
            or not all(isinstance(value, str) for value in scopes)
            or len(scopes) != len(set(scopes))
            or not set(scopes) <= valid_scopes
        ):
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        if "homelab" in scopes and "homelab" not in corpora:
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        if "ai" in scopes and "ai" not in corpora:
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        if "both" in scopes and set(corpora) != valid_corpora:
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        token = os.getenv(token_env)
        if (
            not token
            or not re.fullmatch(r"[A-Za-z0-9._~-]{32,256}", token)
            or token in seen_tokens
        ):
            raise HTTPException(status_code=503, detail="V2 client authorization is invalid")
        seen_tokens.add(token)
        resolved.append((AuthorizedClient(name, frozenset(corpora), frozenset(scopes)), token))
    return resolved


def authorize_v2(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> AuthorizedClient:
    return _authorize_credentials(credentials, _load_clients())


def _authorize_credentials(
    credentials: Optional[HTTPAuthorizationCredentials],
    configured_clients: list[tuple[AuthorizedClient, str]],
) -> AuthorizedClient:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    presented = credentials.credentials
    if not re.fullmatch(r"[A-Za-z0-9._~-]{32,256}", presented or ""):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    match: Optional[AuthorizedClient] = None
    for client, expected in configured_clients:
        if hmac.compare_digest(presented, expected):
            match = client
    if match is None:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return match


def _collection_descriptor(corpus: str, force_refresh: bool = False) -> dict:
    profile = CORPUS_REGISTRY[corpus]
    response = httpx.get(
        f"{CHROMA_BASE}/collections/{profile['collection']}", timeout=10
    )
    if response.status_code != 200:
        raise RuntimeError(f"collection_lookup_http_{response.status_code}")
    descriptor = response.json()
    collection_id = descriptor.get("id")
    if not isinstance(collection_id, str) or not collection_id:
        raise RuntimeError("collection descriptor has an invalid id")
    # A name lookup is authoritative even if an old UUID still returns 200.
    _collection_ids[corpus] = collection_id
    return descriptor


def _corpus_health(corpus: str, reranker_ready: bool) -> CorpusHealthV2:
    profile = CORPUS_REGISTRY[corpus]
    try:
        db = sqlite3.connect(f"file:{profile['db_path']}?mode=ro", uri=True)
        try:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("sqlite_integrity")
            db.execute("SELECT id FROM entries LIMIT 1").fetchall()
        finally:
            db.close()
        descriptor = _collection_descriptor(corpus)
        metadata = descriptor.get("metadata") or {}
        expected = {
            "hnsw:space": "cosine",
            "corpus": corpus,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "schema_version": COLLECTION_SCHEMA_VERSION,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise RuntimeError("collection_metadata_mismatch")
        if not metadata.get("created_at"):
            raise RuntimeError("collection_metadata_mismatch")
        space = ((descriptor.get("configuration_json") or {}).get("hnsw") or {}).get("space")
        if space != "cosine":
            raise RuntimeError("collection_metric_mismatch")
        if not reranker_ready:
            raise RuntimeError("reranker_unavailable")
        return CorpusHealthV2(ready=True, collection=profile["collection"])
    except Exception as exc:
        reason = str(exc)
        if not reason or "/" in reason or "\\" in reason:
            reason = "dependency_unavailable"
        return CorpusHealthV2(
            ready=False,
            collection=profile["collection"],
            reason=reason[:120],
        )


def _query_collection(
    corpus: str, embedding: list[float], router_config: RouterConfig
) -> list[dict]:
    payload = {
        "query_embeddings": [embedding],
        "n_results": router_config.candidate_k,
        "include": ["distances", "documents", "metadatas"],
    }
    response = None
    for attempt in range(2):
        if attempt or corpus not in _collection_ids:
            _collection_descriptor(corpus, force_refresh=bool(attempt))
        response = httpx.post(
            f"{CHROMA_BASE}/collections/{_collection_ids[corpus]}/query",
            json=payload,
            timeout=30,
        )
        if response.status_code != 404:
            break
    if response is None or response.status_code != 200:
        status = response.status_code if response is not None else "unavailable"
        raise RuntimeError(f"collection_query_http_{status}")
    value = response.json()
    ids_outer = value.get("ids")
    distances_outer = value.get("distances")
    if (
        not isinstance(ids_outer, list)
        or len(ids_outer) != 1
        or not isinstance(ids_outer[0], list)
        or not isinstance(distances_outer, list)
        or len(distances_outer) != 1
        or not isinstance(distances_outer[0], list)
        or len(ids_outer[0]) != len(distances_outer[0])
    ):
        raise RuntimeError("collection query response shape is invalid")
    ids = ids_outer[0]
    distances = distances_outer[0]
    candidates = []
    for index, raw_id in enumerate(ids):
        entry_id = str(raw_id)
        if not entry_id.isdigit() or int(entry_id) < 1:
            continue
        distance = distances[index]
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise RuntimeError("collection query returned an invalid distance")
        distance = float(distance)
        if not math.isfinite(distance):
            raise RuntimeError("collection query returned a non-finite distance")
        if distance > router_config.max_distance[corpus]:
            continue
        candidates.append({"entry_id": int(entry_id), "distance": round(float(distance), 4)})
    return candidates


def _fetch_candidates(corpus: str, raw: list[dict]) -> list[Candidate]:
    if not raw:
        return []
    ids = [item["entry_id"] for item in raw]
    placeholders = ",".join("?" for _ in ids)
    profile = CORPUS_REGISTRY[corpus]
    db = sqlite3.connect(f"file:{profile['db_path']}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(
            f"SELECT id, title, content, summary, tags, source, created_at "
            f"FROM entries WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        db.close()
    row_map = {row["id"]: row for row in rows}
    output = []
    for item in raw:
        row = row_map.get(item["entry_id"])
        if row is None:
            continue
        output.append(Candidate(
            corpus=corpus,
            entry_id=row["id"],
            title=row["title"] or "Untitled",
            content=row["content"] or None,
            summary=row["summary"] or None,
            tags=row["tags"] or None,
            source=row["source"] or None,
            date=row["created_at"],
            distance=item["distance"],
        ))
    return output


def _retrieve_corpus(
    corpus: str, embedding: list[float], router_config: RouterConfig
) -> list[Candidate]:
    return _fetch_candidates(corpus, _query_collection(corpus, embedding, router_config))


def _rerank_batch(query: str, candidates: list[Candidate], model) -> None:
    if not candidates:
        return
    pairs = [(query, (item.content or item.summary or "")[:1500]) for item in candidates]
    raw_scores = list(model.predict(pairs))
    if len(raw_scores) != len(candidates):
        raise RuntimeError("reranker score cardinality mismatch")
    relevance_scores = []
    for score in raw_scores:
        numeric = float(score)
        if not math.isfinite(numeric):
            raise RuntimeError("reranker returned a non-finite score")
        try:
            relevance = 1.0 / (1.0 + math.exp(-numeric))
        except OverflowError:
            relevance = 1.0 if numeric > 0 else 0.0
        relevance_scores.append(round(relevance, 4))
    for item, relevance in zip(candidates, relevance_scores):
        item.relevance = relevance


def _apply_decay(candidate: Candidate, router_config: RouterConfig) -> None:
    if candidate.corpus == "ai" and router_config.ai_decay_mode == "disabled":
        candidate.final_score = candidate.relevance
        return
    days_old = 0.0
    if candidate.date:
        try:
            created = datetime.fromisoformat(candidate.date)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            days_old = max((datetime.now(timezone.utc) - created).total_seconds() / 86400.0, 0)
        except (TypeError, ValueError):
            pass
    if candidate.corpus == "ai":
        half_life = router_config.ai_decay_half_life_days
        floor = router_config.ai_decay_floor
        if half_life is None or floor is None:
            raise RuntimeError("AI decay config is incomplete")
    else:
        half_life = HOMELAB_DECAY_HALF_LIFE
        floor = HOMELAB_DECAY_FLOOR
    decay = max(1.0 / (1.0 + days_old / half_life), floor)
    candidate.final_score = round(candidate.relevance * decay, 4)


def _public_url(source: Optional[str]) -> Optional[str]:
    if not source:
        return None
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return source
    return None


def _result(candidate: Candidate) -> SearchResultV2:
    public_source_url = _public_url(candidate.source)
    return SearchResultV2(
        corpus=candidate.corpus,
        entry_id=candidate.entry_id,
        ref=f"{candidate.corpus}:{candidate.entry_id}",
        title=candidate.title,
        content=candidate.content or candidate.summary,
        tags=candidate.tags,
        public_source_url=public_source_url,
        link=public_source_url or f"kb://{candidate.corpus}/{candidate.entry_id}",
        distance=candidate.distance,
        relevance=candidate.relevance,
        final_score=candidate.final_score,
    )


def _empty_corpora() -> dict[str, CorpusResultsV2]:
    return {
        name: CorpusResultsV2(searched=False, available=False, count=0, results=[])
        for name in CORPUS_REGISTRY
    }


def _audit(event: str, **fields) -> None:
    safe = {"event": event, **fields}
    print("[v2-audit] " + json.dumps(safe, sort_keys=True, separators=(",", ":")), flush=True)


def create_v2_app(
    embed: Callable[[str], list[float]],
    reranker: Callable[[], object],
) -> FastAPI:
    try:
        client_snapshot = _load_clients()
    except HTTPException:
        client_snapshot = None
    try:
        router_snapshot = _load_router_config()
    except RuntimeError:
        router_snapshot = None

    def authorize_snapshot(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    ) -> AuthorizedClient:
        if client_snapshot is None:
            raise HTTPException(
                status_code=503,
                detail="V2 client authorization is unavailable",
            )
        return _authorize_credentials(credentials, client_snapshot)

    v2 = FastAPI(
        title="KB multi-corpus API",
        version="2.0.0",
        docs_url=None,
        redoc_url=None,
    )

    @v2.get("/health", response_model=HealthResponseV2)
    def health_v2(client: AuthorizedClient = Depends(authorize_snapshot)):
        model = reranker()
        router_ready = router_snapshot is not None
        corpora = {
            corpus: (
                _corpus_health(corpus, model is not None)
                if router_ready
                else CorpusHealthV2(
                    ready=False,
                    collection=CORPUS_REGISTRY[corpus]["collection"],
                    reason="router_config_unavailable",
                )
            )
            for corpus in sorted(client.allowed_corpora)
        }
        status = "ok" if all(item.ready for item in corpora.values()) else "degraded"
        return HealthResponseV2(status=status, corpora=corpora)

    @v2.post("/kb/search", response_model=SearchResponseV2)
    def search_v2(
        request: SearchRequestV2,
        client: AuthorizedClient = Depends(authorize_snapshot),
    ):
        started = time.perf_counter()
        if request.scope not in client.allowed_scopes:
            raise HTTPException(status_code=403, detail="Scope is not allowed for this client")
        if router_snapshot is None:
            raise HTTPException(
                status_code=503,
                detail={"reason": "router_config_unavailable"},
            )
        router_config = router_snapshot
        if request.scope == "auto":
            raise HTTPException(
                status_code=409,
                detail={"reason": "auto_routing_not_enabled"},
            )
        requested = [request.scope] if request.scope in CORPUS_REGISTRY else ["homelab", "ai"]
        model = reranker()
        health = {name: _corpus_health(name, model is not None) for name in requested}
        degraded = [name for name, value in health.items() if not value.ready]
        if degraded and (request.scope != "both" or not request.allow_degraded):
            _audit("search", client=client.name, scope=request.scope, status=503, degraded=degraded)
            raise HTTPException(
                status_code=503,
                detail={"reason": "required_corpus_unavailable", "corpora": degraded},
            )
        searchable = [name for name in requested if health[name].ready]
        if not searchable:
            raise HTTPException(status_code=503, detail={"reason": "all_corpora_unavailable"})

        try:
            embedding = embed(request.query)
        except Exception as exc:
            _audit("search", client=client.name, scope=request.scope, status=503, failure="retrieval")
            raise HTTPException(status_code=503, detail={"reason": "retrieval_unavailable"}) from exc
        candidates = []
        retrieval_failures = []
        with ThreadPoolExecutor(max_workers=len(searchable)) as executor:
            futures = {
                executor.submit(_retrieve_corpus, corpus, embedding, router_config): corpus
                for corpus in searchable
            }
            for future in as_completed(futures):
                corpus = futures[future]
                try:
                    candidates.extend(future.result())
                except Exception:
                    retrieval_failures.append(corpus)
        if retrieval_failures:
            if request.scope != "both" or not request.allow_degraded:
                _audit(
                    "search",
                    client=client.name,
                    scope=request.scope,
                    status=503,
                    failure="retrieval",
                    degraded=sorted(retrieval_failures),
                )
                raise HTTPException(
                    status_code=503,
                    detail={
                        "reason": "required_corpus_unavailable",
                        "corpora": sorted(retrieval_failures),
                    },
                )
            degraded.extend(name for name in retrieval_failures if name not in degraded)
            searchable = [name for name in searchable if name not in retrieval_failures]
            candidates = [item for item in candidates if item.corpus in searchable]
        if not searchable:
            raise HTTPException(status_code=503, detail={"reason": "all_corpora_unavailable"})
        degraded = [name for name in CORPUS_REGISTRY if name in degraded]
        try:
            _rerank_batch(request.query, candidates, model)
        except Exception as exc:
            _audit("search", client=client.name, scope=request.scope, status=503, failure="rerank")
            raise HTTPException(status_code=503, detail={"reason": "reranker_unavailable"}) from exc

        by_corpus: dict[str, list[Candidate]] = {name: [] for name in CORPUS_REGISTRY}
        for candidate in candidates:
            if candidate.relevance < router_config.reject_threshold:
                continue
            _apply_decay(candidate, router_config)
            by_corpus[candidate.corpus].append(candidate)
        corpora = _empty_corpora()
        for corpus in searchable:
            ranked = sorted(by_corpus[corpus], key=lambda item: -item.final_score)[:request.top_k]
            results = [_result(item) for item in ranked]
            corpora[corpus] = CorpusResultsV2(
                searched=True,
                available=True,
                count=len(results),
                results=results,
            )
        for corpus in degraded:
            corpora[corpus] = CorpusResultsV2(
                searched=False,
                available=False,
                count=0,
                results=[],
            )

        selected: SelectedScope
        if request.scope == "both" and len(searchable) == 1:
            selected = searchable[0]  # type: ignore[assignment]
        else:
            selected = request.scope  # type: ignore[assignment]
        reason = "degraded_explicit_both" if degraded else "explicit_scope"
        response_value = SearchResponseV2(
            query=request.query,
            requested_scope=request.scope,
            selected_scope=selected,
            routing_mode="explicit",
            routing_reason=reason,
            needs_clarification=False,
            router_version=None,
            degraded_corpora=degraded,
            total_count=sum(item.count for item in corpora.values()),
            corpora=CorporaV2(**corpora),
        )
        _audit(
            "search",
            client=client.name,
            scope=request.scope,
            selected_scope=selected,
            status=200,
            degraded=degraded,
            homelab_count=corpora["homelab"].count,
            ai_count=corpora["ai"].count,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response_value

    return v2
