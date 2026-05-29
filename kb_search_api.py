#!/usr/bin/env python3
"""FastAPI service: POST /kb/search — semantic search over kb_collection with cross-encoder reranking."""

import json, sqlite3, socket, math, os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# --- config ---
EMBED_SOCKET = "/run/kb-embed/embed.sock"
CHROMA_BASE = "http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database"
CHROMA_COLLECTION = "kb_collection"
DB_PATH = "/opt/kb/kb.db"
MAX_DISTANCE = 0.40          # cosine floor — discard obvious noise
N_RESULTS = 25               # broad recall before reranking (was 10)
RERANK_THRESHOLD = 0.5       # initial relevance threshold, calibrated after testing
DECAY_HALF_LIFE = 540.0      # days (~1.5yr) — conservative for homelab technical docs
DECAY_FLOOR = 0.3            # minimum decay multiplier (entry never drops below 30%)
TOP_FULL = 5                 # cap for full format
TOP_WEBSEARCH = 3            # cap for Open WebUI format
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- global model reference (loaded at startup) ---
rerank_model = None


# --- sigmoid: map raw cross-encoder score to [0, 1] relevance ---
def _sigmoid(x: float) -> float:
    # ms-marco typically outputs in [-10, 10]; sigmoid centers at ~0.5 for x≈0
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 1.0 if x > 0 else 0.0


# --- fastapi with lifespan (model load + warmup + graceful shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rerank_model
    print("[startup] Loading cross-encoder model...", flush=True)

    try:
        from sentence_transformers import CrossEncoder
        rerank_model = CrossEncoder(RERANK_MODEL)
        # Warmup with a dummy pair so first real query isn't slow due to lazy init
        _ = rerank_model.predict([("warmup query", "warmup document")])
        print(f"[startup] Cross-encoder model loaded: {RERANK_MODEL}", flush=True)
    except Exception as e:
        print(f"[startup] WARNING: Failed to load cross-encoder model: {e}", flush=True)
        print("[startup] Reranking disabled — falling back to distance-only ranking.", flush=True)

    yield  # app runs here

    # Shutdown cleanup
    rerank_model = None
    print("[shutdown] Cross-encoder model released.", flush=True)


app = FastAPI(
    title="KB Search API",
    servers=[{"url": "http://192.168.1.174:8050", "description": "Nexus KB Search"}],
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str
    format: str = "full"  # "full" → SearchResponse, "websearch" → Open WebUI format


class SearchResult(BaseModel):
    id: int
    title: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None
    distance: float = 0.0          # original cosine distance (lower = better)
    relevance: float = 0.0         # cross-encoder relevance score [0, 1]
    final_score: float = 0.0       # relevance × decay


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    count: int


# --- embedding ---
def embed_query(text: str) -> list[float]:
    """Send text to embed daemon via Unix socket, return embedding vector."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect(EMBED_SOCKET)
    sock.sendall((text.strip() + "\n").encode())

    chunks = []
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
            if b"\n" in data:
                break
        except socket.timeout:
            break
    sock.close()

    line = b"".join(chunks).decode().strip()
    if line == "null":
        raise RuntimeError("Embed daemon returned null")
    return json.loads(line)


# --- chromadb ---
def get_collection_id() -> str:
    """Resolve collection name to UUID."""
    client = httpx.Client(timeout=10)
    resp = client.get(f"{CHROMA_BASE}/collections/{CHROMA_COLLECTION}")
    if resp.status_code != 200:
        raise RuntimeError(f"Collection lookup failed: {resp.status_code}")
    return resp.json()["id"]


def query_chromadb(embedding: list[float]) -> list[dict]:
    """Query ChromaDB with embedding, return top N results (broad recall)."""
    collection_id = get_collection_id()
    payload = {
        "query_embeddings": [embedding],
        "n_results": N_RESULTS,
        "include": ["distances", "documents", "metadatas"],
    }

    client = httpx.Client(timeout=30)
    resp = client.post(
        f"{CHROMA_BASE}/collections/{collection_id}/query",
        json=payload,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"ChromaDB error {resp.status_code}: {resp.text}")

    data = resp.json()

    if not data.get("ids") or not data["ids"][0]:
        return []

    results = []
    ids_ = data["ids"][0]
    distances = data.get("distances", [[]])[0]
    metadatas = data.get("metadatas", [[]])[0]

    for i, entry_id in enumerate(ids_):
        if entry_id.startswith("gemini_"):
            continue
        dist = distances[i] if i < len(distances) else 0
        if dist > MAX_DISTANCE:
            continue

        meta = metadatas[i] if i < len(metadatas) else {}
        results.append({
            "id": str(entry_id),
            "distance": round(dist, 4),
            "title": meta.get("title", ""),
            "tags": meta.get("tags", ""),
            "source": meta.get("raw_path", ""),
        })

    return results


# --- sqlite enrichment ---
def fetch_metadata(results: list[dict]) -> list[SearchResult]:
    """Enrich ChromaDB results with full content and metadata from SQLite."""
    if not results:
        return []

    ids = [r["id"] for r in results]
    placeholders = ",".join("?" * len(ids))

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        f"SELECT id, title, content, summary, tags, created_at FROM entries WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    db.close()

    row_map = {str(row["id"]): row for row in rows}

    enriched = []
    for r in results:
        row = row_map.get(r["id"])
        if not row:
            # ChromaDB has it but SQLite doesn't (unlikely — handle gracefully)
            enriched.append(SearchResult(
                id=int(r["id"]),
                title=r["title"],
                tags=r["tags"],
                distance=r["distance"],
            ))
            continue
        enriched.append(SearchResult(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            summary=row["summary"],
            tags=row["tags"] or r.get("tags", ""),
            source=r.get("source", ""),
            date=row["created_at"],
            distance=r["distance"],
        ))

    # Sort by distance (ChromaDB already sorts, defensive)
    enriched.sort(key=lambda x: x.distance)
    return enriched


# --- cross-encoder reranking ---
def _rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Rerank results using cross-encoder for semantic relevance.
    Operates on chunk content (first 500 chars to respect 512-token limit).
    Falls back to distance-sort if model not available."""
    if rerank_model is None or not results:
        return results

    try:
        pairs = [(query, (r.content or r.summary or "")[:500]) for r in results]
        raw_scores = rerank_model.predict(pairs)

        for i, score in enumerate(raw_scores):
            results[i].relevance = round(_sigmoid(float(score)), 4)

        results.sort(key=lambda x: -x.relevance)
        return results

    except Exception as e:
        print(f"[rerank] Cross-encoder error: {e}, falling back to distance sort", flush=True)
        results.sort(key=lambda x: x.distance)
        return results


# --- dedup: keep best chunk per document ---
def _dedup(results: list[SearchResult]) -> list[SearchResult]:
    """If multiple chunks from the same entry exist, keep only the highest-scoring one."""
    seen = {}
    for r in results:
        if r.id not in seen or r.relevance > seen[r.id].relevance:
            seen[r.id] = r
    return list(seen.values())


# --- time decay (applied AFTER rerank as recency correction) ---
def _apply_decay(results: list[SearchResult]) -> list[SearchResult]:
    """Multiply relevance by rational time decay: final = relevance × max(1/(1+days/540), 0.3)"""
    if not results:
        return results

    now = datetime.now(timezone.utc)
    for r in results:
        days_old = 0.0
        if r.date:
            try:
                created = datetime.fromisoformat(r.date)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days_old = (now - created).total_seconds() / 86400.0
            except (ValueError, TypeError):
                pass
        decay = 1.0 / (1.0 + max(days_old, 0) / DECAY_HALF_LIFE)
        decay = max(decay, DECAY_FLOOR)
        r.final_score = round(r.relevance * decay, 4)

    results.sort(key=lambda x: -x.final_score)
    return results


# --- full pipeline ---
def _do_search(query: str) -> list[SearchResult]:
    """Layer 1: ChromaDB recall → Layer 2: cross-encoder rerank → dedup → Layer 3: decay → Layer 4: cutoff"""
    embedding = embed_query(query)
    raw = query_chromadb(embedding)
    results = fetch_metadata(raw)

    if not results:
        return []

    # Layer 2: cross-encoder reranking (or fallback to distance sort)
    results = _rerank(query, results)

    # Dedup: keep best chunk per entry
    results = _dedup(results)

    # Layer 3: time decay as recency correction
    results = _apply_decay(results)

    # Layer 4: threshold filter + cap (no fallback — empty is honest)
    passed = [r for r in results if r.final_score >= RERANK_THRESHOLD]
    return passed


def _to_websearch(results: list[SearchResult]) -> list[dict]:
    """Convert SearchResult list to Open WebUI websearch format."""
    return [
        {
            "title": r.title or "Untitled",
            "snippet": (r.content or r.summary or "")[:3000],
            "link": r.source or f"kb://{r.id}",
        }
        for r in results
    ]


# --- endpoints ---
@app.post("/kb/search")
async def kb_search(req: SearchRequest):
    """Semantic search over KB. format=full (default) or format=websearch for Open WebUI."""
    results = _do_search(req.query)

    cap = TOP_WEBSEARCH if req.format == "websearch" else TOP_FULL
    if len(results) > cap:
        results = results[:cap]

    if req.format == "websearch":
        return _to_websearch(results)
    return SearchResponse(
        results=results,
        query=req.query,
        count=len(results),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "rerank_model": RERANK_MODEL if rerank_model is not None else "unavailable"}


@app.post("/kb/websearch")
async def kb_websearch(req: SearchRequest):
    """[DEPRECATED] Use /kb/search with {"format": "websearch"}."""
    results = _do_search(req.query)
    if len(results) > TOP_WEBSEARCH:
        results = results[:TOP_WEBSEARCH]
    return _to_websearch(results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
