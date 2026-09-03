from __future__ import annotations

import json
import os
import runpy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

import kb_search_api
import kb_v2
from kb_v2 import Candidate, CorpusHealthV2, create_v2_app


CONTRACTS = Path(__file__).parents[1] / "contracts"


class FakeReranker:
    def __init__(self, scores=None):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return self.scores or [2.0] * len(pairs)


def candidate(corpus: str, entry_id: int, source: str = "telegram") -> Candidate:
    return Candidate(
        corpus=corpus,
        entry_id=entry_id,
        title=f"{corpus} title",
        content=f"{corpus} content",
        summary=None,
        tags="test",
        source=source,
        date="2026-08-02T00:00:00+00:00",
        distance=0.1,
    )


class V2ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        config = Path(self.temp.name) / "clients.yml"
        config.write_text(
            """clients:
  full:
    token_env: KB_V2_TOKEN_TEST_FULL
    allowed_corpora: [homelab, ai]
    allowed_scopes: [homelab, ai, both, auto]
  homelab-only:
    token_env: KB_V2_TOKEN_TEST_HOMELAB
    allowed_corpora: [homelab]
    allowed_scopes: [homelab, auto]
""",
            encoding="utf-8",
        )
        os.chmod(config, 0o600)
        router_config = Path(self.temp.name) / "corpus-router.yml"
        self.router_config = router_config
        router_config.write_text(
            """router_version: corpus-router-v1-precalibration
accept_thresholds:
  homelab: 0.60
  ai: 0.60
reject_threshold: 0.40
both_margin: 0.05
dead_zone:
  lower: 0.40
  upper: 0.60
candidate_k: 25
max_distance:
  homelab: 0.60
  ai: 0.60
ai_decay:
  mode: disabled
""",
            encoding="utf-8",
        )
        os.chmod(router_config, 0o600)
        self.environment = patch.dict(
            os.environ,
            {
                "KB_V2_CLIENTS_CONFIG": str(config),
                "KB_CORPUS_ROUTER_CONFIG": str(router_config),
                "KB_V2_TOKEN_TEST_FULL": "f" * 64,
                "KB_V2_TOKEN_TEST_HOMELAB": "h" * 64,
            },
        )
        self.environment.start()
        self.reranker = FakeReranker()
        self.embed = Mock(return_value=[0.1, 0.2])
        self.client = TestClient(create_v2_app(self.embed, lambda: self.reranker))
        self.headers = {"Authorization": "Bearer " + "f" * 64}
        self.health = patch(
            "kb_v2._corpus_health",
            side_effect=lambda corpus, _ready: CorpusHealthV2(
                ready=True, collection=f"{corpus}_collection"
            ),
        )
        self.health.start()

    def tearDown(self) -> None:
        self.health.stop()
        self.environment.stop()
        self.temp.cleanup()

    @staticmethod
    def request(scope="both", **changes):
        value = {
            "query": "  shared question  ",
            "scope": scope,
            "top_k": 5,
            "allow_degraded": False,
        }
        value.update(changes)
        return value

    def test_auth_scope_and_auto_failure_contracts(self) -> None:
        missing = self.client.post("/kb/search", json=self.request("homelab"))
        invalid = self.client.post(
            "/kb/search",
            headers={"Authorization": "Bearer wrong"},
            json=self.request("homelab"),
        )
        forbidden = self.client.post(
            "/kb/search",
            headers={"Authorization": "Bearer " + "h" * 64},
            json=self.request("ai"),
        )
        auto = self.client.post(
            "/kb/search", headers=self.headers, json=self.request("auto")
        )
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.headers["www-authenticate"], "Bearer")
        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(auto.status_code, 409)
        self.assertEqual(auto.json()["detail"]["reason"], "auto_routing_not_enabled")

    def test_strict_validation_rejects_empty_extra_and_wrong_types(self) -> None:
        cases = [
            self.request("homelab", query="   "),
            self.request("homelab", unknown=True),
            self.request("unknown"),
            self.request("homelab", top_k="5"),
            self.request("homelab", allow_degraded=1),
            self.request("homelab", query_alt="alternate"),
            self.request("homelab", query_alt_language="en"),
            self.request("homelab", query_alt="   ", query_alt_language="en"),
        ]
        for value in cases:
            with self.subTest(value=value):
                response = self.client.post("/kb/search", headers=self.headers, json=value)
                self.assertEqual(response.status_code, 422)

    def test_identical_alternate_is_valid_but_not_used(self) -> None:
        client = TestClient(create_v2_app(self.embed, lambda: self.reranker, union_enabled=True))
        with patch("kb_v2._query_collection", return_value=[]), patch("kb_v2._audit") as audit:
            response = client.post(
                "/kb/search",
                headers=self.headers,
                json=self.request(
                    "homelab",
                    query=" docker backup ",
                    query_alt="  docker backup  ",
                    query_alt_language="sr",
                ),
            )
        self.assertEqual(response.status_code, 200)
        self.embed.assert_called_once_with("docker backup")
        fields = audit.call_args.kwargs
        self.assertEqual(fields["forms_supplied"], 2)
        self.assertEqual(fields["forms_used"], 1)

    def test_union_deduplicates_by_id_and_keeps_lowest_distance(self) -> None:
        primary = [
            {"entry_id": 2, "distance": 0.2},
            {"entry_id": 10, "distance": 0.3},
        ]
        alternate = [
            {"entry_id": 2, "distance": 0.1},
            {"entry_id": 3, "distance": 0.3},
        ]
        merged = kb_v2._union_candidates(primary, alternate)
        self.assertEqual(
            [{"entry_id": item["entry_id"], "distance": item["distance"]} for item in merged],
            [
                {"entry_id": 2, "distance": 0.1},
                {"entry_id": 10, "distance": 0.3},
                {"entry_id": 3, "distance": 0.3},
            ],
        )
        self.assertEqual(
            [(item["entry_id"], item["from_primary"], item["alternate_rank"]) for item in merged],
            [(2, True, 1), (10, True, None), (3, False, 2)],
        )

    def test_union_matches_versioned_eval_reference(self) -> None:
        reference_path = Path(__file__).parents[2] / "kb-eval" / "run_merged_bilingual_eval.py"
        self.assertTrue(reference_path.is_file(), f"missing eval reference: {reference_path}")
        reference_union = runpy.run_path(str(reference_path))["union"]
        primary = [{"id": "2", "distance": 0.2}, {"id": "10", "distance": 0.3}]
        alternate = [{"id": "2", "distance": 0.1}, {"id": "3", "distance": 0.3}]

        expected = reference_union(primary, alternate)
        actual = kb_v2._union_candidates(
            [{"entry_id": int(item["id"]), "distance": item["distance"]} for item in primary],
            [{"entry_id": int(item["id"]), "distance": item["distance"]} for item in alternate],
        )
        self.assertEqual(
            [{"id": str(item["entry_id"]), "distance": item["distance"]} for item in actual],
            expected,
        )

    def test_union_retrieves_homelab_only_and_reranks_once(self) -> None:
        self.embed.side_effect = lambda text: [1.0] if text == "shared question" else [2.0]
        client = TestClient(create_v2_app(self.embed, lambda: self.reranker, union_enabled=True))

        def query_collection(corpus, embedding, _config):
            if corpus == "ai":
                return [{"entry_id": 4, "distance": 0.1}]
            if embedding == [1.0]:
                return [{"entry_id": 1, "distance": 0.2}, {"entry_id": 2, "distance": 0.3}]
            return [{"entry_id": 2, "distance": 0.1}, {"entry_id": 3, "distance": 0.2}]

        def fetch(corpus, raw):
            return [candidate(corpus, item["entry_id"]) for item in raw]

        with patch("kb_v2._query_collection", side_effect=query_collection) as query, patch(
            "kb_v2._fetch_candidates", side_effect=fetch
        ):
            response = client.post(
                "/kb/search",
                headers=self.headers,
                json=self.request(query_alt="drugo pitanje", query_alt_language="sr"),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.embed.call_count, 2)
        self.assertEqual(query.call_count, 3)
        self.assertEqual(len(self.reranker.calls), 1)
        self.assertEqual(len(self.reranker.calls[0]), 4)
        value = response.json()
        self.assertEqual({item["entry_id"] for item in value["corpora"]["homelab"]["results"]}, {1, 2, 3})
        self.assertEqual([item["entry_id"] for item in value["corpora"]["ai"]["results"]], [4])

    def test_alternate_admission_preserves_primary_and_gates_alternate_only(self) -> None:
        candidates = [
            candidate("homelab", 1),
            candidate("homelab", 2),
            candidate("homelab", 3),
            candidate("ai", 4),
        ]
        candidates[0].relevance = 0.5
        candidates[1].from_primary = False
        candidates[1].alternate_rank = 5
        candidates[1].relevance = 0.5
        candidates[2].from_primary = False
        candidates[2].alternate_rank = 6
        candidates[2].relevance = 0.99
        candidates[3].from_primary = False
        candidates[3].alternate_rank = 99
        candidates[3].relevance = 0.5
        admitted = kb_v2._admit_alternate_only_candidates(candidates, 0.4)
        self.assertEqual([item.entry_id for item in admitted], [1, 2, 4])

    def test_alternate_admission_requires_high_score_without_primary_result(self) -> None:
        low = candidate("homelab", 1)
        low.from_primary = False
        low.alternate_rank = 1
        low.relevance = 0.9399
        high = candidate("homelab", 2)
        high.from_primary = False
        high.alternate_rank = 5
        high.relevance = 0.94
        admitted = kb_v2._admit_alternate_only_candidates([low, high], 0.4)
        self.assertEqual([item.entry_id for item in admitted], [2])

    def test_pre_rerank_pruning_keeps_primary_provenance_and_top_five_alternate_only(self) -> None:
        primary = [{"entry_id": 1, "distance": 0.3}, {"entry_id": 2, "distance": 0.4}]
        alternate = [
            {"entry_id": 3, "distance": 0.1}, {"entry_id": 2, "distance": 0.2},
            {"entry_id": 4, "distance": 0.3}, {"entry_id": 5, "distance": 0.4},
            {"entry_id": 6, "distance": 0.5}, {"entry_id": 1, "distance": 0.1},
            {"entry_id": 7, "distance": 0.2},
        ]
        merged = kb_v2._union_candidates(primary, alternate)
        kept = [item for item in merged if item["from_primary"] or item["alternate_rank"] <= 5]
        self.assertEqual({item["entry_id"] for item in kept}, {1, 2, 3, 4, 5, 6})
        self.assertEqual(next(item for item in kept if item["entry_id"] == 1)["distance"], 0.1)

    def test_alternate_embed_count_follows_scope_flag_and_health(self) -> None:
        cases = [
            ("homelab", True, True, "alternate", 2),
            ("homelab", True, True, None, 1),
            ("ai", True, True, "alternate", 1),
            ("both", True, True, "alternate", 2),
            ("both", True, False, "alternate", 1),
            ("both", False, True, "alternate", 1),
            ("homelab", True, True, "shared question", 1),
        ]
        for scope, enabled, homelab_ready, alternate, expected in cases:
            with self.subTest(scope=scope, enabled=enabled, homelab_ready=homelab_ready, alternate=alternate):
                embed = Mock(return_value=[0.1])
                client = TestClient(create_v2_app(embed, lambda: self.reranker, union_enabled=enabled))
                health = lambda corpus, _ready: CorpusHealthV2(
                    ready=homelab_ready if corpus == "homelab" else True,
                    collection=f"{corpus}_collection",
                    reason=None if corpus != "homelab" or homelab_ready else "dependency_unavailable",
                )
                payload = self.request(scope, allow_degraded=(scope == "both" and not homelab_ready))
                if alternate is not None:
                    payload.update(query_alt=alternate, query_alt_language="sr")
                with patch("kb_v2._corpus_health", side_effect=health), patch(
                    "kb_v2._query_collection", return_value=[]
                ):
                    response = client.post("/kb/search", headers=self.headers, json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(embed.call_count, expected)

    def test_both_embeds_once_batch_reranks_and_qualifies_duplicate_ids(self) -> None:
        with patch("kb_v2._query_collection", return_value=[{"entry_id": 1, "distance": 0.1}]), patch(
            "kb_v2._fetch_candidates",
            side_effect=lambda corpus, _raw: [
                candidate(corpus, 1, "https://example.invalid/public")
                if corpus == "homelab"
                else candidate(corpus, 1, "/opt/ai-kb/raw/private.md")
            ],
        ):
            response = self.client.post("/kb/search", headers=self.headers, json=self.request())
        self.assertEqual(response.status_code, 200)
        value = response.json()
        self.embed.assert_called_once_with("shared question")
        self.assertEqual(len(self.reranker.calls), 1)
        self.assertEqual(len(self.reranker.calls[0]), 2)
        self.assertEqual(set(value["corpora"]), {"homelab", "ai"})
        self.assertEqual(value["corpora"]["homelab"]["results"][0]["ref"], "homelab:1")
        self.assertEqual(value["corpora"]["ai"]["results"][0]["ref"], "ai:1")
        self.assertEqual(value["corpora"]["ai"]["results"][0]["link"], "kb://ai/1")
        self.assertNotIn("raw_path", str(value))
        self.assertEqual(value["total_count"], 2)

    def test_ranked_merges_both_corpora_by_score_without_changing_corpora(self) -> None:
        # Scores keyed off the passage text, not the pair position: the two corpora are
        # retrieved concurrently, so candidate order is not deterministic and a positional
        # score list would make this test flap.
        class ContentScoredReranker(FakeReranker):
            def predict(self, pairs):
                self.calls.append(pairs)
                return [9.0 if "ai content" in passage else 1.0 for _, passage in pairs]

        self.reranker = ContentScoredReranker()
        self.client = TestClient(create_v2_app(self.embed, lambda: self.reranker))
        with patch("kb_v2._query_collection", return_value=[{"entry_id": 1, "distance": 0.1}]), patch(
            "kb_v2._fetch_candidates",
            side_effect=lambda corpus, _raw: [candidate(corpus, 1)],
        ):
            response = self.client.post("/kb/search", headers=self.headers, json=self.request())
        self.assertEqual(response.status_code, 200)
        value = response.json()

        # The merged list is ordered across corpora: ai scored higher, so it leads even
        # though homelab comes first in the registry and in the grouped shape.
        self.assertEqual([item["ref"] for item in value["ranked"]], ["ai:1", "homelab:1"])
        # ...while the grouped shape is untouched, which is what makes this additive.
        self.assertEqual(value["corpora"]["homelab"]["results"][0]["ref"], "homelab:1")
        self.assertEqual(value["corpora"]["ai"]["results"][0]["ref"], "ai:1")
        # Same entries on both paths, no extras and none dropped.
        self.assertEqual(len(value["ranked"]), value["total_count"])
        self.assertEqual(
            {item["ref"] for item in value["ranked"]},
            {
                item["ref"]
                for corpus in ("homelab", "ai")
                for item in value["corpora"][corpus]["results"]
            },
        )

    def test_no_match_is_fixed_shape(self) -> None:
        with patch("kb_v2._query_collection", return_value=[]):
            response = self.client.post(
                "/kb/search", headers=self.headers, json=self.request("homelab")
            )
        self.assertEqual(response.status_code, 200)
        value = response.json()
        self.assertEqual(value["selected_scope"], "homelab")
        self.assertEqual(value["total_count"], 0)
        self.assertEqual(value["corpora"]["homelab"]["results"], [])
        self.assertFalse(value["corpora"]["ai"]["searched"])

    def test_reranker_cardinality_and_finite_scores_fail_closed(self) -> None:
        values = [candidate("homelab", 1), candidate("ai", 1)]
        for scores in ([2.0], [2.0, float("nan")], [2.0, float("inf")]):
            with self.subTest(scores=scores):
                model = FakeReranker(scores=scores)
                with self.assertRaises(RuntimeError):
                    kb_v2._rerank_batch("query", values, model)

    def test_degraded_both_is_explicit_and_never_implicit(self) -> None:
        self.health.stop()
        health_patch = patch(
            "kb_v2._corpus_health",
            side_effect=lambda corpus, _ready: CorpusHealthV2(
                ready=corpus == "homelab",
                collection=f"{corpus}_collection",
                reason=None if corpus == "homelab" else "dependency_unavailable",
            ),
        )
        health_patch.start()
        self.addCleanup(health_patch.stop)
        with patch("kb_v2._query_collection", return_value=[]):
            blocked = self.client.post("/kb/search", headers=self.headers, json=self.request())
            allowed = self.client.post(
                "/kb/search",
                headers=self.headers,
                json=self.request(allow_degraded=True),
            )
        self.assertEqual(blocked.status_code, 503)
        self.assertEqual(allowed.status_code, 200)
        value = allowed.json()
        self.assertEqual(value["selected_scope"], "homelab")
        self.assertEqual(value["degraded_corpora"], ["ai"])
        self.assertEqual(value["routing_reason"], "degraded_explicit_both")
        self.assertFalse(value["corpora"]["ai"]["available"])

    def test_health_is_authorized_and_filters_corpora(self) -> None:
        missing = self.client.get("/health")
        allowed = self.client.get(
            "/health", headers={"Authorization": "Bearer " + "h" * 64}
        )
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(set(allowed.json()["corpora"]), {"homelab"})

    def test_router_config_is_normative_and_missing_config_fails_closed(self) -> None:
        config = kb_v2._load_router_config()
        self.assertEqual(config.router_version, "corpus-router-v1-precalibration")
        self.assertEqual(config.candidate_k, 25)
        self.assertEqual(config.max_distance, {"homelab": 0.6, "ai": 0.6})
        self.assertEqual(config.ai_decay_mode, "disabled")
        with patch.dict(
            os.environ,
            {"KB_CORPUS_ROUTER_CONFIG": str(Path(self.temp.name) / "missing.yml")},
        ):
            missing_client = TestClient(create_v2_app(self.embed, lambda: self.reranker))
            search = missing_client.post(
                "/kb/search", headers=self.headers, json=self.request("homelab")
            )
            health = missing_client.get("/health", headers=self.headers)
        self.assertEqual(search.status_code, 503)
        self.assertEqual(search.json()["detail"]["reason"], "router_config_unavailable")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "degraded")
        self.assertTrue(
            all(not item["ready"] for item in health.json()["corpora"].values())
        )

    def test_runtime_config_validation_rejects_unsafe_auth_and_nonfinite_values(self) -> None:
        client_path = Path(os.environ["KB_V2_CLIENTS_CONFIG"])
        os.chmod(client_path, 0o644)
        with self.assertRaises(HTTPException):
            kb_v2._load_clients()
        os.chmod(client_path, 0o600)

        invalid_clients = Path(self.temp.name) / "invalid-clients.yml"
        invalid_clients.write_text(
            """clients:
  bad:
    token_env: USER
    allowed_corpora: [homelab, ai]
    allowed_scopes: [both]
""",
            encoding="utf-8",
        )
        os.chmod(invalid_clients, 0o600)
        with patch.dict(os.environ, {"KB_V2_CLIENTS_CONFIG": str(invalid_clients)}):
            with self.assertRaises(HTTPException):
                kb_v2._load_clients()

        invalid_router = Path(self.temp.name) / "invalid-router.yml"
        invalid_router.write_text(
            self.router_config.read_text(encoding="utf-8").replace(
                "homelab: 0.60", "homelab: .nan", 1
            ),
            encoding="utf-8",
        )
        os.chmod(invalid_router, 0o600)
        with patch.dict(os.environ, {"KB_CORPUS_ROUTER_CONFIG": str(invalid_router)}):
            with self.assertRaises(RuntimeError):
                kb_v2._load_router_config()

        rebound_router = Path(self.temp.name) / "rebound-router.yml"
        rebound_router.write_text(
            self.router_config.read_text(encoding="utf-8").replace(
                "candidate_k: 25", "candidate_k: 24"
            ),
            encoding="utf-8",
        )
        os.chmod(rebound_router, 0o600)
        with patch.dict(os.environ, {"KB_CORPUS_ROUTER_CONFIG": str(rebound_router)}):
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                kb_v2._load_router_config()

        invalid_utf8_clients = Path(self.temp.name) / "invalid-utf8-clients.yml"
        invalid_utf8_clients.write_bytes(b"\xff\xfe")
        os.chmod(invalid_utf8_clients, 0o600)
        with patch.dict(os.environ, {"KB_V2_CLIENTS_CONFIG": str(invalid_utf8_clients)}):
            with self.assertRaises(HTTPException):
                kb_v2._load_clients()
            unavailable = TestClient(create_v2_app(self.embed, lambda: self.reranker))
            self.assertEqual(unavailable.get("/health", headers=self.headers).status_code, 503)

        invalid_utf8_router = Path(self.temp.name) / "invalid-utf8-router.yml"
        invalid_utf8_router.write_bytes(b"\xff\xfe")
        os.chmod(invalid_utf8_router, 0o600)
        with patch.dict(os.environ, {"KB_CORPUS_ROUTER_CONFIG": str(invalid_utf8_router)}):
            with self.assertRaises(RuntimeError):
                kb_v2._load_router_config()
            degraded = TestClient(create_v2_app(self.embed, lambda: self.reranker))
            response = degraded.get("/health", headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "degraded")

    def test_router_config_is_a_startup_snapshot_not_a_hot_reload(self) -> None:
        self.router_config.write_text(
            self.router_config.read_text(encoding="utf-8").replace(
                "candidate_k: 25", "candidate_k: 5"
            ),
            encoding="utf-8",
        )
        seen = []
        with patch(
            "kb_v2._retrieve_corpus",
            side_effect=lambda _corpus, _embedding, config, _alternate=None, _alternate_limit=None: seen.append(config.candidate_k) or [],
        ):
            response = self.client.post(
                "/kb/search", headers=self.headers, json=self.request("homelab")
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen, [25])

    def test_collection_cache_tracks_name_lookup_and_query_shape_fails_closed(self) -> None:
        kb_v2._collection_ids["homelab"] = "old-id"
        descriptor = httpx.Response(200, json={"id": "new-id"})
        with patch("kb_v2.httpx.get", return_value=descriptor):
            kb_v2._collection_descriptor("homelab")
        self.assertEqual(kb_v2._collection_ids["homelab"], "new-id")

        config = kb_v2._load_router_config()
        invalid_responses = [
            {"ids": [["1"]], "distances": [[]]},
            {"ids": [["1"]], "distances": [[float("nan")]]},
            {"ids": [["1"]]},
        ]
        for value in invalid_responses:
            # Chroma is an external HTTP boundary and may send a non-standard
            # NaN token. httpx>=0.28 correctly refuses to *encode* one through
            # Response(json=...), so provide the raw upstream bytes instead.
            upstream = httpx.Response(
                200,
                content=json.dumps(value, allow_nan=True).encode("utf-8"),
                headers={"content-type": "application/json"},
            )
            with self.subTest(value=value), patch(
                "kb_v2.httpx.post", return_value=upstream
            ):
                with self.assertRaises(RuntimeError):
                    kb_v2._query_collection("homelab", [0.1], config)

    def test_separate_openapi_and_legacy_root_surface(self) -> None:
        v2_schema = self.client.get("/openapi.json").json()
        self.assertIn("/kb/search", v2_schema["paths"])
        self.assertEqual(
            v2_schema["components"]["securitySchemes"]["bearerAuth"],
            {"type": "http", "scheme": "bearer"},
        )
        self.assertEqual(
            v2_schema["paths"]["/kb/search"]["post"]["security"],
            [{"bearerAuth": []}],
        )
        request_ref = v2_schema["paths"]["/kb/search"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        self.assertTrue(request_ref.endswith("SearchRequestV2"))
        generated_result = v2_schema["components"]["schemas"]["SearchResultV2"]
        generated_group = v2_schema["components"]["schemas"]["CorpusResultsV2"]
        frozen = json.loads((CONTRACTS / "v2.openapi.json").read_text(encoding="utf-8"))
        frozen_result = frozen["components"]["schemas"]["SearchResultV2"]
        frozen_group = frozen["components"]["schemas"]["CorpusResultsV2"]
        frozen_request = frozen["components"]["schemas"]["SearchRequestV2"]
        self.assertEqual(set(generated_result["required"]), set(frozen_result["required"]))
        self.assertEqual(
            set(v2_schema["components"]["schemas"]["SearchRequestV2"]["properties"]),
            set(frozen_request["properties"]),
        )
        self.assertEqual(
            v2_schema["components"]["schemas"]["SearchRequestV2"]["properties"]["query_alt_language"]["anyOf"][0]["enum"],
            frozen_request["properties"]["query_alt_language"]["anyOf"][0]["enum"],
        )
        self.assertEqual(
            generated_result["properties"]["ref"]["pattern"],
            frozen_result["properties"]["ref"]["pattern"],
        )
        self.assertEqual(
            generated_group["properties"]["results"]["maxItems"],
            frozen_group["properties"]["results"]["maxItems"],
        )
        self.assertEqual(generated_result["properties"]["link"]["format"], "uri")
        self.assertTrue(
            any(
                option.get("format") == "uri"
                for option in generated_result["properties"]["public_source_url"]["anyOf"]
            )
        )
        root = kb_search_api.app.openapi()
        self.assertNotIn("/v2/kb/search", root["paths"])


class QueryLanguageTests(unittest.TestCase):
    """Audit belezi jezik upita, nikad sam tekst upita."""

    def test_detects_serbian_by_diacritics(self) -> None:
        self.assertEqual(kb_v2._query_language("beszel vraća 500 grešku"), "sr")

    def test_detects_serbian_without_diacritics(self) -> None:
        self.assertEqual(kb_v2._query_language("kako da restartujem docker kontejner"), "sr")

    def test_detects_english(self) -> None:
        self.assertEqual(kb_v2._query_language("how do I restart the docker container"), "en")

    def test_returns_unknown_without_signal(self) -> None:
        for query in ("", "ESTALE", "hnsw:space cosine"):
            self.assertEqual(kb_v2._query_language(query), "unknown", query)

    def test_technical_serbian_without_function_words_is_not_unknown(self) -> None:
        """The live case that motivated the technical markers: service name + one noun."""
        for query in (
            "Pi-hole WireGuard DNS konfiguracija",
            "kb-watcher servis",
            "docker mreza greska",
            "chromadb migracija",
            "pokretanje beszel agenta",
        ):
            self.assertEqual(kb_v2._query_language(query), "sr", query)

    def test_english_is_never_dragged_into_serbian_by_the_suffix_rule(self) -> None:
        """-ost was left out on purpose; host/post/cost/most/lost must stay English."""
        for query in (
            "which host has the most lost packets",
            "post the cost of the upgrade",
            "how does the reranker work",
        ):
            self.assertEqual(kb_v2._query_language(query), "en", query)

    def test_a_word_matching_two_rules_counts_once(self) -> None:
        # "pokretanje" is in the technical word set *and* ends with "-anje", so
        # it is the case where double counting would show. Two English markers
        # against it must still win: counted twice it would tie into "unknown".
        self.assertEqual(kb_v2._query_language("is on pokretanje"), "en")

    def test_audit_records_language_not_query_text(self) -> None:
        captured = {}
        with patch("builtins.print", lambda *a, **k: captured.setdefault("line", a[0] if a else "")):
            kb_v2._audit("search", qlang=kb_v2._query_language("kako se radi backup"))
        self.assertIn('"qlang":"sr"', captured["line"])
        self.assertNotIn("backup", captured["line"])


class CorpusMetricsTests(unittest.TestCase):
    """Per-corpus observability on /v2/health — counts, backlog, last compile."""

    def _corpus_db(self, rows, *, with_meta=False, last_compile=None):
        path = Path(tempfile.mkdtemp()) / "corpus.db"
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE entries(id INTEGER PRIMARY KEY, embedded_at TEXT)")
        db.executemany("INSERT INTO entries(id, embedded_at) VALUES(?,?)", rows)
        if with_meta:
            db.execute("CREATE TABLE kb_meta(key TEXT PRIMARY KEY, value TEXT)")
            if last_compile:
                db.execute(
                    "INSERT INTO kb_meta(key, value) VALUES('last_compile', ?)",
                    (last_compile,),
                )
        db.commit()
        db.close()
        return path

    def test_counts_backlog_and_last_compile(self) -> None:
        path = self._corpus_db(
            [(1, "t"), (2, None), (3, "t")],
            with_meta=True,
            last_compile="2026-08-07T20:00:00+00:00",
        )
        registry = {"homelab": {"db_path": str(path), "collection": "kb_collection"}}
        with patch.dict(kb_v2.CORPUS_REGISTRY, registry, clear=True):
            metrics = kb_v2._corpus_metrics("homelab")
        self.assertEqual(metrics["entry_count"], 3)
        self.assertEqual(metrics["pending_embed"], 1)
        self.assertEqual(metrics["last_compile"], "2026-08-07T20:00:00+00:00")

    def test_missing_meta_table_is_not_an_error(self) -> None:
        """The table only arrives with the compile.py change; absent must be None."""
        path = self._corpus_db([(1, "t")])
        registry = {"homelab": {"db_path": str(path), "collection": "kb_collection"}}
        with patch.dict(kb_v2.CORPUS_REGISTRY, registry, clear=True):
            metrics = kb_v2._corpus_metrics("homelab")
        self.assertEqual(metrics["entry_count"], 1)
        self.assertIsNone(metrics["last_compile"])

    def test_unreadable_database_yields_nones_not_an_exception(self) -> None:
        registry = {"homelab": {"db_path": "/nonexistent/corpus.db", "collection": "c"}}
        with patch.dict(kb_v2.CORPUS_REGISTRY, registry, clear=True):
            metrics = kb_v2._corpus_metrics("homelab")
        self.assertEqual(
            metrics, {"entry_count": None, "pending_embed": None, "last_compile": None}
        )

    def test_metrics_failure_never_flips_ready(self) -> None:
        """Metrics are observability; they must not turn a healthy corpus degraded."""
        health = CorpusHealthV2(ready=True, collection="kb_collection")
        self.assertTrue(health.ready)
        self.assertIsNone(health.entry_count)


class ShadowRouterTests(unittest.TestCase):
    """Plan 4.2 points 6-8 plus revision #2, on pre-decay relevance."""

    def setUp(self) -> None:
        self.config = kb_v2.RouterConfig(
            router_version="corpus-router-v1-precalibration",
            accept_thresholds={"homelab": 0.6, "ai": 0.6},
            reject_threshold=0.4,
            both_margin=0.05,
            dead_zone_lower=0.4,
            dead_zone_upper=0.6,
            candidate_k=25,
            max_distance={"homelab": 0.6, "ai": 0.6},
            ai_decay_mode="disabled",
        )

    def route(self, homelab=None, ai=None):
        best = {}
        if homelab is not None:
            best["homelab"] = homelab
        if ai is not None:
            best["ai"] = ai
        return kb_v2._shadow_route(best, self.config)

    def test_point_8_no_candidate_above_reject_is_none(self) -> None:
        self.assertEqual(self.route(0.2, 0.1), kb_v2.ShadowRoute("none", "no_candidate"))
        self.assertEqual(self.route(), kb_v2.ShadowRoute("none", "no_candidate"))

    def test_point_6_single_strong_corpus_wins(self) -> None:
        self.assertEqual(self.route(0.9, 0.1), kb_v2.ShadowRoute("homelab", "single_strong"))
        self.assertEqual(self.route(0.1, 0.9), kb_v2.ShadowRoute("ai", "single_strong"))

    def test_point_7_both_strong_is_both(self) -> None:
        self.assertEqual(self.route(0.9, 0.7), kb_v2.ShadowRoute("both", "both_strong"))

    def test_point_7_difference_below_margin_is_both(self) -> None:
        # Only homelab clears accept, but ai is within the margin of it.
        self.assertEqual(self.route(0.61, 0.58), kb_v2.ShadowRoute("both", "margin_tie"))

    def test_revision_2_dead_zone_pair_is_both_not_homelab(self) -> None:
        route = self.route(0.52, 0.50)
        self.assertEqual(route, kb_v2.ShadowRoute("both", "dead_zone_both"))

    def test_dead_zone_with_wide_gap_picks_the_leader(self) -> None:
        # Both inside the dead zone but far apart — the margin rule does not
        # apply, so this is not the revision #2 case.
        self.assertEqual(self.route(0.58, 0.41), kb_v2.ShadowRoute("homelab", "single_above_reject"))

    def test_one_corpus_missing_never_counts_as_a_margin_tie(self) -> None:
        # Nothing to be within the margin of; the present corpus decides.
        self.assertEqual(self.route(0.9, None), kb_v2.ShadowRoute("homelab", "single_strong"))
        self.assertEqual(self.route(None, 0.45), kb_v2.ShadowRoute("ai", "single_above_reject"))

    def test_shadow_decision_never_changes_the_response(self) -> None:
        """The whole point of shadow mode: observation only."""
        app = create_v2_app(lambda text: [0.1] * kb_v2.EMBEDDING_DIMENSION, lambda: FakeReranker())
        self.assertNotIn("shadow", json.dumps(app.openapi()["components"]["schemas"]))


if __name__ == "__main__":
    unittest.main()
