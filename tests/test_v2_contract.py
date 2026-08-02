from __future__ import annotations

import json
import os
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
        ]
        for value in cases:
            with self.subTest(value=value):
                response = self.client.post("/kb/search", headers=self.headers, json=value)
                self.assertEqual(response.status_code, 422)

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
            side_effect=lambda _corpus, _embedding, config: seen.append(config.candidate_k) or [],
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
            with self.subTest(value=value), patch(
                "kb_v2.httpx.post", return_value=httpx.Response(200, json=value)
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
        self.assertEqual(set(generated_result["required"]), set(frozen_result["required"]))
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


if __name__ == "__main__":
    unittest.main()
