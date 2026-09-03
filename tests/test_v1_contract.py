from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import kb_search_api
from kb_search_api import SearchResult


FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def golden_result(entry_id: int = 42) -> SearchResult:
    return SearchResult(
        id=entry_id,
        title="Golden entry",
        content="Golden content",
        summary="Golden summary",
        tags="alpha,beta",
        source="https://example.invalid/source",
        date="2026-08-02",
        distance=0.1234,
        relevance=0.8,
        final_score=0.7,
    )


class V1ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = kb_search_api.create_root_app(v1_enabled=True)
        self.client = TestClient(self.app)

    def test_full_response_fixture(self) -> None:
        with patch("kb_search_api._do_search", return_value=[golden_result()]):
            response = self.client.post("/kb/search", json={"query": "golden"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), fixture("v1_full_response.json"))

    def test_websearch_and_alias_response_fixture(self) -> None:
        expected = fixture("v1_websearch_response.json")
        with patch("kb_search_api._do_search", return_value=[golden_result()]):
            search = self.client.post(
                "/kb/search", json={"query": "golden", "format": "websearch"}
            )
            alias = self.client.post("/kb/websearch", json={"query": "golden"})
        self.assertEqual(search.status_code, 200)
        self.assertEqual(alias.status_code, 200)
        self.assertEqual(search.json(), expected)
        self.assertEqual(alias.json(), expected)

    def test_caps_and_legacy_permissive_parameters_are_frozen(self) -> None:
        results = [golden_result(entry_id) for entry_id in range(1, 8)]
        with patch("kb_search_api._do_search", return_value=results):
            full = self.client.post(
                "/kb/search",
                json={"query": "golden", "format": "unknown", "scope": "ai"},
            )
            web = self.client.post(
                "/kb/search", json={"query": "golden", "format": "websearch"}
            )
        self.assertEqual(full.status_code, 200)
        self.assertEqual(full.json()["count"], 5)
        self.assertEqual(len(full.json()["results"]), 5)
        self.assertEqual(len(web.json()), 3)

    def test_root_openapi_surface_fixture(self) -> None:
        schema = self.app.openapi()
        projection = {
            "openapi": schema["openapi"],
            "health": {
                "method": "get",
                "response_schema": schema["paths"]["/health"]["get"]
                ["responses"]["200"]["content"]["application/json"]["schema"],
            },
            "search": {
                "method": "post",
                "request_schema": schema["paths"]["/kb/search"]["post"]
                ["requestBody"]["content"]["application/json"]["schema"]["$ref"],
                "response_schema": schema["paths"]["/kb/search"]["post"]
                ["responses"]["200"]["content"]["application/json"]["schema"],
            },
            "websearch": {
                "method": "post",
                "request_schema": schema["paths"]["/kb/websearch"]["post"]
                ["requestBody"]["content"]["application/json"]["schema"]["$ref"],
                "response_schema": schema["paths"]["/kb/websearch"]["post"]
                ["responses"]["200"]["content"]["application/json"]["schema"],
            },
            "search_request_schema": schema["components"]["schemas"]["SearchRequest"],
        }
        self.assertEqual(projection, fixture("v1_openapi_surface.json"))
        self.assertNotIn("/v2", " ".join(schema["paths"]))
        self.assertNotIn("/kb/synthesize/nexus-relevance", schema["paths"])


class V1RerankerFailClosedTests(unittest.TestCase):
    """v1 used to fall back to `1.0 - distance` whenever the cross-encoder was
    missing, then apply RERANK_THRESHOLD — a cutoff tuned for cross-encoder
    sigmoid scores — to that cosine-distance scale. Same number, different
    meaning, no signal to the caller. v2 already answers 503; v1 now matches."""

    def setUp(self) -> None:
        self.client = TestClient(kb_search_api.create_root_app(v1_enabled=True))

    def test_rerank_raises_when_model_missing(self) -> None:
        with patch.object(kb_search_api, "rerank_model", None):
            with self.assertRaises(kb_search_api.RerankerUnavailable):
                kb_search_api._rerank("query", [golden_result()])

    def test_rerank_raises_when_model_errors(self) -> None:
        class Exploding:
            def predict(self, pairs):
                raise RuntimeError("boom")

        with patch.object(kb_search_api, "rerank_model", Exploding()):
            with self.assertRaises(kb_search_api.RerankerUnavailable):
                kb_search_api._rerank("query", [golden_result()])

    def test_search_answers_503_when_reranker_unavailable(self) -> None:
        with patch.object(kb_search_api, "rerank_model", None), \
             patch("kb_search_api.embed_query", return_value=[0.0]), \
             patch("kb_search_api.query_chromadb", return_value={}), \
             patch("kb_search_api.fetch_metadata", return_value=[golden_result()]):
            response = self.client.post("/kb/search", json={"query": "golden"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["reason"], "reranker_unavailable")

    def test_empty_recall_still_answers_200(self) -> None:
        """Nothing recalled means nothing to rerank; that must not become a 503."""
        with patch.object(kb_search_api, "rerank_model", None), \
             patch("kb_search_api.embed_query", return_value=[0.0]), \
             patch("kb_search_api.query_chromadb", return_value={}), \
             patch("kb_search_api.fetch_metadata", return_value=[]):
            response = self.client.post("/kb/search", json={"query": "golden"})
        self.assertEqual(response.status_code, 200)


class V1RetirementSwitchTests(unittest.TestCase):
    def test_strict_flag_parser_defaults_disabled(self) -> None:
        self.assertFalse(kb_search_api._parse_v1_search_enabled(None))
        self.assertTrue(kb_search_api._parse_v1_search_enabled("true"))
        self.assertTrue(kb_search_api._parse_v1_search_enabled(" TRUE "))
        self.assertFalse(kb_search_api._parse_v1_search_enabled("false"))
        for value in ("0", "yes", "tru"):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, value):
                kb_search_api._parse_v1_search_enabled(value)

    def test_bilingual_union_flag_parser_defaults_disabled(self) -> None:
        self.assertFalse(kb_search_api._parse_bilingual_union_enabled(None))
        self.assertTrue(kb_search_api._parse_bilingual_union_enabled("true"))
        self.assertTrue(kb_search_api._parse_bilingual_union_enabled(" TRUE "))
        self.assertFalse(kb_search_api._parse_bilingual_union_enabled("false"))
        for value in ("0", "yes", "tru"):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, value):
                kb_search_api._parse_bilingual_union_enabled(value)

    def test_disabled_routes_always_answer_410_and_are_hidden_from_schema(self) -> None:
        app = kb_search_api.create_root_app(v1_enabled=False)
        client = TestClient(app)
        for path in ("/kb/search", "/kb/websearch"):
            for kwargs in (
                {},
                {"content": b"not-json", "headers": {"content-type": "application/json"}},
                {"json": {"unexpected": True}},
            ):
                with self.subTest(path=path, kwargs=kwargs):
                    response = client.post(path, **kwargs)
                    self.assertEqual(response.status_code, 410)
                    self.assertEqual(response.json(), {"detail": "KB Search v1 is retired"})
        self.assertEqual(
            sorted(app.openapi()["paths"]),
            fixture("v1_disabled_openapi_surface.json")["paths"],
        )

    def test_synthesis_still_works_when_v1_is_disabled(self) -> None:
        app = kb_search_api.create_root_app(v1_enabled=False)
        client = TestClient(app)
        request = {
            "query": "query",
            "source_type": "video",
            "video_title": "Title",
        }
        with patch("kb_search_api._authorize_synthesis"), patch(
            "kb_search_api._do_search", return_value=[]
        ):
            response = client.post("/kb/synthesize/nexus-relevance", json=request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "not_confirmed")


if __name__ == "__main__":
    unittest.main()
