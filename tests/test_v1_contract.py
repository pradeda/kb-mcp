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
        self.client = TestClient(kb_search_api.app)

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
        schema = kb_search_api.app.openapi()
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


if __name__ == "__main__":
    unittest.main()
