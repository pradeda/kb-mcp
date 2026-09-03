from __future__ import annotations

import runpy
import sys
import types
import unittest
import os
from unittest.mock import patch

import httpx


SERVER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "mcp_server.py")


class FakeFastMCP:
    instances: list["FakeFastMCP"] = []

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs
        self.tools: list[str] = []
        self.run_calls: list[dict] = []
        self.__class__.instances.append(self)

    def tool(self, **_kwargs):
        def decorate(function):
            self.tools.append(function.__name__)
            return function

        return decorate

    def run(self, **kwargs) -> None:
        self.run_calls.append(kwargs)


def fake_mcp_modules() -> dict[str, types.ModuleType]:
    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP
    return {
        "mcp": mcp_module,
        "mcp.server": server_module,
        "mcp.server.fastmcp": fastmcp_module,
    }


class MCPServerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeFastMCP.instances.clear()

    def load_server(self, *arguments: str, run_name: str = "mcp_contract") -> dict:
        with patch.dict(sys.modules, fake_mcp_modules()), patch.object(
            sys, "argv", [SERVER_PATH, *arguments]
        ):
            return runpy.run_path(SERVER_PATH, run_name=run_name)

    def test_tool_names_and_semantic_search_backend_are_stable(self) -> None:
        namespace = self.load_server()
        server = FakeFastMCP.instances[-1]
        self.assertEqual(server.tools, ["semantic_search", "corpus_search", "add"])
        self.assertEqual(
            server.kwargs,
            {"host": "127.0.0.1", "port": 8000, "log_level": "WARNING"},
        )

        # semantic_search reads both corpora exclusively through the v2 plane.
        response = httpx.Response(
            200,
            json={
                "corpora": {
                    "homelab": {"searched": True, "results": [
                        {"ref": "homelab:1", "title": "T", "tags": "a", "content": "C"}
                    ]},
                    "ai": {"searched": True, "results": []},
                }
            },
        )
        with patch.dict(os.environ, {"KB_V2_TOKEN_MCP_LOCAL": "secret"}), patch(
            "httpx.post", return_value=response
        ) as post:
            value = namespace["semantic_search"]("query", "upit", "sr")
        _, kwargs = post.call_args
        self.assertEqual(kwargs["json"]["scope"], "both")
        self.assertEqual(kwargs["json"]["query"], "query")
        self.assertEqual(kwargs["json"]["query_alt"], "upit")
        self.assertEqual(kwargs["json"]["query_alt_language"], "sr")
        self.assertIn("homelab:1", value)

    def test_semantic_search_reports_v2_failure_without_subprocess_fallback(self) -> None:
        namespace = self.load_server()
        with patch.dict(os.environ, {"KB_V2_TOKEN_MCP_LOCAL": "secret"}), patch(
            "httpx.post", side_effect=httpx.ConnectError("down")
        ), patch("subprocess.run") as process:
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                namespace["semantic_search"]("query")
        process.assert_not_called()

    def test_corpus_search_calls_v2_with_explicit_non_degraded_scope(self) -> None:
        namespace = self.load_server()
        response = httpx.Response(
            200,
            json={"corpora": {"homelab": {}, "ai": {}}, "total_count": 0},
        )
        with patch.dict(os.environ, {"KB_V2_TOKEN_MCP_LOCAL": "secret"}), patch(
            "httpx.post", return_value=response
        ) as post:
            value = namespace["corpus_search"]("query", "both", 3)
        self.assertEqual(value["total_count"], 0)
        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer secret"})
        self.assertEqual(
            kwargs["json"],
            {"query": "query", "scope": "both", "top_k": 3, "allow_degraded": False},
        )
        self.assertEqual(kwargs["timeout"], 45)

    def test_corpus_search_omits_optional_alternate_fields(self) -> None:
        namespace = self.load_server()
        response = httpx.Response(200, json={"corpora": {}, "total_count": 0})
        with patch.dict(os.environ, {"KB_V2_TOKEN_MCP_LOCAL": "secret"}), patch(
            "httpx.post", return_value=response
        ) as post:
            namespace["corpus_search"]("query", "ai", 2)
        payload = post.call_args.kwargs["json"]
        self.assertNotIn("query_alt", payload)
        self.assertNotIn("query_alt_language", payload)

    def test_corpus_search_failure_contracts(self) -> None:
        namespace = self.load_server()
        with patch.dict(os.environ, {"KB_V2_TOKEN_MCP_LOCAL": "secret"}):
            with patch("httpx.post", side_effect=httpx.ReadTimeout("slow")):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    namespace["corpus_search"]("query", "homelab", 5)
            for status in (401, 403, 409, 422, 503):
                response = httpx.Response(status, json={"detail": {"reason": "test_reason"}})
                with self.subTest(status=status), patch("httpx.post", return_value=response):
                    with self.assertRaisesRegex(RuntimeError, f"HTTP {status}: test_reason"):
                        namespace["corpus_search"]("query", "homelab", 5)

    def test_stdio_transport_contract(self) -> None:
        self.load_server(run_name="__main__")
        server = FakeFastMCP.instances[-1]
        self.assertEqual(server.kwargs["host"], "127.0.0.1")
        self.assertEqual(server.kwargs["port"], 8000)
        self.assertEqual(server.run_calls, [{}])

    def test_sse_transport_contract(self) -> None:
        self.load_server("--sse", run_name="__main__")
        server = FakeFastMCP.instances[-1]
        self.assertEqual(server.kwargs["host"], "0.0.0.0")
        self.assertEqual(server.kwargs["port"], 9100)
        self.assertEqual(server.run_calls, [{"transport": "sse"}])

    def test_streamable_http_transport_contract(self) -> None:
        self.load_server("--http", run_name="__main__")
        server = FakeFastMCP.instances[-1]
        self.assertEqual(server.kwargs["host"], "0.0.0.0")
        self.assertEqual(server.kwargs["port"], 9101)
        self.assertEqual(server.run_calls, [{"transport": "streamable-http"}])


if __name__ == "__main__":
    unittest.main()
