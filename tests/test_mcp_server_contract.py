from __future__ import annotations

import runpy
import subprocess
import sys
import types
import unittest
from unittest.mock import patch


SERVER_PATH = "/home/turok/projects/kb-mcp/mcp_server.py"


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
        self.assertEqual(server.tools, ["semantic_search", "add"])
        self.assertEqual(
            server.kwargs,
            {"host": "127.0.0.1", "port": 8000, "log_level": "WARNING"},
        )

        completed = subprocess.CompletedProcess([], 0, stdout="answer", stderr="")
        with patch("subprocess.run", return_value=completed) as process:
            self.assertEqual(namespace["semantic_search"]("query"), "answer")
        process.assert_called_once_with(
            ["/usr/local/bin/kb", "ask", "query"],
            capture_output=True,
            text=True,
            timeout=180,
        )

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
