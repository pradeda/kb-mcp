#!/opt/kb/venv/bin/python3
import os
import sys
import subprocess

from mcp.server.fastmcp import FastMCP

# SSE transport — host/port passed to FastMCP constructor
# (env vars are overridden by explicit params, so we pass them directly)
_sse_mode = "--sse" in sys.argv
_host = "0.0.0.0" if _sse_mode else "127.0.0.1"
_port = 9100 if _sse_mode else 8000

mcp = FastMCP("kb", host=_host, port=_port, log_level="WARNING")


@mcp.tool(
    description=(
        "MUST be called before any research, implementation, debugging, or configuration task. "
        "Search the homelab knowledge base using semantic search (ChromaDB + embeddings). "
        "Returns relevant entries about homelab infrastructure, services, and past work."
    )
)
def kb_search(query: str) -> str:
    result = subprocess.run(
        ["/usr/local/bin/kb", "ask", query],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout or result.stderr or "No results."


@mcp.tool(
    description=(
        "Add a note to the homelab knowledge base. "
        "Use for documenting solutions, gotchas, config changes, or any homelab knowledge worth preserving. "
        "Content is passed via stdin to support multi-line text safely."
    )
)
def kb_add(content: str, title: str, tag: str) -> str:
    result = subprocess.run(
        ["/usr/local/bin/kb", "add", "note", "-", title, tag],
        input=content,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout or result.stderr or "Added successfully."


if __name__ == "__main__":
    if "--sse" in sys.argv:
        mcp.run(transport="sse")
    else:
        mcp.run()
