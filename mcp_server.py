#!/opt/kb/venv/bin/python3
import os
import sys
import subprocess
from typing import Annotated, Literal

import httpx
from pydantic import Field

from mcp.server.fastmcp import FastMCP

# SSE / Streamable HTTP transport — host/port passed to FastMCP constructor
# (env vars are overridden by explicit params, so we pass them directly)
_sse_mode   = "--sse"  in sys.argv
_http_mode  = "--http" in sys.argv
_remote     = _sse_mode or _http_mode
_host       = "0.0.0.0" if _remote else "127.0.0.1"
_port       = 9101 if _http_mode else (9100 if _sse_mode else 8000)

mcp = FastMCP("kb", host=_host, port=_port, log_level="WARNING")


def _load_local_v2_token() -> None:
    if os.getenv("KB_V2_TOKEN_MCP_LOCAL"):
        return
    try:
        with open("/opt/kb/.env", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line.startswith("KB_V2_TOKEN_MCP_LOCAL="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        os.environ["KB_V2_TOKEN_MCP_LOCAL"] = value
                    return
    except OSError:
        return


@mcp.tool(
    description=(
        "MUST be called before any research, implementation, debugging, or configuration task. "
        "Search the homelab knowledge base using semantic search (ChromaDB + embeddings). "
        "Returns relevant entries about homelab infrastructure, services, and past work."
    )
)
def semantic_search(query: str) -> str:
    # 180s: kb ask = search API retries + OpenRouter synthesis (60s client + 429 retry)
    try:
        result = subprocess.run(
            ["/usr/local/bin/kb", "ask", query],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return "KB search timed out — OpenRouter/search API slow or unavailable. Retry or narrow the query."
    return result.stdout or result.stderr or "No results."


@mcp.tool(
    description=(
        "Search the structured Homelab and AI knowledge corpora without an internal LLM call. "
        "Choose an explicit scope (homelab, ai, both, or auto); auto is unavailable until its "
        "calibrated router is enabled. Returns corpus-qualified references and grouped results."
    )
)
def corpus_search(
    query: str,
    scope: Literal["homelab", "ai", "both", "auto"] = "auto",
    top_k: Annotated[int, Field(ge=1, le=5)] = 5,
) -> dict:
    _load_local_v2_token()
    token = os.getenv("KB_V2_TOKEN_MCP_LOCAL")
    if not token:
        raise RuntimeError("KB corpus search is not configured")
    endpoint = os.getenv("KB_V2_SEARCH_URL", "http://127.0.0.1:8050/v2/kb/search")
    try:
        response = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            json={
                "query": query,
                "scope": scope,
                "top_k": top_k,
                "allow_degraded": False,
            },
            timeout=45,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError("KB corpus search timed out") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError("KB corpus search is unavailable") from exc
    if response.status_code != 200:
        reason = "unknown"
        try:
            detail = response.json().get("detail")
            if isinstance(detail, dict):
                reason = str(detail.get("reason", "unknown"))
            elif isinstance(detail, str):
                reason = detail
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"KB corpus search failed (HTTP {response.status_code}: {reason})")
    try:
        value = response.json()
    except ValueError as exc:
        raise RuntimeError("KB corpus search returned invalid JSON") from exc
    if not isinstance(value, dict) or "corpora" not in value:
        raise RuntimeError("KB corpus search returned an invalid response")
    return value


@mcp.tool(
    description=(
        "Add a note to the homelab knowledge base. "
        "Use for documenting solutions, gotchas, config changes, or any homelab knowledge worth preserving. "
        "Content is passed via stdin to support multi-line text safely."
    )
)
def add(content: str, title: str, tag: str) -> str:
    try:
        result = subprocess.run(
            ["/usr/local/bin/kb", "add", "note", "-", title, tag],
            input=content,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return "KB add timed out — check kb-embed/SQLite (WAL lock?)."
    return result.stdout or result.stderr or "Added successfully."


if __name__ == "__main__":
    if "--sse" in sys.argv:
        mcp.run(transport="sse")
    elif "--http" in sys.argv:
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
