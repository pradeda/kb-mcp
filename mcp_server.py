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


def _render_hits(items: list, header: str) -> list[str]:
    lines = [header]
    for item in items:
        lines.append(f"[{item.get('ref')}] {item.get('title')}")
        tags = item.get("tags")
        if tags:
            lines.append(f"Tags: {tags}")
        lines.append(str(item.get("content", "")).strip())
        lines.append("")
    return lines


def _format_corpus_payload(payload: dict) -> str:
    """Render the v2 response as one cross-corpus ranking, each hit naming its corpus.

    Prefers the merged `ranked` list: the reranker scores both corpora in a single
    batch, so ordering them together is the ranking the model actually produced, and
    reading it as one list is what a caller asking a question wants. Falls back to the
    grouped shape when `ranked` is absent, so an older API stays readable.
    """
    ranked = payload.get("ranked")
    if isinstance(ranked, list) and ranked:
        not_searched = [
            corpus
            for corpus in ("homelab", "ai")
            if not (payload.get("corpora", {}).get(corpus) or {}).get("searched")
        ]
        lines = _render_hits(ranked, f"=== {len(ranked)} result(s), best first ===")
        if not_searched:
            lines.append(f"(not searched: {', '.join(not_searched)})")
        return "\n".join(lines).strip()

    lines: list[str] = []
    for corpus in ("homelab", "ai"):
        section = payload.get("corpora", {}).get(corpus)
        if not isinstance(section, dict):
            continue
        if not section.get("searched"):
            lines.append(f"=== {corpus}: not searched ===")
            continue
        results = section.get("results") or []
        lines.extend(_render_hits(results, f"=== {corpus}: {len(results)} result(s) ==="))
    return "\n".join(lines).strip() or "No results."


@mcp.tool(
    description=(
        "MUST be called before any research, implementation, debugging, or configuration task. "
        "Searches both knowledge corpora — Homelab infrastructure and AI research — and returns "
        "corpus-qualified entries grouped per corpus, without an internal LLM call. When supplying "
        "query_alt, translate the same intent faithfully into the other language without adding facts, "
        "and preserve technical literals exactly; query_alt_language is required with it."
    )
)
def semantic_search(
    query: str,
    query_alt: str | None = None,
    query_alt_language: Literal["sr", "en"] | None = None,
) -> str:
    return _format_corpus_payload(
        corpus_search(query, scope="both", query_alt=query_alt, query_alt_language=query_alt_language)
    )


@mcp.tool(
    description=(
        "Search the structured Homelab and AI knowledge corpora without an internal LLM call. "
        "Defaults to both corpora; narrow it with homelab or ai when the target is known. "
        "auto stays unavailable until its calibrated router is enabled. "
        "Returns corpus-qualified references and grouped results. When supplying query_alt, translate "
        "the same intent faithfully into the other language without adding facts, and preserve technical "
        "literals exactly; query_alt_language is required with it."
    )
)
def corpus_search(
    query: str,
    # Not "auto": that returns HTTP 409 until the router is calibrated, so it
    # would make every scope-less call an error.
    scope: Literal["homelab", "ai", "both", "auto"] = "both",
    top_k: Annotated[int, Field(ge=1, le=5)] = 5,
    query_alt: str | None = None,
    query_alt_language: Literal["sr", "en"] | None = None,
) -> dict:
    _load_local_v2_token()
    token = os.getenv("KB_V2_TOKEN_MCP_LOCAL")
    if not token:
        raise RuntimeError("KB corpus search is not configured")
    endpoint = os.getenv("KB_V2_SEARCH_URL", "http://127.0.0.1:8050/v2/kb/search")
    try:
        payload = {
            "query": query,
            "scope": scope,
            "top_k": top_k,
            "allow_degraded": False,
        }
        if query_alt is not None:
            payload["query_alt"] = query_alt
        if query_alt_language is not None:
            payload["query_alt_language"] = query_alt_language
        response = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
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
