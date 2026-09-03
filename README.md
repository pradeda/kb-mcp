# kb-mcp

MCP (Model Context Protocol) server and KB Search API for the homelab knowledge base. Two components:

> **Used by:** [kb-go](https://github.com/pradeda/kb-go) — `kb ask` calls `kb_search_api.py` for retrieval + cross-encoder reranking before LLM synthesis.

| File | Role | Runs as |
|------|------|---------|
| `mcp_server.py` | MCP tools (`semantic_search`, `corpus_search`, `add`) | stdio, SSE :9100, HTTP :9101 |
| `kb_search_api.py` | Authenticated v2 read plane; retired v1 routes return 410 | systemd user unit: `kb-search-api` (:8050) |
| `kb_v2.py` | Strict authenticated multi-corpus retrieval | mounted below `/v2` |
| `provision_v2.py` | Idempotent private token/allowlist provisioning | operator command |

## Requirements

### `mcp_server.py`
- Python 3.13 in `/opt/kb/venv`, synchronized from `requirements/mcp.lock`
- `mcp`, `pydantic`, `httpx`, and `uvicorn`; do not install into the user site
- Local KB stack:
  - `kb` binary at `/usr/local/bin/kb` — CLI for search (`kb ask`) and notes (`kb add`)

### `kb_search_api.py`
- Python 3.13 in `/opt/kb/venv-search`, synchronized from
  `requirements/search.lock` with `--torch-backend cpu`
- `sentence-transformers`, CPU-only `torch`, `fastapi`, `httpx`, `uvicorn`,
  `pydantic`, and `PyYAML`
- Local KB stack:
  - FastEmbed daemon at `/run/kb-embed/embed.sock`
  - ChromaDB at `localhost:8000` (container binds `127.0.0.1` only; host mount must target `/data` — rust Chroma ignores `PERSIST_DIRECTORY`)
  - SQLite DB at `/opt/kb/kb.db`
  - AI SQLite DB at `/opt/ai-kb/ai-kb.db`

## Usage

```bash
# Initial creation only; skip `uv venv` when the destination already exists
uv venv --python /usr/bin/python3 /opt/kb/venv-search
uv venv --python /usr/bin/python3 /opt/kb/venv

# Reproducible synchronization from the owned locks
requirements/sync-search.sh
uv pip sync --python /opt/kb/venv/bin/python requirements/mcp.lock

# MCP server — stdio transport (local, for Claude Code)
/opt/kb/venv/bin/python mcp_server.py

# MCP server — SSE transport (remote access, for desktop clients)
/opt/kb/venv/bin/python mcp_server.py --sse    # listens on 0.0.0.0:9100

# MCP server — Streamable HTTP transport (Claude Code 2.x)
/opt/kb/venv/bin/python mcp_server.py --http   # listens on 0.0.0.0:9101/mcp

# Run KB Search API (systemd or manual)
/opt/kb/venv-search/bin/python kb_search_api.py
```

Both venvs must keep `include-system-site-packages = false`, and import paths must
remain under the matching venv rather than `~/.local`. Never omit
`--torch-backend cpu`: Nexus has no usable GPU and CUDA/NVIDIA wheels add gigabytes
without providing a runnable acceleration path. See `requirements/README.md` for
lock regeneration and isolation checks.

### SSE transport (`--sse`)

Adding `--sse` switches from stdio to SSE transport on port 9100. Server binds to `0.0.0.0` so remote desktop clients can connect over LAN/WireGuard.

Host and port are set directly in `FastMCP()` constructor — NOT via environment variables:

```python
mcp = FastMCP("kb", host="0.0.0.0", port=9100)
```

FastMCP's `__init__` has explicit default params (`host="127.0.0.1"`, `port=8000`) that take priority over `FASTMCP_HOST` / `FASTMCP_PORT` env vars. Constructor params are the only reliable way.

Systemd unit for 24/7 operation:
```
# /etc/systemd/system/kb-mcp-sse.service
[Service]
ExecStart=/opt/kb/venv/bin/python3 /opt/kb/mcp_server.py --sse
```

### Streamable HTTP transport (`--http`)

Adding `--http` starts the independent Streamable HTTP endpoint on port 9101 at `/mcp`. It preserves the same tool schema as stdio/SSE and is the preferred remote transport for Claude Code 2.x.

```
# /etc/systemd/system/kb-mcp-http.service
[Service]
ExecStart=/opt/kb/venv/bin/python3 /opt/kb/mcp_server.py --http
```

## Architecture

```
LLM Agent → MCP Protocol → semantic_search() ─┐
          └──────────────→ corpus_search() ───┴─→ POST /v2/kb/search
                                                    │
                                          kb_search_api.py (:8050)
                                          ├── FastEmbed → ChromaDB (top 25)
                                          ├── multilingual cross-encoder rerank
                                          ├── time decay (half-life 540d, floor 0.3)
                                          └── threshold 0.40 + cap top 5

kb ask ──→ POST /v2/kb/search ──→ OpenRouter synthesis

add() → subprocess → kb add → SQLite + raw notes
 ```

### Endpoint design

All HTTP handlers are **plain `def`** — not `async def`. The pipeline is fully blocking (Unix socket embed, sync `httpx`, CPU-bound cross-encoder rerank); FastAPI runs plain-def endpoints in a threadpool, so `/health` stays responsive (~9ms) even while a search is in flight. A previous `async def` version froze the event loop for the full search duration.

### Experimental Nexus relevance synthesis

The live service contains a purpose-bound `POST /kb/synthesize/nexus-relevance` contract for video and article ingest. It is hidden from OpenAPI/MCP discovery and hard-disabled unless `KB_SYNTHESIS_TOKEN` is configured; callers must send the same value in `X-KB-Synthesis-Token`. Requests identify the untrusted item as `source_type: video|article` (default `video` for compatibility). Retrieval stays local and requires `final_score >= 0.60` for videos or `>= 0.50` for article related-knowledge candidates, sends at most three excerpts of 2,000 characters to the configured synthesis provider, validates strict JSON, and rejects citations outside the retrieved ID set. The `nexus-relevance-v2` response separates a related-knowledge match (`kb_match_confirmed`) from operational Nexus relevance (`direct`, `indirect`, or `not_confirmed`), so the lower article candidate threshold cannot by itself establish a service/workflow/hardware/roadmap connection. Every cited entry also carries a validated, English `match_reason` (maximum 500 characters) explaining the concrete overlap without persisting the raw excerpt. No-match responses skip the model call.

The ChromaDB collection UUID is **cached at startup** (resolved once in the `lifespan` handler) instead of being looked up over HTTP on every request. On 404 (collection recreated with a new UUID), the cache is invalidated and re-resolved automatically — one retry is built into `query_chromadb`.

### V2 read plane

`POST /v2/kb/search` is a separate mounted FastAPI application. Its schema is
served at `/v2/openapi.json`. The retired `/kb/search` and `/kb/websearch`
routes return 410 unconditionally and are hidden from the root OpenAPI schema;
they can be restored only by explicitly setting `KB_V1_SEARCH_ENABLED=true`.
V2 accepts strict `query`, `scope`, `top_k` and
`allow_degraded` fields, requires a Bearer token, and always returns separate
`homelab` and `ai` groups with corpus-qualified references. Private `raw_path`
values are never returned. `GET /v2/health` uses the same authorization and only
shows corpora allowed to that client.

Local client names and token environment-variable names live in
`/opt/kb/v2-clients.yml` (0600); token values remain in `/opt/kb/.env` (0600).
Retrieval and future routing parameters live in `/opt/kb/corpus-router.yml`
(0600). Missing, malformed, wrongly owned, or incomplete router config makes v2
health degraded and blocks every v2 search with 503; callers fail closed rather
than falling back to unauthenticated v1.
Provision or validate them without printing secrets:

```bash
/opt/kb/venv-search/bin/python provision_v2.py --install
/opt/kb/venv-search/bin/python provision_v2.py --check
```

Explicit `homelab`, `ai` and `both` scopes are active in Phase C. `auto` is
fail-closed with HTTP 409 until the version-bound router holdout passes Phase E.
The initial `corpus-router-v1-precalibration` version fixes candidate/max-distance
and threshold values for explicit retrieval, with AI decay explicitly disabled;
it is not an approval to activate auto routing.

## MCP Tools

### `semantic_search(query, query_alt=None, query_alt_language=None)`

Semantic search over the knowledge base. MUST be called before any research, implementation, debugging, or configuration task.

Returns relevant entries about infrastructure, services, and past work.

```
Tool: semantic_search
Args: query (string)
      query_alt (optional string)
      query_alt_language (optional `sr|en`, required with query_alt)
```

The tool calls authenticated v2 directly with scope `both`, formats the grouped
results and propagates HTTP, authentication and reranker errors. It never shells
out to `kb ask` and never falls back to the unauthenticated v1 route.

### `corpus_search(query, scope="both", top_k=5, query_alt=None, query_alt_language=None)`

Structured multi-corpus search. It calls `/v2/kb/search` directly with
`allow_degraded=false`, returns grouped JSON, and never invokes OpenRouter. The
scope enum is `homelab|ai|both|auto`; `top_k` is 1–5 per corpus. Its 45-second
timeout and HTTP/auth failures are independent of `semantic_search`.

`query_alt` must be the same question faithfully translated into the other
language: do not add facts or broaden/narrow the intent, and preserve service
names, paths, flags, commands, hostnames, error strings and numbers exactly.
The server applies the alternate only to Homelab Layer 1. It unions both
candidate lists, then runs the existing mMARCO cross-encoder once with the
original query; AI retrieval is unchanged. Alternate-only candidates outside the
top five alternate Layer-1 ranks are pruned before the SQLite fetch/rerank,
while the full alternate list still preserves distances for candidates shared
with the primary list. Omitting the pair preserves single-query retrieval.

### `add(content, title, tag)`

Add a note to the knowledge base. Use for documenting solutions, gotchas, config changes, or any knowledge worth preserving. Content is passed via stdin (multi-line safe). Timeout 15s, handled gracefully.

```
Tool: add
Args: content (string), title (string), tag (string)
```

## Registering in MCP Clients

### Stdio transport (local)

For MCP clients running on the same machine:

**Gemini CLI** — `~/.gemini/config/mcp_config.json`:
```json
{
    "mcpServers": {
        "kb": {
            "command": "/path/to/venv/bin/python3",
            "args": ["/path/to/kb-mcp/mcp_server.py"]
        }
    }
}
```

**Claude Code** — `~/.claude.json`:
```json
{
    "mcpServers": {
        "kb": {
            "type": "stdio",
            "command": "/path/to/venv/bin/python3",
            "args": ["/path/to/kb-mcp/mcp_server.py"],
            "env": {}
        }
    }
}
```

### SSE transport (remote desktop)

For MCP clients on other machines (LAN or WireGuard). Requires the SSE server to be running (systemd unit or manual `--sse`).

**Claude Desktop** / **OpenCode** / any SSE-compatible client:
```json
{
    "mcpServers": {
        "kb": {
            "url": "http://<nexus-ip>:9100/sse"
        }
    }
}
```

No `command`, `args`, or `env` fields — SSE transport connects to an already-running server. Real-world example:

```json
{
    "mcpServers": {
        "kb": {
            "url": "http://<your-server-ip>:9100/sse"
        }
    }
}
```

### Streamable HTTP transport (remote Claude Code)

```json
{
    "mcpServers": {
        "kb": {
            "type": "http",
            "url": "http://<your-server-ip>:9101/mcp"
        }
    }
}
```
