# kb-mcp

MCP (Model Context Protocol) server and KB Search API for the homelab knowledge base. Two components:

> **Used by:** [kb-go](https://github.com/pradeda/kb-go) — `kb ask` calls `kb_search_api.py` for retrieval + cross-encoder reranking before LLM synthesis.

| File | Role | Runs as |
|------|------|---------|
| `mcp_server.py` | MCP tools for LLM agents (`semantic_search`, `add`) | Registered in Claude Code / Gemini |
| `kb_search_api.py` | FastAPI service with 4-layer ranking pipeline | systemd user unit: `kb-search-api` (:8050) |

## Requirements

### `mcp_server.py`
- Python 3.10+
- `fastmcp` — `pip install fastmcp`
- Local KB stack:
  - `kb` binary at `/usr/local/bin/kb` — CLI for search (`kb ask`) and notes (`kb add`)

### `kb_search_api.py`
- Python 3.10+
- `sentence-transformers`, `fastapi`, `httpx`, `uvicorn`
- Local KB stack:
  - FastEmbed daemon at `/run/kb-embed/embed.sock`
  - ChromaDB at `localhost:8000` (container binds `127.0.0.1` only; host mount must target `/data` — rust Chroma ignores `PERSIST_DIRECTORY`)
  - SQLite DB at `/opt/kb/kb.db`

## Usage

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install fastmcp fastapi uvicorn httpx sentence-transformers

# MCP server — stdio transport (local, for Claude Code)
python3 mcp_server.py

# MCP server — SSE transport (remote access, for desktop clients)
python3 mcp_server.py --sse    # listens on 0.0.0.0:9100

# MCP server — Streamable HTTP transport (Claude Code 2.x)
python3 mcp_server.py --http   # listens on 0.0.0.0:9101/mcp

# Run KB Search API (systemd or manual)
python3 kb_search_api.py
```

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
LLM Agent → MCP Protocol → semantic_search() → subprocess → kb ask
                           │                            │
                           │              POST /kb/search {"format": "full"}
                           │                            │
                           │              ┌─────────────┘
                           │              ▼
                           │    kb_search_api.py (:8050)
                           │    ├── FastEmbed → ChromaDB (top 25)
                           │    ├── Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
                           │    ├── Dedup + time decay (half-life 540d, floor 0.3)
                           │    └── Threshold 0.40 + cap top 5
                           │              │
                           │              ▼
                           └── LLM synthesis (OpenRouter)

add() → subprocess → kb add → SQLite + raw notes
 ```

### Endpoint design

All HTTP handlers are **plain `def`** — not `async def`. The pipeline is fully blocking (Unix socket embed, sync `httpx`, CPU-bound cross-encoder rerank); FastAPI runs plain-def endpoints in a threadpool, so `/health` stays responsive (~9ms) even while a search is in flight. A previous `async def` version froze the event loop for the full search duration.

### Experimental Nexus relevance synthesis

The live service contains a purpose-bound `POST /kb/synthesize/nexus-relevance` contract for video and article ingest. It is hidden from OpenAPI/MCP discovery and hard-disabled unless `KB_SYNTHESIS_TOKEN` is configured; callers must send the same value in `X-KB-Synthesis-Token`. Requests identify the untrusted item as `source_type: video|article` (default `video` for compatibility). Retrieval stays local and requires `final_score >= 0.60` for videos or `>= 0.50` for article related-knowledge candidates, sends at most three excerpts of 2,000 characters to the configured synthesis provider, validates strict JSON, and rejects citations outside the retrieved ID set. The `nexus-relevance-v2` response separates a related-knowledge match (`kb_match_confirmed`) from operational Nexus relevance (`direct`, `indirect`, or `not_confirmed`), so the lower article candidate threshold cannot by itself establish a service/workflow/hardware/roadmap connection. Every cited entry also carries a validated, English `match_reason` (maximum 500 characters) explaining the concrete overlap without persisting the raw excerpt. No-match responses skip the model call.

The ChromaDB collection UUID is **cached at startup** (resolved once in the `lifespan` handler) instead of being looked up over HTTP on every request. On 404 (collection recreated with a new UUID), the cache is invalidated and re-resolved automatically — one retry is built into `query_chromadb`.

## MCP Tools

### `semantic_search(query)`

Semantic search over the knowledge base. MUST be called before any research, implementation, debugging, or configuration task.

Returns relevant entries about infrastructure, services, and past work.

```
Tool: semantic_search
Args: query (string)
```

Subprocess timeout is **180s** (`kb ask` = search API retries + OpenRouter synthesis with its own 60s client and 429 retry — 30s was mathematically impossible for the slow path). On timeout the tool returns a readable message instead of an unhandled `TimeoutExpired` traceback.

The LLM synthesis step is **intentional**: consumers include small-context models (local Qwen, DeepSeek in debates) over SSE, so a cheap model (Gemini Flash Lite) compresses 5×3KB raw chunks into a compact answer to save consumer-side tokens. Do not "optimize" it away by returning raw results.

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
            "url": "http://192.168.1.174:9100/sse"
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
            "url": "http://192.168.1.174:9101/mcp"
        }
    }
}
```
