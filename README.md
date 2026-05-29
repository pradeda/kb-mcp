# kb-mcp

MCP (Model Context Protocol) server and KB Search API for the homelab knowledge base. Two components:

> **Used by:** [kb-go](https://github.com/pradeda/kb-go) — `kb ask` calls `kb_search_api.py` for retrieval + cross-encoder reranking before LLM synthesis.

| File | Role | Runs as |
|------|------|---------|
| `mcp_server.py` | MCP tools for LLM agents (`kb_search`, `kb_add`) | Registered in Claude Code / Gemini |
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
  - ChromaDB at `localhost:8000`
  - SQLite DB at `/opt/kb/kb.db`

## Usage

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install fastmcp fastapi uvicorn httpx sentence-transformers

# Run MCP server (registered via MCP client config)
python3 mcp_server.py

# Run KB Search API (systemd or manual)
python3 kb_search_api.py
```

## Architecture

```
LLM Agent → MCP Protocol → kb_search() → subprocess → kb ask
                           │                            │
                           │              POST /kb/search {"format": "full"}
                           │                            │
                           │              ┌─────────────┘
                           │              ▼
                           │    kb_search_api.py (:8050)
                           │    ├── FastEmbed → ChromaDB (top 25)
                           │    ├── Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
                           │    ├── Dedup + time decay (half-life 540d, floor 0.3)
                           │    └── Threshold 0.5 + cap top 5
                           │              │
                           │              ▼
                           └── LLM synthesis (OpenRouter)

                           kb_add() → subprocess → kb add → SQLite + raw notes
```

## MCP Tools

### `kb_search(query)`

Semantic search over the knowledge base. MUST be called before any research, implementation, debugging, or configuration task.

Returns relevant entries about infrastructure, services, and past work.

```
Tool: kb_search
Args: query (string)
```

### `kb_add(content, title, tag)`

Add a note to the knowledge base. Use for documenting solutions, gotchas, config changes, or any knowledge worth preserving.

```
Tool: kb_add
Args: content (string), title (string), tag (string)
```

## Registering in MCP Clients

### Gemini CLI

Add to `~/.gemini/config/mcp_config.json`:

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

### Claude Code

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

