# kb-mcp

MCP (Model Context Protocol) server for the homelab knowledge base. Exposes two tools to LLM agents — semantic search and note creation — backed by a local ChromaDB + SQLite pipeline.

## Requirements

- Python 3.10+
- `fastmcp` — `pip install fastmcp`
- Local KB stack:
  - `kb-ask` binary at `/usr/local/bin/kb-ask` — semantic search via ChromaDB + OpenRouter synthesis
  - `kb` binary at `/usr/local/bin/kb` — CLI for adding notes (SQLite + raw markdown)

## Usage

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install fastmcp

# Run
python3 mcp_server.py
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

## Architecture

```
LLM Agent → MCP Protocol → kb_search() → subprocess → kb-ask → ChromaDB + OpenRouter
                          kb_add()    → subprocess → kb add  → SQLite + raw notes
```
