# KB search and MCP Python environments

The two manifests deliberately describe separate runtimes. Regenerate them with:

```bash
uv pip compile requirements/search.in -o requirements/search.lock --python-version 3.13 --torch-backend cpu
uv pip compile requirements/mcp.in -o requirements/mcp.lock --python-version 3.13
```

Install search with the CPU backend explicitly:

```bash
uv venv --python /usr/bin/python3 /opt/kb/venv-search
requirements/sync-search.sh
```

Create the MCP venv once, then synchronize it independently:

```bash
uv venv --python /usr/bin/python3 /opt/kb/venv
uv pip sync --python /opt/kb/venv/bin/python requirements/mcp.lock
```

Never omit `--torch-backend cpu`: Nexus has no usable GPU and the default PyPI
resolution previously installed roughly 2.7 GB of dead CUDA/NVIDIA packages. Every
venv must have `include-system-site-packages = false`; verify imports with `python -s`
and reject any path under `~/.local`.

The requirements format records `torch==2.12.0+cpu` but cannot bind only that package
to uv's PyTorch CPU index. A direct `uv pip sync requirements/search.lock` therefore
fails loudly. Use `sync-search.sh`, which carries the required backend option in an
executable installation path; `KB_UV_BIN` and `KB_SEARCH_PYTHON` exist for isolated
verification only.
