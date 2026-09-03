#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
UV_BIN="${KB_UV_BIN:-uv}"
SEARCH_PYTHON="${KB_SEARCH_PYTHON:-/opt/kb/venv-search/bin/python}"

"$UV_BIN" pip sync \
    --python "$SEARCH_PYTHON" \
    "$SCRIPT_DIR/search.lock" \
    --torch-backend cpu
