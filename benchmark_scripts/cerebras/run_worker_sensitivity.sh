#!/usr/bin/env bash
# User-facing preset: detach by default; --foreground runs in this terminal.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Resolve --output relative to the caller's directory, including on resume.
exec uv run --no-sync --project "$PROJECT_ROOT" -- python -u \
    "$PROJECT_ROOT/src/cerebras/modelzoo/models/gnn/tools/autotune.py" \
    --backend csx --mode sensitivity --detach "$@"
