#!/usr/bin/env bash
# One foreground client at a time; every worker count gets the same repeats.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Resolve --output relative to the caller's directory, including on resume.
exec uv run --no-sync --project "$PROJECT_ROOT" -- python \
    "$PROJECT_ROOT/src/cerebras/modelzoo/models/gnn/tools/autotune.py" \
    --backend csx --mode sensitivity "$@"
