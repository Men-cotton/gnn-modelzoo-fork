#!/usr/bin/env bash
# Sequential CSX experiments with opt-in remote resource diagnostics.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec uv run --no-sync --project "$PROJECT_ROOT" -- python \
  "$PROJECT_ROOT/src/cerebras/modelzoo/models/gnn/tools/worker_campaign.py" "$@"
