#!/usr/bin/env bash
# One detached driver: record learning curves, then tune input parameters.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec uv run --no-sync --project "$PROJECT_ROOT" -- python -u \
  "$PROJECT_ROOT/src/cerebras/modelzoo/models/gnn/tools/learning_campaign.py" \
  --detach "$@"
