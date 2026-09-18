#!/usr/bin/env bash
# Run sequential CSX experiments in a detached tmux session by default.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec uv run --no-sync --project "$PROJECT_ROOT" -- python \
  "$SCRIPT_DIR/worker_launcher.py" campaign "$@"
