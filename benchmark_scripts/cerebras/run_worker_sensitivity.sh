#!/usr/bin/env bash
# Detach by default; --foreground is used by the enclosing worker campaign.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Resolve --output relative to the caller's directory, including on resume.
exec uv run --no-sync --project "$PROJECT_ROOT" -- python \
    "$SCRIPT_DIR/worker_launcher.py" sensitivity "$@"
