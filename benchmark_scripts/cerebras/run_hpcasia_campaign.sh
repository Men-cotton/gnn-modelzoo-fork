#!/usr/bin/env bash
# Run fixed-input learning and throughput measurements, completing each seed first.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec uv run --no-sync --project "$PROJECT_ROOT" -- python -u \
  "$PROJECT_ROOT/src/cerebras/modelzoo/models/gnn/tools/hpcasia_campaign.py" \
  --detach "$@"
