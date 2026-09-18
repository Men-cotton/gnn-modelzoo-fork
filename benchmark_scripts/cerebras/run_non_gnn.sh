#!/usr/bin/env bash
# Launch exactly one non-GNN Model Zoo training profile on one CS-3.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${project_root}/.venv/bin/python" "${project_root}/benchmark_scripts/non_gnn/run.py" \
    --backend CSX "$@"
