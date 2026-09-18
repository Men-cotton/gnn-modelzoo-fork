#!/usr/bin/env bash
# Default: prepare/reuse data and launch all profiles. --profile selects one.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
entry=campaign.py
for arg in "$@"; do
    case "$arg" in
        --profile|--profile=*|--data-dir|--data-dir=*) entry=run.py ;;
    esac
done
exec "${project_root}/.venv/bin/python" "${project_root}/benchmark_scripts/non_gnn/${entry}" \
    --backend CSX "$@"
