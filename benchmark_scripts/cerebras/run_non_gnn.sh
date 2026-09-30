#!/usr/bin/env bash
# Prepare, measure and summarize in tmux; --foreground waits in this terminal.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
if [[ -z "${JOB_SOURCE_ROOT:-}" && " $* " != *" --help "* && " $* " != *" -h "* && " $* " != *" --list-configs "* && " $* " != *" --check-dependencies "* ]]; then
    export PYTHONPATH="${project_root}/src"
    exec "${project_root}/.venv/bin/python" -m cerebras.modelzoo.tools.job_source --entry "${BASH_SOURCE[0]}" -- "$@"
fi
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
entry=campaign.py
for arg in "$@"; do
    case "$arg" in
        --profile|--profile=*|--data-dir|--data-dir=*|--execute-config|--execute-config=*|--check-dependencies) entry=run.py ;;
    esac
done
exec uv run --no-sync --project "${project_root}" python -u "${project_root}/benchmark_scripts/non_gnn/${entry}" \
    --backend CSX --detach "$@"
