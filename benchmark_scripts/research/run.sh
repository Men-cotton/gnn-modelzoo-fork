#!/usr/bin/env bash
# Use this fork's existing uv environment for preparation, execution and tmux.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
    cat <<'HELP'
Usage: benchmark_scripts/research/run.sh MODULE [ARG ...]
MODULE: history_embedding.hardware|pilot|campaign|run|preflight|verify_sdk|evaluate
        feature_sharing.device.prepare|pilot|run|preflight
        feature_sharing.probe_shared_tree_features
        logs.RUNNER
        launcher  (existing ModelZoo tmux launcher)
Example: benchmark_scripts/research/run.sh history_embedding.pilot --help
Uses uv run --no-sync --project FORK_ROOT; provision the uv environment first.
HELP
    exit 0
fi
case "$1" in
    launcher) module=cerebras.modelzoo.tools.benchmark_launcher ;;
    history_embedding.*|feature_sharing.*|logs.*) module="benchmark_scripts.research.$1" ;;
    *) echo "Unknown research module: $1" >&2; exit 2 ;;
esac
shift
export PYTHONPATH="${project_root}:${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec uv run --no-sync --project "${project_root}" -- python -u -m "${module}" "$@"
