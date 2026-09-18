#!/usr/bin/env bash
# Run one prepared Model Zoo config on one Pegasus GPU.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
config=""
dry_run=0
while (( $# )); do
    case "$1" in
        --config)
            [[ -n "${2:-}" ]] || { echo '--config requires a value' >&2; exit 2; }
            config="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) echo "Usage: $0 --config PATH [--dry-run]"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 2; }
config="$(realpath -- "$config")"
output_dir="$(dirname "$config")"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
command=(uv run --no-sync --project "${project_root}" python -u "${project_root}/benchmark_scripts/non_gnn/run.py"
    --backend GPU --execute-config "$config")
if (( dry_run )); then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
# The scheduler owns this shell. Preserve failures before the Python client can
# start, using the same trial status and measurement records as ordinary runs.
record_preflight_failure() {
    local status=$?
    trap - EXIT
    if (( status != 0 )); then
        "${command[@]}" \
            --preflight-error "GPU shell preflight failed with status ${status}; see PBS stdout/stderr for job ${PBS_JOBID:-local}" \
            --preflight-exit-code "$status" || true
    fi
    exit "$status"
}
trap record_preflight_failure EXIT
source "${project_root}/common.sh"
source "${script_dir}/gpu_env.sh"
load_cuda_module
require_cuda_toolkit
[[ -f "${project_root}/.venv/.setup_successful" ]] || {
    echo 'Run ./setup.sh --target-env gpu before launching Pegasus jobs.' >&2; exit 4;
}
uv run --no-sync --project "${project_root}" python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
cd "$project_root"
{
    echo "[non_gnn] hostname=$(hostname) job=${PBS_JOBID:-local} config=${config}"
    echo "[non_gnn] commit=$(git rev-parse HEAD)"
    nvidia-smi
} 2>&1 | tee "${output_dir}/hardware.log"
trap - EXIT
exec "${command[@]}"
