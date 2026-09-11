#!/usr/bin/env bash
# Run the ModelZoo fixed-shape GraphSAGE representation on one Pegasus GPU.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
args=()
config=""
output_dir=""
dry_run=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --config)
            [[ -n "${2:-}" ]] || { echo '--config requires a value' >&2; exit 2; }
            config="$2"; args+=(--config "$2"); shift 2 ;;
        --output-dir)
            [[ -n "${2:-}" ]] || { echo '--output-dir requires a value' >&2; exit 2; }
            output_dir="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --config PATH [--output-dir DIR] [--compile] [--measure-neighbor-padding] [--precision fp32|fp16|bf16] [--num-workers N] [--max-steps N] [--warmup-steps N] [--dry-run]"
            exit 0 ;;
        *) args+=("$1"); shift ;;
    esac
done
[[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 2; }
output_dir="${output_dir:-${project_root}/model_dirs/fixed_shape_gpu/$(date +%Y%m%d_%H%M%S)_${PBS_JOBID:-local}_$$}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
command=("${project_root}/.venv/bin/python" -m cerebras.modelzoo.models.gnn.fixed_shape_gpu
    "${args[@]}" --output-dir "${output_dir}")
if (( dry_run )); then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
source "${project_root}/common.sh"
source "${script_dir}/gpu_env.sh"
load_cuda_module
require_cuda_toolkit
if [[ ! -f "${project_root}/.venv/.setup_successful" ]]; then
    echo "Run ./setup.sh --target-env gpu before launching Pegasus jobs." >&2
    exit 4
fi
echo "[fixed_shape_gpu] hostname=$(hostname) job=${PBS_JOBID:-local} output=${output_dir}"
echo "[fixed_shape_gpu] commit=$(git -C "${project_root}" rev-parse HEAD)"
exec "${command[@]}"
