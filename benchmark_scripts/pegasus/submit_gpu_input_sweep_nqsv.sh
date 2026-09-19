#!/usr/bin/env bash
# Submit one sequential GPU input campaign; --dry-run never calls qsub.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
backend=both
dataset=arxiv
pyg_config=""
output=""
workers="2 4 8 12 16 24 32 40 48 64"
phase=all
compile=0
dry_run=0
usage() {
    echo "Usage: $0 --output DIR [--dataset arxiv|products] [--backend both|fixed_shape|pyg] [--pyg-base-config PATH] [--workers '2 4 ... 64'] [--phase all|tune|workers|prefetch1|persistent-off|feature-cache] [--compile] [--dry-run]"
}
while (( $# )); do
    case "$1" in
        --pyg-base-config|--dataset|--backend|--output|--workers|--phase)
            [[ -n "${2:-}" ]] || { usage >&2; exit 2; }
            case "$1" in
                --pyg-base-config) pyg_config="$2" ;;
                --backend) backend="$2" ;;
                --dataset) dataset="$2" ;;
                --output) output="$2" ;;
                --workers) workers="$2" ;;
                --phase) phase="$2" ;;
            esac
            shift 2 ;;
        --compile) compile=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n "$output" ]] || { usage >&2; exit 2; }
case "$dataset" in
    arxiv) measure_steps=800 ;;
    products) measure_steps=1600 ;;
    *) echo 'Dataset must be arxiv or products' >&2; exit 2 ;;
esac
pyg_config="${pyg_config:-${project_root}/src/cerebras/modelzoo/models/gnn/configs/autotune/${dataset}_w40.yaml}"
case "$backend" in both|fixed_shape|pyg) ;; *) usage >&2; exit 2 ;; esac
if [[ "$backend" != pyg ]]; then
    config="${project_root}/src/cerebras/modelzoo/models/gnn/configs/fixed_shape_gpu/${dataset}.yaml"
    [[ -f "$config" ]] || { echo 'Fixed-shape config not found' >&2; exit 2; }
    config="$(cd "$(dirname "$config")" && pwd -P)/$(basename "$config")"
fi
if [[ "$backend" != fixed_shape ]]; then
    [[ -f "$pyg_config" ]] || { echo 'PyG config not found' >&2; exit 2; }
    pyg_config="$(cd "$(dirname "$pyg_config")" && pwd -P)/$(basename "$pyg_config")"
fi
[[ "$workers" =~ ^[0-9]+([[:blank:]]+[0-9]+)*$ ]] || { echo 'Invalid --workers list' >&2; exit 2; }
case "$phase" in all|tune|workers|prefetch1|persistent-off|feature-cache) ;; *) usage >&2; exit 2 ;; esac
[[ "$output" == /* ]] || output="${PWD}/${output}"
for value in "$pyg_config" "$output"; do
    if [[ "$value" == *','* || "$value" == *$'\n'* ]]; then
        echo 'Paths must not contain commas or newlines.' >&2; exit 2
    fi
done
command=(qsub -v "GPU_SWEEP_DATASET=${dataset},GPU_SWEEP_OUTPUT=${output},GPU_SWEEP_COMPILE=${compile},GPU_SWEEP_WORKERS=${workers},GPU_SWEEP_PHASE=${phase},GPU_SWEEP_BACKEND=${backend},GPU_SWEEP_PYG_CONFIG=${pyg_config}"
    "${script_dir}/run_gpu_input_sweep_nqsv.pbs")
cd "$project_root"
if (( dry_run )); then
    printf 'cd %q && ' "$project_root"
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
