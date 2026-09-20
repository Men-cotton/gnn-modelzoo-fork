#!/usr/bin/env bash
# Submit one PBS job per GPU input setting; --dry-run never calls qsub.
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
budget_sec=10800
trial_timeout_sec=1800
walltime_hours=24
record_resources=0

usage() {
    echo "Usage: $0 --output DIR [--dataset arxiv|products|all] [--backend both|fixed_shape|pyg] [--pyg-base-config PATH] [--workers '2 4 ... 64'] [--phase all|tune|workers|prefetch1|persistent-off|feature-cache] [--budget-sec N] [--trial-timeout-sec N] [--walltime-hours 1..24] [--record-resources] [--compile] [--dry-run]"
    echo
    echo "Submits one PBS job per dataset/backend/phase/num_workers setting."
    echo "The control phases use num_workers=4 and are submitted once per dataset/backend."
}

while (( $# )); do
    case "$1" in
        --pyg-base-config|--dataset|--backend|--output|--workers|--phase|--budget-sec|--trial-timeout-sec|--walltime-hours)
            [[ -n "${2:-}" ]] || { usage >&2; exit 2; }
            case "$1" in
                --pyg-base-config) pyg_config="$2" ;;
                --backend) backend="$2" ;;
                --dataset) dataset="$2" ;;
                --output) output="$2" ;;
                --workers) workers="$2" ;;
                --phase) phase="$2" ;;
                --budget-sec) budget_sec="$2" ;;
                --trial-timeout-sec) trial_timeout_sec="$2" ;;
                --walltime-hours) walltime_hours="$2" ;;
            esac
            shift 2 ;;
        --compile) compile=1; shift ;;
        --record-resources) record_resources=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
for value in "$budget_sec" "$trial_timeout_sec" "$walltime_hours"; do
    [[ "$value" =~ ^[1-9][0-9]{0,5}$ ]] || { echo 'Time limits must be positive integers without leading zeros.' >&2; exit 2; }
done
(( walltime_hours <= 24 && budget_sec >= trial_timeout_sec + 10 && budget_sec <= walltime_hours * 3600 - 300 )) || {
    echo 'Require timeout + 10 <= budget <= PBS walltime - 300 seconds; walltime must be 1..24 hours.' >&2; exit 2;
}
printf -v walltime '%02d:00:00' "$walltime_hours"

[[ -n "$output" ]] || { usage >&2; exit 2; }
case "$dataset" in
    arxiv|products) datasets=("$dataset") ;;
    all)
        [[ -z "$pyg_config" ]] || {
            echo '--pyg-base-config cannot be combined with --dataset all; use dataset-specific submissions.' >&2
            exit 2
        }
        datasets=(arxiv products)
        ;;
    *) echo 'Dataset must be arxiv, products, or all' >&2; exit 2 ;;
esac
case "$backend" in
    both) backends=(fixed_shape pyg) ;;
    fixed_shape|pyg) backends=("$backend") ;;
    *) usage >&2; exit 2 ;;
esac
case "$phase" in
    all) phases=(tune workers prefetch1 persistent-off feature-cache) ;;
    tune|workers|prefetch1|persistent-off|feature-cache) phases=("$phase") ;;
    *) usage >&2; exit 2 ;;
esac
[[ "$workers" =~ ^[0-9]+([[:blank:]]+[0-9]+)*$ ]] || { echo 'Invalid --workers list' >&2; exit 2; }
read -r -a worker_counts <<< "$workers"
[[ "${#worker_counts[@]}" -gt 0 ]] || { echo 'Worker list must not be empty' >&2; exit 2; }
for value in "${worker_counts[@]}"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "Invalid worker count: $value" >&2; exit 2; }
done
[[ "$output" == /* ]] || output="${PWD}/${output}"
for value in "$pyg_config" "$output"; do
    if [[ "$value" == *','* || "$value" =~ [[:space:]] ]]; then
        echo 'Paths must not contain commas or whitespace.' >&2; exit 2
    fi
done

if (( ! dry_run )) && ! command -v qsub >/dev/null 2>&1; then
    echo '[ERROR] qsub command not found. Use --dry-run to print commands.' >&2
    exit 1
fi

submit_job() {
    local job_dataset="$1"
    local job_backend="$2"
    local job_phase="$3"
    local job_worker="$4"
    local job_pyg_config="${5:-}"
    local job_output="${output}"
    local vars="GPU_SWEEP_DATASET=${job_dataset},GPU_SWEEP_OUTPUT=${job_output},GPU_SWEEP_COMPILE=${compile},GPU_SWEEP_WORKER=${job_worker},GPU_SWEEP_PHASE=${job_phase},GPU_SWEEP_BACKEND=${job_backend},GPU_SWEEP_PYG_CONFIG=${job_pyg_config},GPU_SWEEP_BUDGET_SEC=${budget_sec},GPU_SWEEP_TRIAL_TIMEOUT_SEC=${trial_timeout_sec},GPU_SWEEP_WALLTIME_HOURS=${walltime_hours},GPU_SWEEP_RECORD_RESOURCES=${record_resources}"
    local command=(qsub -l "elapstim_req=${walltime}" -v "$vars" "${script_dir}/run_gpu_input_sweep_nqsv.pbs")

    if (( dry_run )); then
        printf 'cd %q && ' "$project_root"
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        printf '[submit] dataset=%s backend=%s phase=%s num_workers=%s\n' \
            "$job_dataset" "$job_backend" "$job_phase" "$job_worker"
        "${command[@]}"
    fi
}

for job_dataset in "${datasets[@]}"; do
    pyg_config_for_dataset="$pyg_config"
    if [[ "$backend" != fixed_shape ]]; then
        pyg_config_for_dataset="${pyg_config_for_dataset:-${project_root}/src/cerebras/modelzoo/models/gnn/configs/autotune/${job_dataset}_w40.yaml}"
        [[ -f "$pyg_config_for_dataset" ]] || {
            echo "PyG config not found: $pyg_config_for_dataset" >&2; exit 2;
        }
        pyg_config_for_dataset="$(cd "$(dirname "$pyg_config_for_dataset")" && pwd -P)/$(basename "$pyg_config_for_dataset")"
    fi
    if [[ "$backend" != pyg ]]; then
        config="${project_root}/src/cerebras/modelzoo/models/gnn/configs/fixed_shape_gpu/${job_dataset}.yaml"
        [[ -f "$config" ]] || { echo "Fixed-shape config not found: $config" >&2; exit 2; }
    fi
    for job_backend in "${backends[@]}"; do
        for job_phase in "${phases[@]}"; do
            if [[ "$job_phase" == prefetch1 || "$job_phase" == persistent-off || "$job_phase" == feature-cache ]]; then
                submit_job "$job_dataset" "$job_backend" "$job_phase" 4 "$pyg_config_for_dataset"
            else
                for job_worker in "${worker_counts[@]}"; do
                    submit_job "$job_dataset" "$job_backend" "$job_phase" "$job_worker" "$pyg_config_for_dataset"
                done
            fi
        done
    done
done
