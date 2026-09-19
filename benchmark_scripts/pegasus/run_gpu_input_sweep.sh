#!/usr/bin/env bash
# Matched CS-3 input studies on one GPU in an existing Pegasus allocation.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
backend=both
dataset=arxiv
pyg_config=""
output=""
phase=all
workers="2 4 8 12 16 24 32 40 48 64"
budget_sec=10800
trial_timeout_sec=1800
compile=0
dry_run=0
usage() {
    echo "Usage: $0 --output DIR [--dataset arxiv|products] [--backend both|fixed_shape|pyg] [--pyg-base-config PATH] [--phase all|tune|workers|prefetch1|persistent-off|feature-cache] [--workers '2 4 ... 64'] [--compile] [--budget-sec N] [--trial-timeout-sec N] [--dry-run]"
}
while (( $# )); do
    case "$1" in
        --pyg-base-config|--dataset|--backend|--output|--phase|--workers|--budget-sec|--trial-timeout-sec)
            [[ -n "${2:-}" ]] || { usage >&2; exit 2; }
            case "$1" in
                --pyg-base-config) pyg_config="$2" ;;
                --backend) backend="$2" ;;
                --dataset) dataset="$2" ;;
                --output) output="$2" ;;
                --phase) phase="$2" ;;
                --workers) workers="$2" ;;
                --budget-sec) budget_sec="$2" ;;
                --trial-timeout-sec) trial_timeout_sec="$2" ;;
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
case "$backend" in
    both) backends=(fixed_shape pyg) ;;
    fixed_shape|pyg) backends=("$backend") ;;
    *) usage >&2; exit 2 ;;
esac
if [[ "$backend" != pyg ]]; then
    config="${project_root}/src/cerebras/modelzoo/models/gnn/configs/fixed_shape_gpu/${dataset}.yaml"
    [[ -f "$config" ]] || { echo 'Fixed-shape config not found' >&2; exit 2; }
    config="$(cd "$(dirname "$config")" && pwd -P)/$(basename "$config")"
fi
if [[ "$backend" != fixed_shape ]]; then
    [[ -f "$pyg_config" ]] || { echo 'PyG config not found' >&2; exit 2; }
    pyg_config="$(cd "$(dirname "$pyg_config")" && pwd -P)/$(basename "$pyg_config")"
fi
case "$phase" in
    all) phases=(tune workers prefetch1 persistent-off feature-cache) ;;
    tune|workers|prefetch1|persistent-off|feature-cache) phases=("$phase") ;;
    *) usage >&2; exit 2 ;;
esac
read -r -a worker_counts <<< "$workers"
(( ${#worker_counts[@]} )) || { echo 'Empty worker list' >&2; exit 2; }
for count in "${worker_counts[@]}" "$budget_sec" "$trial_timeout_sec"; do
    [[ "$count" =~ ^[0-9]+$ ]] || { echo "Invalid integer: $count" >&2; exit 2; }
done
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
python="${project_root}/.venv/bin/python"
if (( ! dry_run )); then
    source "${project_root}/common.sh"
    source "${script_dir}/gpu_env.sh"
    load_cuda_module
    require_cuda_toolkit
    [[ -f "${project_root}/.venv/.setup_successful" ]] || {
        echo 'Run ./setup.sh --target-env gpu before launching Pegasus jobs.' >&2; exit 4;
    }
    # Require the requested allocation even in winner-selection mode, which
    # otherwise silently omits worker counts exceeding CPU affinity.
    checked_workers=("${worker_counts[@]}")
    case "$phase" in prefetch1|persistent-off|feature-cache) checked_workers=(4) ;; esac
    "$python" -c 'import sys; from cerebras.modelzoo.models.gnn.worker_validation import get_available_cpu_cores; n=get_available_cpu_cores(); requested=list(map(int,sys.argv[1:])); print(f"CPU allocation: {n}; requested workers: {requested}"); sys.exit(0 if n is None or max(requested)<=n else "Requested workers exceed CPU allocation; request more CPUs or explicitly narrow --workers.")' "${checked_workers[@]}"
fi
common=(--dataset "$dataset"
    --warmup-steps 40 --measure-steps "$measure_steps" --repeats 3
    --prefetch-factor 2 --persistent-workers --cache none
    --stability-tolerance-percent 2 --budget-sec "$budget_sec"
    --trial-timeout-sec "$trial_timeout_sec" --foreground)
(( ! dry_run )) || common+=(--dry-run)
status=0
for route in "${backends[@]}"; do
    route_args=(--backend "$route")
    if [[ "$route" == fixed_shape ]]; then
        route_args+=(--base-config "$config")
        (( ! compile )) || route_args+=(--compile)
    else
        route_args+=(--base-config "$pyg_config")
        # PyG's existing runner selects compilation through NO_COMPILE.
        if (( compile )); then unset NO_COMPILE; else export NO_COMPILE=1; fi
    fi
    route_output="${output}/${route}"
    for study in "${phases[@]}"; do
        args=("${common[@]}" "${route_args[@]}" --output "${route_output}/${study}")
        case "$study" in
            tune) args+=(--mode autotune --workers "${worker_counts[@]}"
                --prefetch-factors 1 2 4 --top-k 2 --confirm-steps "$measure_steps") ;;
            workers) args+=(--mode sensitivity --workers "${worker_counts[@]}" --continue-on-failure) ;;
            prefetch1) args+=(--mode sensitivity --workers 4 --prefetch-factor 1 --continue-on-failure) ;;
            persistent-off) args+=(--mode sensitivity --workers 4 --no-persistent-workers --continue-on-failure) ;;
            feature-cache) args+=(--mode sensitivity --workers 4 --cache full --continue-on-failure) ;;
        esac
        printf '[gpu_input_sweep] %s\n' "${route}/${study}"
        if "$python" "${project_root}/src/cerebras/modelzoo/models/gnn/tools/autotune.py" "${args[@]}"; then
            continue
        else
            rc=$?
        fi
        # A completed sensitivity grid can retain OOM/timeout failures. Continue
        # independent controls, but return nonzero so missing measurements stay visible.
        if [[ "$rc" == 2 && -f "${route_output}/${study}/study.json" ]] &&
            "$python" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "completed_with_missing_measurements")' "${route_output}/${study}/study.json"; then
            status=2
        else
            exit "$rc"
        fi
    done
done
exit "$status"
