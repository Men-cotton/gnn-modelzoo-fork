#!/usr/bin/env bash
# Prepare exactly one profile before qsub. --dry-run creates no files/jobs.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
output_dir="${project_root}/model_dirs/non_gnn/gpu_$(date +%Y%m%d_%H%M%S)_$$"
args=()
dry_run=0
while (( $# )); do
    case "$1" in
        --output-dir)
            [[ -n "${2:-}" ]] || { echo '--output-dir requires a value' >&2; exit 2; }
            output_dir="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --backend|--prepare-only)
            echo "$1 is managed by this submit script" >&2; exit 2 ;;
        -h|--help|--list-configs)
            exec "${project_root}/.venv/bin/python" "${project_root}/benchmark_scripts/non_gnn/run.py" --backend GPU "$1" ;;
        *) args+=("$1"); shift ;;
    esac
done
output_dir="$(realpath -m -- "$output_dir")"
# NQSV -v uses commas as separators; shell quoting cannot escape them there.
if [[ "$output_dir" == *','* || "$output_dir" == *$'\n'* ]]; then
    echo 'Output paths must not contain commas or newlines.' >&2
    exit 2
fi
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
prepare=("${project_root}/.venv/bin/python" "${project_root}/benchmark_scripts/non_gnn/run.py"
    --backend GPU "${args[@]}" --output-dir "$output_dir" --prepare-only)
command=(qsub -v "NON_GNN_CONFIG=${output_dir}/params.yaml" "${script_dir}/run_non_gnn_nqsv.pbs")
if (( dry_run )); then
    "${prepare[@]}" --dry-run
    printf 'cd %q && ' "$project_root"
    printf '%q ' "${command[@]}"
    printf '\n'
else
    command -v qsub > /dev/null || { echo 'qsub not found; submit from Pegasus.' >&2; exit 2; }
    "${prepare[@]}"
    cd "$project_root"
    "${command[@]}" | tee "${output_dir}/qsub.log"
fi
