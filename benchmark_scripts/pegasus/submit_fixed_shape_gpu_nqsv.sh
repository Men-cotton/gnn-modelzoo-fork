#!/usr/bin/env bash
# Submit exactly one fixed-shape GraphSAGE config; --dry-run never calls qsub.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
config=""
compile=0
measure_neighbor_padding=0
measure_input=0
dry_run=0
usage() {
    echo "Usage: $0 --config PATH [--compile] [--measure-neighbor-padding] [--measure-input] [--dry-run]"
}
while (( $# )); do
    case "$1" in
        --config)
            [[ -n "${2:-}" ]] || { usage >&2; exit 2; }
            config="$2"; shift 2 ;;
        --compile) compile=1; shift ;;
        --measure-neighbor-padding) measure_neighbor_padding=1; shift ;;
        --measure-input) measure_input=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 2; }
config="$(cd "$(dirname "$config")" && pwd -P)/$(basename "$config")"
# NQSV's -v value is comma-separated; refuse ambiguous environment values.
if [[ "$config" == *','* || "$config" == *$'\n'* ]]; then
    echo 'Config paths must not contain commas or newlines.' >&2
    exit 2
fi
command=(qsub -v "FIXED_SHAPE_CONFIG=${config},FIXED_SHAPE_COMPILE=${compile},FIXED_SHAPE_MEASURE_NEIGHBOR_PADDING=${measure_neighbor_padding},FIXED_SHAPE_MEASURE_INPUT=${measure_input}"
    "${script_dir}/run_fixed_shape_gpu_nqsv.pbs")
cd "${project_root}"
if (( dry_run )); then
    printf 'cd %q && ' "${project_root}"
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
