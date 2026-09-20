#!/usr/bin/env bash
# One independent PBS job per dataset, seed and learning/throughput/cache run.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
dataset=all
output=""
hours=4
compile=1
dry_run=0
usage() {
    echo "Usage: $0 --output DIR [--dataset all|arxiv|products] [--hours 3|4] [--compile|--no-compile] [--dry-run]"
    echo "Submit 30 runs for both datasets: 6 learning, 18 uncached, 6 cached."
}
while (( $# )); do
    case "$1" in
        --output|--dataset|--hours)
            [[ -n "${2:-}" ]] || { usage >&2; exit 2; }
            case "$1" in
                --output) output="$2" ;;
                --dataset) dataset="$2" ;;
                --hours) hours="$2" ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --compile) compile=1; shift ;;
        --no-compile) compile=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n "$output" ]] || { usage >&2; exit 2; }
case "$dataset" in
    all) datasets=(arxiv products) ;;
    arxiv|products) datasets=("$dataset") ;;
    *) usage >&2; exit 2 ;;
esac
case "$hours" in 3|4) ;; *) usage >&2; exit 2 ;; esac
[[ "$output" == /* ]] || output="${PWD}/${output}"
if [[ "$output" == *','* || "$output" =~ [[:space:]] ]]; then
    echo 'Output path must not contain commas or whitespace.' >&2
    exit 2
fi
# Leave five minutes for result collection before the scheduler kills the job.
timeout=$((hours * 3600 - 300))
if (( ! dry_run )); then
    command -v qsub >/dev/null || { echo 'qsub not found; use --dry-run to preview.' >&2; exit 1; }
    [[ ! -e "$output" ]] || { echo 'Use a fresh output directory.' >&2; exit 2; }
    mkdir -p -- "$output"
fi
cd "$project_root"
for job_dataset in "${datasets[@]}"; do
    for seed in 42 43 44; do
        for run in learning_r1 throughput_r1 throughput_r2 throughput_r3 cache_r1; do
            run_id="seed_${seed}/${run}"
            vars="HPCASIA_DATASET=${job_dataset},HPCASIA_RUN_ID=${run_id},HPCASIA_OUTPUT=${output},HPCASIA_TIMEOUT=${timeout},HPCASIA_COMPILE=${compile}"
            command=(qsub -l "elapstim_req=0${hours}:00:00" -v "$vars" "${script_dir}/run_hpcasia_nqsv.pbs")
            if (( dry_run )); then
                printf 'cd %q && ' "$project_root"
                printf '%q ' "${command[@]}"
                printf '\n'
            else
                printf '[submit] %s %s\n' "$job_dataset" "$run_id"
                "${command[@]}"
            fi
        done
    done
done
