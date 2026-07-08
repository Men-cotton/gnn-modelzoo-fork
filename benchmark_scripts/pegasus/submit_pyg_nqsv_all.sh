#!/usr/bin/env bash
# Submit supported PyG Pegasus PBS benchmark jobs.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
pbs_script="${script_dir}/run_pyg_nqsv.pbs"
hwinfo_pbs_script="${script_dir}/collect_hwinfo_nqsv.pbs"
configs_dir="${project_root}/src/cerebras/modelzoo/models/gnn/configs"

dry_run=0
no_compile="${PYG_NO_COMPILE:-0}"
only_filters=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [--dry-run] [--no-compile] [--only BENCHMARK[/PROFILE]]

Submits Pegasus GPU benchmark configs supported by run_pyg_nqsv.pbs.
By default, all supported jobs are submitted.
Hardware info is collected once through a separate GPU PBS job.
Set PYG_NO_COMPILE=1 or pass --no-compile to disable torch.compile.

Options:
  --only BENCHMARK[/PROFILE]  Submit only this benchmark, or one benchmark/profile pair.
                              May be passed more than once.

Examples:
  $(basename "$0") --only graphsage_ogbn_papers100m
  $(basename "$0") --only graphsage_ogbn_arxiv/throughput_nocache --only graphsage_ogbn_papers100m/throughput_nocache
EOF
}

require_arg() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "${value}" ]]; then
        echo "[ERROR] ${option} requires an argument" >&2
        usage >&2
        exit 2
    fi
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        --no-compile)
            no_compile=1
            shift
            ;;
        --only)
            require_arg "$1" "${2:-}"
            only_filters+=("$2")
            shift 2
            ;;
        --only=*)
            require_arg "--only" "${1#*=}"
            only_filters+=("${1#*=}")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "${pbs_script}" ]]; then
    echo "[ERROR] PBS script not found: ${pbs_script}" >&2
    exit 2
fi

if [[ ! -f "${hwinfo_pbs_script}" ]]; then
    echo "[ERROR] hardware info PBS script not found: ${hwinfo_pbs_script}" >&2
    exit 2
fi

validate_num_workers() {
    "${project_root}/benchmark_scripts/validate_num_workers.sh" \
        "${configs_dir}"
}

matches_only() {
    local benchmark="$1"
    local profile="$2"
    local filter

    if [[ "${#only_filters[@]}" -eq 0 ]]; then
        return 0
    fi

    for filter in "${only_filters[@]}"; do
        if [[ "${filter}" == "${benchmark}" || "${filter}" == "${benchmark}/${profile}" ]]; then
            return 0
        fi
    done

    return 1
}

submit_hwinfo_job() {
    if [[ "${dry_run}" -eq 1 ]]; then
        printf "qsub '%s'\n" "${hwinfo_pbs_script}"
        return 0
    fi

    echo "[submit] hardware info"
    qsub "${hwinfo_pbs_script}"
}

submit_job() {
    local benchmark="$1"
    local profile="$2"
    local config_prefix="$3"
    local config_path="${configs_dir}/${config_prefix}_${profile}.yaml"
    local vars="PYG_BENCHMARK=${benchmark},PYG_RUN_PROFILE=${profile},PYG_NO_COMPILE=${no_compile}"

    if [[ ! -f "${config_path}" ]]; then
        echo "[ERROR] config not found: ${config_path}" >&2
        exit 2
    fi

    if [[ "${dry_run}" -eq 1 ]]; then
        printf "qsub -v '%s' '%s'\n" "${vars}" "${pbs_script}"
        return 0
    fi

    echo "[submit] ${benchmark} ${profile}"
    qsub -v "${vars}" "${pbs_script}"
}

jobs=(
    "graphsage_ogbn_arxiv throughput_nocache params_graphsage_ogbn_arxiv"
    "graphsage_ogbn_arxiv throughput_cache params_graphsage_ogbn_arxiv"
    "graphsage_ogbn_arxiv accuracy_nocache params_graphsage_ogbn_arxiv"
    "graphsage_ogbn_products throughput_nocache params_graphsage_ogbn_products"
    "graphsage_ogbn_products throughput_cache params_graphsage_ogbn_products"
    "graphsage_ogbn_products accuracy_nocache params_graphsage_ogbn_products"
    "graphsage_ogbn_papers100m throughput_nocache params_graphsage_ogbn_papers100m"
    "graphsage_ogbn_papers100m throughput_cache params_graphsage_ogbn_papers100m"
    "graphsage_ogbn_papers100m accuracy_nocache params_graphsage_ogbn_papers100m"
    "gcn_ogbn_arxiv throughput_nocache params_gcn_ogbn_arxiv"
    "gcn_ogbn_arxiv accuracy_nocache params_gcn_ogbn_arxiv"
)

selected_jobs=()
for job in "${jobs[@]}"; do
    read -r benchmark profile config_prefix <<< "${job}"
    if matches_only "${benchmark}" "${profile}"; then
        selected_jobs+=("${job}")
    fi
done

if [[ "${#selected_jobs[@]}" -eq 0 ]]; then
    echo "[ERROR] no jobs matched --only filters" >&2
    exit 2
fi

if [[ "${dry_run}" -eq 0 ]] && ! command -v qsub >/dev/null 2>&1; then
    echo "[ERROR] qsub command not found. Use --dry-run to print commands." >&2
    exit 1
fi

validate_num_workers
submit_hwinfo_job

for job in "${selected_jobs[@]}"; do
    read -r benchmark profile config_prefix <<< "${job}"
    submit_job "${benchmark}" "${profile}" "${config_prefix}"
done
