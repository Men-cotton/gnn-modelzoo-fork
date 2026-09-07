#!/bin/bash
# Executes the GNN ModelZoo training/evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../common.sh
source "${PROJECT_ROOT}/common.sh"

SCRIPT_NAME="$(basename "$0")"

PYTHON_SCRIPT_NAME="cszoo"
CONFIG_INDEX=""
CONFIG_PATH=""
DRY_RUN=0
CONFIGS_DIR="${PROJECT_ROOT}/${SHARED_MODEL_SUBDIR}/configs"

CONFIG_LABELS=(
    "ogbn-arxiv throughput"
    "ogbn-arxiv accuracy"
    "ogbn-products throughput"
    "ogbn-products accuracy"
    "ogbn-arxiv throughput cache"
    "ogbn-products throughput cache"
    "ogbn-arxiv GCN throughput"
    "ogbn-arxiv GCN accuracy"
)
CONFIG_PATHS=(
    "configs/params_graphsage_ogbn_arxiv.yaml"
    "configs/params_graphsage_ogbn_arxiv_accuracy.yaml"
    "configs/params_graphsage_ogbn_products.yaml"
    "configs/params_graphsage_ogbn_products_accuracy.yaml"
    "configs/params_graphsage_ogbn_arxiv_throughput_cache.yaml"
    "configs/params_graphsage_ogbn_products_throughput_cache.yaml"
    "configs/params_gcn_ogbn_arxiv_throughput_nocache.yaml"
    "configs/params_gcn_ogbn_arxiv_accuracy_nocache.yaml"
)

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} --config-index <N> [--dry-run]

Required options:
  --config-index <N>  Select exactly one Cerebras config by number.

Optional:
  --dry-run           Validate the config and print the command without running it.

Available configs:
EOF
    local index
    for index in "${!CONFIG_LABELS[@]}"; do
        printf "  %d  %-32s %s\n" \
            "$((index + 1))" \
            "${CONFIG_LABELS[$index]}" \
            "${CONFIG_PATHS[$index]}"
    done
}

parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --config-index)
                if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
                    log_error "Missing value for --config-index"
                    usage
                    return 2
                fi
                CONFIG_INDEX="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            --list-configs)
                usage
                return 0
                ;;
            -h|--help)
                usage
                return 0
                ;;
            *)
                log_error "Unknown argument: $1"
                usage
                return 2
                ;;
        esac
    done

    if [[ -z "${CONFIG_INDEX}" ]]; then
        log_error "Missing required --config-index <N>"
        usage
        return 2
    fi
    if ! [[ "${CONFIG_INDEX}" =~ ^[0-9]+$ ]]; then
        log_error "--config-index must be a positive integer: ${CONFIG_INDEX}"
        usage
        return 2
    fi
    if (( CONFIG_INDEX < 1 || CONFIG_INDEX > ${#CONFIG_PATHS[@]} )); then
        log_error "--config-index out of range: ${CONFIG_INDEX}"
        usage
        return 2
    fi

    CONFIG_PATH="${CONFIG_PATHS[$((CONFIG_INDEX - 1))]}"
}

validate_num_workers() {
    case "${CONFIG_PATH}" in
        configs/params_graphsage_*)
            ;;
        *)
            return 0
            ;;
    esac

    "${PROJECT_ROOT}/benchmark_scripts/validate_num_workers.sh" \
        "${CONFIGS_DIR}"
    log_info "Validated GraphSAGE neighbor num_workers (runtime checks CPU availability)"
}

main() {
    parse_args "$@"
    local parse_status=$?
    if [[ "${parse_status}" -eq 0 && -z "${CONFIG_PATH}" ]]; then
        return 0
    fi
    if [[ "${parse_status}" -ne 0 ]]; then
        return "${parse_status}"
    fi

    log_info "Starting GNN ModelZoo batch run"

    if ! command -v uv &> /dev/null; then
        log_error "'uv' command not found. Please install uv: https://github.com/astral-sh/uv"
        return 1
    fi

    local model_run_dir="${PROJECT_ROOT}/${SHARED_MODEL_SUBDIR}"
    if [[ ! -d "${model_run_dir}" ]]; then
        log_error "Model script directory '${model_run_dir}' not found."
        return 1
    fi

    if [[ ! -f "${model_run_dir}/${CONFIG_PATH}" ]]; then
        log_error "Config not found: ${model_run_dir}/${CONFIG_PATH}"
        return 2
    fi

    validate_num_workers

    log_info "Executing model script in: ${model_run_dir}"
    ( # Subshell for model execution
        cd "${model_run_dir}"
        local python_script_args=("fit" "${CONFIG_PATH}" "--target_device" "CSX")
        local selected_label="${CONFIG_LABELS[$((CONFIG_INDEX - 1))]}"
        local full_command=("${PYTHON_SCRIPT_NAME}" "${python_script_args[@]}")
        log_info "Selected config ${CONFIG_INDEX}: ${selected_label}"
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            printf "uv run --"
            printf " %q" "${full_command[@]}"
            printf "\n"
            return 0
        fi
        log_info "Executing: uv run ${full_command[*]}"
        uv run -- "${full_command[@]}"
        log_info "Model execution finished."
    )
    local execution_status=$?

    if [ "${execution_status}" -ne 0 ]; then
        log_error "Model execution process failed (status: ${execution_status})."
        return "${execution_status}"
    fi

    log_info "Model execution process completed successfully."
    return 0
}

main "$@"
exit_status=$?
if [ "${exit_status}" -ne 0 ]; then
    log_error "${SCRIPT_NAME} finished with errors (status: ${exit_status})."
else
    log_info "${SCRIPT_NAME} finished successfully."
fi
exit "${exit_status}"
