#!/bin/bash
# Executes the GNN ModelZoo training/evaluation.

set -euo pipefail

PROJECT_ROOT="${HOME}/gnn-modelzoo"
# shellcheck source=./common.sh
source "${PROJECT_ROOT}/common.sh"

SCRIPT_NAME="$(basename "$0")"

PYTHON_SCRIPT_NAME="run.py"
# Arguments for PYTHON_SCRIPT_NAME
PYTHON_SCRIPT_ARGS=("GPU" "-p" "configs/params.yaml" "-m" "train_and_eval")

main() {
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

    log_info "Executing model script in: ${model_run_dir}"
    ( # Subshell for model execution
        cd "${model_run_dir}"
        local full_command=("${PYTHON_SCRIPT_NAME}" "${PYTHON_SCRIPT_ARGS[@]}")
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
