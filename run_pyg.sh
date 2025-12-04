#!/bin/bash
# Executes the PyG GraphSAGE training.

set -euo pipefail

PROJECT_ROOT="${HOME}/gnn-modelzoo-v2.6"
# shellcheck source=./common.sh
# Assuming common.sh exists and is compatible, otherwise we might need to adjust or remove this source.
# Based on run_modelzoo.sh, it seems to set up logging and env.
# However, PROJECT_ROOT in run_modelzoo.sh was set to "${HOME}/gnn-modelzoo". 
# The current workspace is "${HOME}/gnn-modelzoo-v2.6".
# I will use the current workspace path.

if [ -f "${PROJECT_ROOT}/common.sh" ]; then
    source "${PROJECT_ROOT}/common.sh"
else
    # Fallback logging if common.sh is missing
    log_info() { echo "[INFO] $1"; }
    log_error() { echo "[ERROR] $1" >&2; }
fi

SCRIPT_NAME="$(basename "$0")"

PYTHON_SCRIPT_NAME="src/cerebras/modelzoo/models/gnn/graphsage_pyg.py"
CONFIG_PATH="src/cerebras/modelzoo/models/gnn/configs/params_graphsage_ogbn_arxiv.yaml"
# Arguments for PYTHON_SCRIPT_NAME
PYTHON_SCRIPT_ARGS=("--config" "${CONFIG_PATH}")

main() {
    log_info "Starting PyG GraphSAGE run"

    if ! command -v uv &> /dev/null; then
        log_error "'uv' command not found. Please install uv: https://github.com/astral-sh/uv"
        return 1
    fi

    # We run from PROJECT_ROOT
    local model_run_dir="${PROJECT_ROOT}"
    
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
