#!/bin/bash
# Sets up Python virtual environment, installs dependencies, and pre-downloads datasets.
# Run once from project root on a node with internet access.

set -euo pipefail

# shellcheck source=./common.sh
source "$(dirname "$0")/common.sh"

PROJECT_ROOT="$(pwd)"
SCRIPT_NAME="$(basename "$0")"

VENV_DIR_NAME=".venv"
VENV_PATH="${PROJECT_ROOT}/${VENV_DIR_NAME}"
SETUP_MARKER_FILE="${VENV_PATH}/.setup_successful"
PYTHON_VERSION_TARGET="3.11"

DOWNLOAD_SCRIPT_NAME="download.py"

log_step() {
    echo -e "\n$(_log_timestamp) --- $* ---"
}

main() {
    log_step "Starting GNN ModelZoo Setup"
    log_info "Project Root: ${PROJECT_ROOT}, Target Venv: ${VENV_PATH}, Python: ${PYTHON_VERSION_TARGET}"

    if ! command -v uv &> /dev/null; then
        log_error "'uv' command not found. Please install uv: https://github.com/astral-sh/uv"
        return 1
    fi

    log_step "Checking for conflicting environment modules"
    local target_module="intelpython/2022.3.1" # Specific module to check
    if command -v module &> /dev/null; then
        if module list 2>&1 | grep -qw "$target_module"; then
            log_info "Module '${target_module}' is loaded. Attempting to unload..."
            if module unload "$target_module"; then
                log_info "'${target_module}' unloaded successfully."
            else
                log_error "Failed to unload '${target_module}'. Continuing..." # Non-fatal
            fi
        else
            log_info "Module '${target_module}' not loaded."
        fi
    else
        log_info "'module' command not found. Skipping module check for '${target_module}'."
    fi

    unset PYTHONPATH
    rm -f "${SETUP_MARKER_FILE}"

    local python_specifier="python${PYTHON_VERSION_TARGET}"

    log_step "Creating Virtual Environment (Python ${PYTHON_VERSION_TARGET})"
    if ! uv venv "${VENV_PATH}" -p "${python_specifier}"; then
        log_error "'uv venv' failed. Ensure Python ${PYTHON_VERSION_TARGET} is discoverable."
        log_error "Try: 'uv python list' or 'uv python install python@${PYTHON_VERSION_TARGET}'."
        return 1
    fi
    log_info "Venv created/updated: '${VENV_PATH}'"

    log_step "Installing Packages"
    log_info "Installing dependencies from req.txt..."
    if ! uv pip install -r req.txt; then
        log_error "Requirements install failed."
        return 1
    fi

    log_step "Pre-downloading GNN Datasets"
    local download_dir="${PROJECT_ROOT}/${SHARED_MODEL_SUBDIR}"
    if [ ! -d "${download_dir}" ]; then
        log_error "Download script directory not found: ${download_dir}"
        return 1
    fi

    log_info "Attempting dataset download in '${download_dir}'..."
    ( # Subshell for dataset download
        cd "${download_dir}"
        if [ ! -f "${DOWNLOAD_SCRIPT_NAME}" ]; then
            log_error "Download script not found: $(pwd)/${DOWNLOAD_SCRIPT_NAME}"
            exit 1
        fi
        log_info "Running dataset download script: ${DOWNLOAD_SCRIPT_NAME}"
        uv run "${DOWNLOAD_SCRIPT_NAME}"
        log_info "Dataset download script finished."
    )
    local download_status=$?

    if [ "${download_status}" -ne 0 ]; then
        log_error "Dataset download process failed (status: ${download_status})."
        return "${download_status}"
    fi
    log_info "Dataset download process completed."

    touch "${SETUP_MARKER_FILE}"
    log_step "Setup Complete: Venv '${VENV_PATH}' is ready."
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
