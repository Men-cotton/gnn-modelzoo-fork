#!/bin/bash
# Sets up Python virtual environment, installs dependencies, and pre-downloads datasets.
# Run once from project root on a node with internet access.

set -euo pipefail

# shellcheck source=./common.sh
source "$(dirname "$0")/common.sh"

PROJECT_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd -P)"
SCRIPT_NAME="$(basename "$0")"

VENV_DIR_NAME=".venv"
VENV_PATH="${PROJECT_ROOT}/${VENV_DIR_NAME}"
SETUP_MARKER_FILE="${VENV_PATH}/.setup_successful"
PYTHON_VERSION_TARGET="3.11"

DOWNLOAD_SCRIPT_PATH="src/cerebras/modelzoo/models/gnn/tools/download_datasets.py"

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

    log_step "Initializing Project and Installing Packages"

    if [ ! -f "pyproject.toml" ]; then
        log_info "pyproject.toml not found. Initializing..."
        if ! uv init --python "${PYTHON_VERSION_TARGET}" --no-readme; then
            log_error "uv init failed."
            return 1
        fi
        # Restrict Python version to avoid solving for incompatible newer versions (e.g. 3.12)
        sed -i 's/requires-python = ">=3.11"/requires-python = "==3.11.*"/' pyproject.toml
        log_info "Updated pyproject.toml requires-python to ==3.11.*"
    fi

    log_step "Creating virtual environment..."
    if ! uv venv --python "${PYTHON_VERSION_TARGET}" "${VENV_PATH}"; then
        log_error "Failed to create virtual environment."
        return 1
    fi

    log_info "Adding dependencies from req.txt..."
    # Extract find-links if present, as uv add might strict-mode ignore it inside file but needs it for lookup
    local find_links_arg=""
    if grep -q "^--find-links" req.txt; then
        local url
        url=$(grep -m 1 "^--find-links" req.txt | awk '{print $2}')
        find_links_arg="--find-links ${url}"
        log_info "Detected --find-links: ${url}"
    fi

    if ! uv add --python "${VENV_PATH}/bin/python" -r req.txt ${find_links_arg}; then
        log_error "Requirements install failed (uv add)."
        return 1
    fi



    log_info "Removing 'outdated' package to prevent deprecation warnings..."
    uv pip uninstall --python "${VENV_PATH}/bin/python" outdated || true

    log_step "Pre-downloading GNN Datasets"
    local download_script_full_path="${PROJECT_ROOT}/${DOWNLOAD_SCRIPT_PATH}"
    if [ ! -f "${download_script_full_path}" ]; then
        log_error "Download script not found: ${download_script_full_path}"
        return 1
    fi

    log_info "Running dataset download script: ${DOWNLOAD_SCRIPT_PATH}"
    if ! uv run --python "${VENV_PATH}/bin/python" "${download_script_full_path}"; then
        log_error "Dataset download process failed."
        return 1
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
