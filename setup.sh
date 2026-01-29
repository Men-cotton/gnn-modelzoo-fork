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
TORCH_CPU_SPEC_DEFAULT="torch==2.4.0"
TORCH_CPU_INDEX_URL_DEFAULT="https://download.pytorch.org/whl/cpu"
TORCH_CUDA_SPEC_DEFAULT="torch==2.4.0+cu121"
TORCH_CUDA_INDEX_URL_DEFAULT="https://download.pytorch.org/whl/cu121"

DOWNLOAD_SCRIPT_PATH="src/cerebras/modelzoo/models/gnn/tools/download_datasets.py"

log_step() {
    echo -e "\n$(_log_timestamp) --- $* ---"
}

detect_cuda_toolkit_root() {
    local root=""
    if [[ -n "${CUDA_TOOLKIT_ROOT_DIR:-}" ]]; then
        root="${CUDA_TOOLKIT_ROOT_DIR}"
    elif [[ -n "${CUDA_HOME:-}" ]]; then
        root="${CUDA_HOME}"
    elif command -v nvcc &> /dev/null; then
        root="$(dirname "$(dirname "$(command -v nvcc)")")"
    fi
    if [[ -n "${root}" ]] && [[ -x "${root}/bin/nvcc" ]]; then
        echo "${root}"
        return 0
    fi
    echo ""
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

    local VENV_PYTHON="${VENV_PATH}/bin/python"
    local python_include=""
    local python_prefix=""
    python_include="$("${VENV_PYTHON}" - <<'PY'
import sysconfig
print(sysconfig.get_path("include") or "")
PY
)"
    python_prefix="$("${VENV_PYTHON}" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("prefix") or "")
PY
)"
    if [[ -n "${python_include}" ]]; then
        export CMAKE_ARGS="${CMAKE_ARGS:-} -DPython3_EXECUTABLE=${VENV_PYTHON} -DPython3_INCLUDE_DIR=${python_include}"
        if [[ -n "${python_prefix}" ]]; then
            export CMAKE_ARGS="${CMAKE_ARGS} -DPython3_ROOT_DIR=${python_prefix}"
        fi
    fi

    log_step "Installing build tools (required for source builds)"
    if ! uv pip install --python "${VENV_PYTHON}" "setuptools>=65" "wheel>=0.41"; then
        log_error "Build tools install failed."
        return 1
    fi

    log_step "Installing PyTorch (auto-select CPU/GPU based on CUDA toolkit)"
    if [[ "${SKIP_TORCH_INSTALL:-0}" != "1" ]]; then
        local cuda_root=""
        local cuda_detected="0"
        local torch_spec="${TORCH_CPU_SPEC_DEFAULT}"
        local torch_index_url="${TORCH_CPU_INDEX_URL_DEFAULT}"
        cuda_root="$(detect_cuda_toolkit_root)"
        if [[ -n "${cuda_root}" ]]; then
            cuda_detected="1"
            torch_spec="${TORCH_CUDA_SPEC_DEFAULT}"
            torch_index_url="${TORCH_CUDA_INDEX_URL_DEFAULT}"
            log_info "CUDA toolkit detected; using CUDA torch."
            export CUDA_HOME="${cuda_root}"
            export CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDA_TOOLKIT_ROOT_DIR=${cuda_root}"
        else
            log_info "CUDA toolkit not found; using CPU torch."
        fi
        local torch_install_args=("${torch_spec}")
        if [[ -n "${torch_index_url}" ]]; then
            torch_install_args+=(--index-url "${torch_index_url}")
        fi
        if [[ -n "${TORCH_EXTRA_INDEX_URL:-}" ]]; then
            torch_install_args+=(--extra-index-url "${TORCH_EXTRA_INDEX_URL}")
        fi
        log_info "Installing torch with: ${torch_spec}"
        if ! uv pip install --python "${VENV_PYTHON}" "${torch_install_args[@]}"; then
            log_error "PyTorch install failed."
            return 1
        fi

        if [[ "${cuda_detected}" = "1" ]]; then
            log_step "Installing pyg-lib (CUDA detected)"
            local pyg_find_links_arg=""
            if grep -q "^--find-links" req.txt; then
                local url
                url=$(grep -m 1 "^--find-links" req.txt | awk '{print $2}')
                pyg_find_links_arg="--find-links ${url}"
            fi
            if ! uv pip install --python "${VENV_PYTHON}" "pyg-lib==0.4.0" ${pyg_find_links_arg}; then
                log_error "pyg-lib install failed."
                return 1
            fi
        fi
    else
        log_info "Skipping torch install (SKIP_TORCH_INSTALL=1)."
    fi

    log_info "Adding dependencies from req.txt..."
    # Extract find-links if present; keep explicit for uv pip install.
    local find_links_arg=""
    if grep -q "^--find-links" req.txt; then
        local url
        url=$(grep -m 1 "^--find-links" req.txt | awk '{print $2}')
        find_links_arg="--find-links ${url}"
        log_info "Detected --find-links: ${url}"
    fi

    if ! uv pip install --python "${VENV_PYTHON}" --no-build-isolation -r req.txt ${find_links_arg}; then
        log_error "Requirements install failed (uv pip install)."
        return 1
    fi

    log_info "Installing project in editable mode..."
    if ! uv pip install --python "${VENV_PYTHON}" -e . --no-deps; then
        log_error "Editable install failed."
        return 1
    fi


    log_info "Removing 'outdated' package to prevent deprecation warnings..."
    uv pip uninstall --python "${VENV_PYTHON}" outdated || true

    log_step "Pre-downloading GNN Datasets"
    local download_script_full_path="${PROJECT_ROOT}/${DOWNLOAD_SCRIPT_PATH}"
    if [ ! -f "${download_script_full_path}" ]; then
        log_error "Download script not found: ${download_script_full_path}"
        return 1
    fi

    log_info "Running dataset download script: ${DOWNLOAD_SCRIPT_PATH}"
    if ! uv run --python "${VENV_PYTHON}" "${download_script_full_path}"; then
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
