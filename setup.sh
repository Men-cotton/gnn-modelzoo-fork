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
TARGET_ENV=""

DOWNLOAD_SCRIPT_PATH="src/cerebras/modelzoo/models/gnn/tools/download_datasets.py"

log_step() {
    echo -e "\n$(_log_timestamp) --- $* ---"
}

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} --target-env <gpu|csx>

  gpu  Pegasus/H100 setup. Tries module load cuda when available; requires a visible CUDA toolkit.
  csx  Cerebras/CSX setup. Allows CPU torch fallback because no local GPU is expected.
EOF
}

parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --target-env)
                if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
                    log_error "Missing value for --target-env"
                    usage
                    return 2
                fi
                TARGET_ENV="$2"
                shift 2
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

    case "${TARGET_ENV}" in
        gpu|csx)
            return 0
            ;;
        "")
            log_error "Missing required --target-env <gpu|csx>"
            usage
            return 2
            ;;
        *)
            log_error "Invalid --target-env '${TARGET_ENV}'. Expected 'gpu' or 'csx'."
            usage
            return 2
            ;;
    esac
}

load_pegasus_gpu_env() {
    if declare -F load_cuda_module > /dev/null && declare -F require_cuda_toolkit > /dev/null; then
        return 0
    fi

    # shellcheck source=./benchmark_scripts/pegasus/gpu_env.sh
    source "${PROJECT_ROOT}/benchmark_scripts/pegasus/gpu_env.sh"
}

main() {
    parse_args "$@"
    local parse_status=$?
    if [[ "${parse_status}" -eq 0 && -z "${TARGET_ENV}" ]]; then
        return 0
    fi
    if [[ "${parse_status}" -ne 0 ]]; then
        return "${parse_status}"
    fi

    log_step "Starting GNN ModelZoo Setup"
    log_info "Project Root: ${PROJECT_ROOT}, Target Venv: ${VENV_PATH}, Python: ${PYTHON_VERSION_TARGET}, Target Env: ${TARGET_ENV}"

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

    if [[ "${TARGET_ENV}" = "gpu" ]]; then
        load_pegasus_gpu_env
        log_step "Preparing CUDA toolkit environment"
        if ! load_cuda_module; then
            log_warn "CUDA module load failed or module command is unavailable; continuing with the current environment."
        fi
        if ! require_cuda_toolkit; then
            log_error "CUDA toolkit is required for the production PyG GPU benchmark setup. Set CUDA_HOME/CUDA_TOOLKIT_ROOT_DIR or make nvcc visible."
            return 3
        fi
    else
        log_step "Skipping CUDA toolkit module"
        log_info "CSX setup selected; local CUDA toolkit is not required."
    fi

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
    local python_build_paths=""
    python_build_paths="$("${VENV_PYTHON}" "${PROJECT_ROOT}/benchmark_scripts/python_build_paths.py")"
    while IFS='=' read -r key value; do
        case "${key}" in
            include)
                python_include="${value}"
                ;;
            prefix)
                python_prefix="${value}"
                ;;
        esac
    done <<< "${python_build_paths}"
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

    log_step "Installing PyTorch"
    if [[ "${SKIP_TORCH_INSTALL:-0}" != "1" ]]; then
        local torch_spec="${TORCH_CPU_SPEC_DEFAULT}"
        local torch_index_url="${TORCH_CPU_INDEX_URL_DEFAULT}"
        if [[ "${TARGET_ENV}" = "gpu" ]]; then
            load_pegasus_gpu_env
            local cuda_root=""
            cuda_root="$(detect_cuda_toolkit_root)"
            if [[ -z "${cuda_root}" ]]; then
                log_error "CUDA toolkit not found; refusing to install CPU torch for a production GPU benchmark."
                return 3
            fi
            torch_spec="${TORCH_CUDA_SPEC_DEFAULT}"
            torch_index_url="${TORCH_CUDA_INDEX_URL_DEFAULT}"
            log_info "CUDA toolkit detected; using CUDA torch."
            export CUDA_HOME="${cuda_root}"
            export CUDA_TOOLKIT_ROOT_DIR="${cuda_root}"
            export CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDA_TOOLKIT_ROOT_DIR=${cuda_root}"
        else
            log_info "CSX setup selected; using CPU torch for local setup tasks."
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

        if [[ "${TARGET_ENV}" = "gpu" ]]; then
            log_step "Installing pyg-lib (CUDA required)"
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
        else
            log_info "Skipping pyg-lib install for CSX setup."
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

    log_step "Installing and checking non-GNN benchmark dependencies"
    if ! bash "${PROJECT_ROOT}/benchmark_scripts/non_gnn/setup.sh"; then
        log_error "Non-GNN dependency setup failed."
        return 1
    fi

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
