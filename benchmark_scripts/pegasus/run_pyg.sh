#!/bin/bash
# Executes a PyG GNN training run.

set -euo pipefail

# Resolve PROJECT_ROOT based on this script's location (robust to differing CWD).
if command -v readlink &> /dev/null; then
    SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]:-$0}" 2>/dev/null || true)"
fi
SCRIPT_PATH="${SCRIPT_PATH:-${BASH_SOURCE[0]:-$0}}"
PEGASUS_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd -P)"
PROJECT_ROOT="$(cd "${PEGASUS_DIR}/../.." && pwd -P)"

# shellcheck source=../../common.sh
source "${PROJECT_ROOT}/common.sh"
# shellcheck source=./gpu_env.sh
source "${PEGASUS_DIR}/gpu_env.sh"

SCRIPT_NAME="$(basename "$0")"

PYTHON_SCRIPT_NAME=""
CONFIG_PATH=""
NO_COMPILE_FLAG=0
# Arguments for PYTHON_SCRIPT_NAME (populated after arg parsing)
PYTHON_SCRIPT_ARGS=()

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} --config <path> [--no-compile]
EOF
}

parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --config)
                if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
                    log_error "Missing value for --config"
                    usage
                    return 2
                fi
                CONFIG_PATH="$2"
                shift 2
                ;;
            --no-compile)
                NO_COMPILE_FLAG=1
                shift
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

    if [[ -z "${CONFIG_PATH}" ]]; then
        log_error "Missing required --config <path>"
        usage
        return 2
    fi

    PYTHON_SCRIPT_ARGS=("--config" "${CONFIG_PATH}")

    case "$(basename "${CONFIG_PATH}")" in
        params_graphsage_*)
            PYTHON_SCRIPT_NAME="src/cerebras/modelzoo/models/gnn/pyg_graphsage.py"
            ;;
        params_gcn_*)
            PYTHON_SCRIPT_NAME="src/cerebras/modelzoo/models/gnn/pyg_gcn.py"
            ;;
        *)
            log_error "Unsupported PyG config for script selection: ${CONFIG_PATH}"
            log_error "Expected config basename to start with params_graphsage_ or params_gcn_"
            return 2
            ;;
    esac
}

log_run_metadata() {
    local venv_python="${PROJECT_ROOT}/.venv/bin/python"

    log_info "Run metadata: script=${SCRIPT_NAME}"
    log_info "Run metadata: python_script=${PYTHON_SCRIPT_NAME}"
    log_info "Run metadata: project_root=${PROJECT_ROOT}"
    log_info "Run metadata: config=${CONFIG_PATH}"
    log_info "Run metadata: no_compile=${NO_COMPILE_FLAG}"
    log_info "Run metadata: hostname=$(hostname 2>/dev/null || echo unknown)"
    log_info "Run metadata: CUDA_HOME=${CUDA_HOME:-}"
    log_info "Run metadata: CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR:-}"
    log_info "Run metadata: PYTHONPATH=${PYTHONPATH:-}"
    if command -v git &> /dev/null; then
        log_info "Run metadata: git_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
        log_info "Run metadata: git_status_short_begin"
        git -C "${PROJECT_ROOT}" status --short 2>/dev/null || true
        log_info "Run metadata: git_status_short_end"
    fi
    if command -v nvidia-smi &> /dev/null; then
        log_info "Run metadata: nvidia-smi_begin"
        nvidia-smi || true
        log_info "Run metadata: nvidia-smi_end"
    fi
    if command -v nvcc &> /dev/null; then
        log_info "Run metadata: nvcc_begin"
        nvcc --version || true
        log_info "Run metadata: nvcc_end"
    fi

    if [[ -x "${venv_python}" ]]; then
        local metadata_args=(
            python
            "${PROJECT_ROOT}/benchmark_scripts/pyg_run_metadata.py"
            --config "${CONFIG_PATH}"
        )
        if [[ "${NO_COMPILE_FLAG}" -eq 1 ]]; then
            metadata_args+=(--no-compile)
        fi
        log_info "Run metadata: python_helper_begin"
        uv run --no-sync --python "${venv_python}" -- "${metadata_args[@]}" || true
        log_info "Run metadata: python_helper_end"
    else
        log_error "Virtual environment python not found: ${venv_python}"
        return 1
    fi
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

    log_info "Starting PyG GNN run"

    if ! command -v uv &> /dev/null; then
        log_error "'uv' command not found. Please install uv: https://github.com/astral-sh/uv"
        return 1
    fi

    log_info "Checking for conflicting environment modules"
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

    log_info "Loading CUDA toolkit module for production GPU run"
    if ! load_cuda_module; then
        log_error "CUDA module load failed. Refusing to start the production GPU run."
        return 3
    fi
    if ! require_cuda_toolkit; then
        log_error "CUDA toolkit is not visible. Refusing to start the production GPU run."
        return 3
    fi

    if [[ ! -f "${PROJECT_ROOT}/.venv/.setup_successful" ]]; then
        log_error "Setup marker not found: ${PROJECT_ROOT}/.venv/.setup_successful"
        log_error "Run './setup.sh --target-env gpu' on Pegasus before launching production runs."
        return 4
    fi

    if [[ "${NO_COMPILE_FLAG}" -eq 1 ]]; then
        export NO_COMPILE=1
    fi
    export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

    log_run_metadata

    # We run from PROJECT_ROOT
    local model_run_dir="${PROJECT_ROOT}"

    log_info "Executing model script in: ${model_run_dir}"
    ( # Subshell for model execution
        cd "${model_run_dir}"
        local full_command=("${PYTHON_SCRIPT_NAME}" "${PYTHON_SCRIPT_ARGS[@]}")
        log_info "Executing: uv run ${full_command[*]}"
        uv run --no-sync --python "${PROJECT_ROOT}/.venv/bin/python" -- "${full_command[@]}"
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
