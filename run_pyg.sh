#!/bin/bash
# Executes the PyG GraphSAGE training.

set -euo pipefail

# Resolve PROJECT_ROOT based on this script's location (robust to differing CWD).
if command -v readlink &> /dev/null; then
    SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]:-$0}" 2>/dev/null || true)"
fi
SCRIPT_PATH="${SCRIPT_PATH:-${BASH_SOURCE[0]:-$0}}"
PROJECT_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd -P)"

if [ -f "${PROJECT_ROOT}/common.sh" ]; then
    source "${PROJECT_ROOT}/common.sh"
else
    # Fallback logging if common.sh is missing
    log_info() { echo "[INFO] $1"; }
    log_error() { echo "[ERROR] $1" >&2; }
fi

SCRIPT_NAME="$(basename "$0")"

PYTHON_SCRIPT_NAME="src/cerebras/modelzoo/models/gnn/graphsage_pyg.py"
CONFIG_PATH=""
# Arguments for PYTHON_SCRIPT_NAME (populated after arg parsing)
PYTHON_SCRIPT_ARGS=()

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} --config <path>
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
}

main() {
    log_info "Starting PyG GraphSAGE run"

    if ! command -v uv &> /dev/null; then
        log_error "'uv' command not found. Please install uv: https://github.com/astral-sh/uv"
        return 1
    fi

    parse_args "$@"
    local parse_status=$?
    if [[ "${parse_status}" -eq 0 && -z "${CONFIG_PATH}" ]]; then
        return 0
    fi
    if [[ "${parse_status}" -ne 0 ]]; then
        return "${parse_status}"
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

    # We run from PROJECT_ROOT
    local model_run_dir="${PROJECT_ROOT}"
    
    log_info "Executing model script in: ${model_run_dir}"
    ( # Subshell for model execution
        cd "${model_run_dir}"
        local full_command=("${PYTHON_SCRIPT_NAME}" "${PYTHON_SCRIPT_ARGS[@]}")
        log_info "Executing: uv run ${full_command[*]}"
        uv run --python "${PROJECT_ROOT}/.venv/bin/python" -- "${full_command[@]}"
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
