#!/usr/bin/env bash
# Create a uv-managed Python environment for this checkout.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_PATH="${PROJECT_ROOT}/.venv"
PYTHON_VERSION_TARGET="${PYTHON_VERSION_TARGET:-3.11}"

log() {
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

main() {
    cd "${PROJECT_ROOT}"

    if ! command -v uv > /dev/null 2>&1; then
        echo "error: uv command not found. Install uv first: https://github.com/astral-sh/uv" >&2
        return 1
    fi

    unset PYTHONPATH

    local sync_args=(--python "${PYTHON_VERSION_TARGET}")
    if [[ "${UV_UPDATE_LOCK:-0}" != "1" ]]; then
        sync_args+=(--locked)
    fi

    log "Syncing ${VENV_PATH} with uv"
    uv sync "${sync_args[@]}"

    log "Verifying Model Zoo CLI"
    uv run cszoo --help > /dev/null

    log "Setup complete. Activate with: source .venv/bin/activate"
}

main "$@"
