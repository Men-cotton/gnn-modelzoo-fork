#!/bin/bash
# common.sh
# Shared utility functions and constants.

if [[ "${BASH_SOURCE[0]}" -ef "$0" ]]; then
    echo "Error: This script is meant to be sourced, not executed directly." >&2
    exit 1
fi

_log_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log_info() {
    echo "$(_log_timestamp) [INFO] $*"
}

log_error() {
    echo "$(_log_timestamp) [ERROR] $*" >&2
}

log_warn() {
    echo "$(_log_timestamp) [WARN] $*" >&2
}

SHARED_MODEL_SUBDIR="src/cerebras/modelzoo/models/gnn"

# A pinned run uses its own project tree; dependencies remain a separate binding.
if [[ -n "${JOB_SOURCE_ROOT:-}" ]]; then
    [[ "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" == "$JOB_SOURCE_ROOT" ]] || { echo 'Pinned source/common.sh mismatch' >&2; return 2; }
    export PYTHONPATH="$JOB_SOURCE_ROOT/src"
    export UV_NO_SYNC=1 UV_OFFLINE=1
fi
