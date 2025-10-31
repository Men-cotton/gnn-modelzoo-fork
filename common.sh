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

SHARED_MODEL_SUBDIR="src/cerebras/modelzoo/models/gnn"
