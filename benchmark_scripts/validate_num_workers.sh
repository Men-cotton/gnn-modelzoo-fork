#!/usr/bin/env bash
# Validate the shared GraphSAGE neighbor dataloader worker count.

set -euo pipefail

configs_dir="${1:?usage: validate_num_workers.sh <configs-dir>}"
config_path="${configs_dir}/components/architectures/input_pipelines/neighbor.yaml"

actual="$(awk -F: '$1 ~ /^[[:space:]]*num_workers[[:space:]]*$/ { sub(/#.*/, "", $2); gsub(/[[:space:]]/, "", $2); print $2; exit }' "${config_path}")"
if ! [[ "${actual}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ${config_path}: num_workers=${actual:-missing}, expected a non-negative integer" >&2
    exit 1
fi
