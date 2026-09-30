#!/usr/bin/env bash
# Compatibility entrypoint; plans and submission live in benchmark_launcher.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${project_root}/.venv/bin/python" -m cerebras.modelzoo.tools.benchmark_launcher jobs --job pyg -- "$@"
