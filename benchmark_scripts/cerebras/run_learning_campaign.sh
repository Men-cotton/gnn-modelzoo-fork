#!/usr/bin/env bash
# Compatibility entrypoint; the campaign retains scientific planning and locks.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
export PATH="${project_root}/.venv/bin:${PATH}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${project_root}/.venv/bin/python" -m cerebras.modelzoo.tools.benchmark_launcher jobs --job csx-learning -- "$@"
