#!/usr/bin/env bash
# Add NLP dependencies to an existing environment without recreating it.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
venv_python="${project_root}/.venv/bin/python"
[[ -x "$venv_python" ]] || {
    echo 'Create .venv with ./setup.sh --target-env gpu (or csx) first.' >&2
    exit 2
}
command -v uv >/dev/null || { echo 'uv is required to install dependencies.' >&2; exit 2; }
# Read the installed build, rather than inferring CUDA from this login node's devices.
torch_build="$(uv run --no-sync --project "$project_root" python - <<'PY'
import torch
if torch.__version__.split('+')[0] != '2.4.0':
    raise SystemExit('Expected the project PyTorch 2.4.0 environment; refusing to replace PyTorch.')
if torch.version.hip:
    raise SystemExit('This benchmark setup supports CPU or CUDA builds, not ROCm.')
flavor = 'cu' + torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'
print(torch.__version__ + ' ' + flavor)
PY
)"
read -r torch_version torch_flavor <<< "$torch_build"
constraints="$(mktemp)"
trap 'rm -f -- "$constraints"' EXIT
printf 'torch===%s\ntorchvision==0.19.0+%s\n' "$torch_version" "$torch_flavor" > "$constraints"
uv pip install --python "$venv_python" \
    --constraint "$constraints" \
    --extra-index-url "https://download.pytorch.org/whl/${torch_flavor}" \
    -r "${script_dir}/requirements.txt"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
uv run --no-sync --project "$project_root" python "${script_dir}/run.py" --backend CSX --check-dependencies
