#!/usr/bin/env bash
# Short R04 probes: 40 warmup steps, then 40 measured steps, at 2 and 40 workers.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE="$PROJECT_ROOT/model_dirs/hpcasia_r04/arxiv_none/sensitivity_w02_p2_s1_r1/params.yaml"
DRY_RUN=false

usage() {
    cat <<'HELP'
Usage: bash benchmark_scripts/cerebras/run_worker_diagnostics.sh [options]

Run two sequential CSX diagnostics using the saved R04 arxiv/no-cache settings.
  --base FILE   Saved R04 params.yaml (relative to the caller or absolute)
  --dry-run     Generate configs and print commands without submitting jobs
  -h, --help    Show this help

Each invocation creates model_dirs/hpcasia_r04/worker_diag_<UTC>_<unique>/.
The repository and output directory must be visible to the remote Worker.
Diagnostic throughput includes instrumentation overhead; use it for diagnosis.
HELP
}
while (($#)); do
    case "$1" in
        --base)
            if (($# < 2)); then
                printf 'Missing FILE after --base\n' >&2
                exit 2
            fi
            BASE="$2"
            shift 2
            ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ ! -f "$BASE" ]]; then
    printf 'Saved R04 configuration not found: %s\nUse --base FILE to select it.\n' "$BASE" >&2
    exit 2
fi
BASE="$(realpath -- "$BASE")"
cd "$PROJECT_ROOT/src/cerebras/modelzoo/models/gnn"

# Resolve inheritance before copying, so relative extends paths remain valid.
# Pass paths as arguments and use a YAML serializer for quoting.
DIAG_ROOT=$(uv run --no-sync python - "$BASE" "$PROJECT_ROOT" "$DRY_RUN" <<'PY'
import copy
import datetime
import pathlib
import sys
import tempfile

import yaml

from cerebras.modelzoo.common.utils.run.config_loader import load_params_file
from cerebras.modelzoo.models.gnn.data_processing.worker_diagnostics_config import (
    WorkerDiagnosticsConfig,
)
from cerebras.modelzoo.models.gnn.worker_validation import validate_num_workers

base_path, project_root, dry_run = sys.argv[1:]
base = load_params_file(base_path)
loader = base["trainer"]["fit"]["train_dataloader"]
if loader["batch_size"] != 4096 or loader["sampling_mode"] != "neighbor":
    raise ValueError("Expected an R04 neighbor configuration with batch_size=4096")
if dry_run != "true":
    validate_num_workers(40, context="R04 diagnostics preflight")
parent = pathlib.Path(project_root) / "model_dirs/hpcasia_r04"
parent.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
root = pathlib.Path(tempfile.mkdtemp(prefix=f"worker_diag_{stamp}_", dir=parent))
(root / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
(root / "base-source.txt").write_text(base_path + "\n")
for workers in (2, 40):
    run = root / f"w{workers:02d}"
    run.mkdir()
    params = copy.deepcopy(base)
    params["trainer"]["init"]["model_dir"] = str(run / "model")
    params["trainer"]["init"]["loop"]["max_steps"] = 80
    train_loader = params["trainer"]["fit"]["train_dataloader"]
    train_loader["num_workers"] = workers
    train_loader["worker_diagnostics"] = WorkerDiagnosticsConfig(
        enabled=True,
        output_dir=str(run / "worker_diagnostics"),
        max_batches=80,
        snapshot_interval_seconds=5.0,
        max_snapshots=80,
    ).model_dump()
    (run / "params.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
print(root)
PY
)
git rev-parse HEAD > "$DIAG_ROOT/git-revision.txt"
hostname > "$DIAG_ROOT/client-hostname.txt"
printf 'Output: %s\n' "$DIAG_ROOT"
for w in 02 40; do
    RUN="$DIAG_ROOT/w$w"
    COMMAND=(uv run --no-sync -- cszoo fit "$RUN/params.yaml"
             --target_device CSX --model_dir "$RUN/model")
    if [[ "$DRY_RUN" == true ]]; then
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
        continue
    fi
    "${COMMAND[@]}" 2>&1 | tee "$RUN/train.log"
    uv run --no-sync tools/measure_window.py "$RUN/train.log" \
        --start-step 40 --end-step 80 --batch-size 4096 > "$RUN/throughput.json"
done
printf 'Completed%s: %s\n' "$(if [[ "$DRY_RUN" == true ]]; then printf ' (dry-run)'; fi)" "$DIAG_ROOT"
