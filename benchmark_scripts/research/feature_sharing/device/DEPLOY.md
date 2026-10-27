# Fixed-shape shared features: CSX admission pilot

This pilot streams the same sampled GraphSAGE tree in two representations:
expanded FP16 features, or one FP16 dictionary plus int32 slot references.
The shared version reconstructs the original padded tree with `torch.gather`
inside the model. Aggregation, masks, logical root batch, initial weights,
AdamW, and loss are identical. Padding and dense GraphSAGE compute remain.
Execution uses `cerebras.pytorch` directly with the native ModelZoo model;
the ModelZoo Trainer configuration path is not exercised by this pilot.
No CSL kernels or custom device primitives are required by the source.
Whether the complete gather graph compiles and runs efficiently on CSX must
be decided by the remote compile and execution phases below.

The bounded input is the first two and tail batches from a fixed seed. It is
not an epoch or an online-sampling implementation. Dictionary capacity covers
only this frozen window. A new window or batch size requires new preparation
and compilation. Overflow is rejected rather than truncated or resampled.
Paired admission uses dropout=0; compiler RNG parity for dropout is untested.

## Fork checkout and real inputs

Run from the shared gnn-modelzoo-fork checkout after site authentication.
The native GraphSAGE, batch and loss implementations are imported from this
fork and hash-checked against the prepared data. No runtime tar or vendored
native model is used. Preparation supports arxiv and products only.

```bash
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN="$PWD/benchmark_scripts/research/run.sh"
ROOT=/shared/project/shared-tree-features
mkdir -p "$ROOT"
"$RUN" feature_sharing.device.prepare --dataset arxiv --batch-size 128 --fanouts 15 10 5 \
  --hidden 128 --dropout 0 --output "$ROOT/input"
"$RUN" feature_sharing.device.preflight --input "$ROOT/input" --output "$ROOT/preflight.json"
cp benchmark_scripts/research/feature_sharing/device/cluster.example.json "$ROOT/cluster.json"
```

Set the site's management endpoint, shared mount and `python_paths` containing
both the fork root and its `src` directory. Keep one CSX/input Worker. Inputs
and checkout must be visible at identical absolute paths on client/Worker.
Preparation uses the entire real graph and freezes first, second and tail
batches. These three batches test representation equality, not an epoch or
end-to-end learning speed. B4096 needs separate preparation/compilation.

## Existing tmux launcher

Run each phase only after inspecting the preceding phase's result. These
commands submit CSX work when run on the configured site. They use the
existing launcher with a dedicated tmux socket and a fresh output directory;
they do not reuse or terminate other agents' sessions.

```bash
# Complete remote compilation of both representations, without execution.
"$RUN" launcher \
  --name shared-tree-compile --output "$ROOT/jobs/compile-launcher" -- \
  "$RUN" feature_sharing.device.pilot --phase compile --backend CSX --precision bf16 \
  --input "$ROOT/input" --output "$ROOT/jobs/compile" \
  --cluster-config "$ROOT/cluster.json" --compile-dir "$ROOT/compile-cache"

# Three training steps per representation; captures logits and final states.
"$RUN" launcher \
  --name shared-tree-verify --output "$ROOT/jobs/verify-launcher" -- \
  "$RUN" feature_sharing.device.pilot --phase verify --backend CSX --precision bf16 \
  --input "$ROOT/input" --output "$ROOT/jobs/verify" \
  --cluster-config "$ROOT/cluster.json" --compile-dir "$ROOT/compile-cache"

# Repeats the same three-input window for 30 steps; requires passed verification.
"$RUN" launcher \
  --name shared-tree-measure --output "$ROOT/jobs/measure-launcher" -- \
  "$RUN" feature_sharing.device.pilot --phase measure --backend CSX --precision bf16 \
  --steps 30 --warmup-steps 3 --input "$ROOT/input" --output "$ROOT/jobs/measure" \
  --verification "$ROOT/jobs/verify/pilot.json" \
  --cluster-config "$ROOT/cluster.json" --compile-dir "$ROOT/compile-cache"
```

Add `--dry-run` to **pilot.py** (not the launcher) to print commands without
submitting work. Compilation and verification use different capture graphs,
so the verify job may require another compilation even with a common cache.
Dense and dictionary jobs run sequentially, never concurrently on the CSX.

The launcher prints the exact attach, status and log commands. Inspect both
`jobs/<phase>-launcher/launcher.json` (`completed`, exit 0) and
`jobs/<phase>/pilot.json`. A detached launch returning 0 is not job success.
Compile success is `compiled_not_executed`; verify/measure success is `passed`.
An exception in either representation stops the pair and records `failed`.
Inspect `<phase>/dense.log`, `dictionary.log`, and each runtime directory for
compiler failures and device metrics. Preserve failed runs; use new paths to retry.

Verification compares the ordered sample IDs, valid root counts, every loss
and logit, final model weights, and optimizer state. FP32 defaults to exact
comparison. BF16 uses explicit rtol=0.01/atol=0.01; the result also records
observed maximum differences. A BF16 tolerance pass is not bitwise equality
or a proof of accuracy over training. `--rtol` and `--atol` can tighten these
bounds; do not silently loosen them after a failure.

Timing excludes the initial warmup activation receipts but includes client
capture overhead and input delivery. It is not isolated device time or an
end-to-end online sampler measurement. Use SDK device metrics and repeated,
order-balanced measurements before making a performance claim. Dictionary
input-byte reduction alone does not establish CSX memory savings or speedup:
the compiler can materialize the expanded tree and widen gather references.

## Local checks

`preflight.py` checks exact FP32 reconstructed inputs, logits, native masked
classification loss and all parameter gradients for every frozen input.
It also reuses one `torch.export` graph on changed dictionaries/references and
the tail mask. This is separate from the Cerebras Python export bridge,
WAF/code generation, memory placement and hardware execution.

The same `pilot.py --backend CPU --precision fp32` exercises SDK DataLoader,
DataExecutor, training, activations, final checkpoint and paired comparison.
It reads the prepared input directory directly. CPU success must not be
reported as CSX compilation or execution success.

## uv entrypoint

`benchmark_scripts/research/run.sh` sets the fork's import paths and calls
`uv run --no-sync --project FORK_ROOT`. For example:

```bash
benchmark_scripts/research/run.sh feature_sharing.device.pilot --help
```

The paired driver also starts both training subprocesses through the same
uv project. Existing `pyproject.toml` and `uv.lock` own dependencies; provision
the site environment with the repository's uv setup before running jobs.
`--no-sync` prevents a job from changing the installed SDK or downloading
dependencies. Remote Workers still need the same site-provisioned SDK.
