# Real-data history and exact-preaggregation experiments

Run from the `gnn-modelzoo-fork` checkout containing this directory. This code,
the native model, and the existing `benchmark_launcher` belong to the same
checkout. Wafer-GNN-manager is used only to analyze saved results.
No standalone runtime archive or synthetic-data job is required.

## Site environment and preparation

After manual SSH/site authentication, activate the site's SDK 2.10 environment.
Use a shared checkout and output directory mounted at the same absolute paths
on the client and the single input Worker. Adapt these paths:

```bash
cd /shared/project/gnn-modelzoo-fork
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
DATA="$PWD/src/cerebras/modelzoo/models/gnn/data/datasets"
WORK=/shared/project/history-arxiv
MODULE=benchmark_scripts.research.history_embedding
mkdir -p "$WORK"
cp benchmark_scripts/research/history_embedding/cluster.example.json "$WORK/cluster.json"
```

Fill the management address and shared mount in `cluster.json`. Its
`python_paths` must include both the fork root (for `benchmark_scripts`) and
its `src` directory (for `cerebras.modelzoo`). Preserve one CSX and one input
Worker. The example placeholders are rejected. The script never authenticates
or submits work until the operator runs a CSX command.

Prepare the entire real graph, its official splits, a fixed sampled training
graph and initial snapshot. ogbn-arxiv has 169343 nodes and 90941 training
nodes. Preparation reads existing processed OGB data; it never downloads or
subsamples the node set. `ogbn-products` is also supported when installed.
Dropout zero makes the initial numerical comparison reproducible:

```bash
uv run --no-sync -- python -m "$MODULE.hardware" prepare --dataset ogbn-arxiv --data-dir "$DATA" \
  --hidden 128 --layers 3 --fanout 10 --dropout 0 --output "$WORK/input"
uv run --no-sync -- python -m "$MODULE.preflight" --input "$WORK/input" --batch-size 256 \
  --output "$WORK/preflight.json"
```

The bundle includes full adjacency and validation/test split IDs. Inputs may
be prepared on a local CPU and copied as data into this shared directory;
the execution code is always the fork checkout. Fresh output directories are
required. Logical B=256 nodes form one SDK sample; SDK sample/s is not node/s.

## Direct real-data execution with the existing tmux launcher

`execute` invokes the SDK's normal compile-and-run path. A separate compile
job is optional, not a prerequisite. Both cases use the same initial weights,
root order and fixed graph. The default comparison is recursive `tree` versus
exact raw-feature preaggregation `tree_mean`, supported by local accuracy
results. This is a 128-step window, not a complete training epoch.

```bash
uv run --no-sync -- python -m cerebras.modelzoo.tools.benchmark_launcher \
  --name history-arxiv --output "$WORK/measure.launcher" -- \
  uv run --no-sync -- python -m "$MODULE.pilot" --backend CSX --phase execute \
  --input "$WORK/input" --output "$WORK/measure" \
  --batch-size 256 --steps 128 --warmup-steps 20 --repeats 3 \
  --cluster-config "$WORK/cluster.json"
```

Use `--dry-run` on `pilot` to print commands without launching a job or creating
output. Add `--cases tree tree_mean history history-export` to measure history
reads and returns. Exported tensors are saved on the client and do not update
an active Worker snapshot. History training with these random batches has a
large measured accuracy gap; this optional comparison tests runtime costs.
All cases use activation cadence 1, so fetch cadence is matched.

The launcher's printed `Attach:` command connects to its dedicated tmux server.
Inspect `measure.launcher/launcher.json`, `driver.log`, `measure/pilot.json`
and every case's `result.json`. A detached launch is not job completion.
`client_observed_targets_per_s_after_warmup` counts observed supervised nodes;
it includes activation retrieval and capture-file I/O. It is not device-only
throughput or network bandwidth. `whole_process_s` includes SDK startup and
checkpoint time.

Numerical checking replays the **same real-data window** and SDK optimizer on
CPU, using the batch size, seed, learning rate and number of steps in the
receipt. It works for each representation; history captures are checked when
present. FP32 tolerances are configurable for device roundoff:

```bash
uv run --no-sync -- python -m "$MODULE.verify_sdk" --input "$WORK/input" \
  --epoch0 "$WORK/measure/tree_mean-r0" --atol 0.0002 --rtol 0.0001 \
  --output "$WORK/tree-mean-numerics.json"
uv run --no-sync -- python -m "$MODULE.evaluate" --bundle "$WORK/input/bundle.pt" \
  --checkpoint "$WORK/measure/tree_mean-r0/checkpoint.mdl" \
  --output "$WORK/window-validation.json"
```

The evaluator uses current checkpoint weights and the **full original graph**,
not stale history. Accuracy after a window must not be called epoch accuracy.
It uses the official validation split by default. Evaluate `--split test` only
for a checkpoint selected using validation, never to choose configurations.

## Complete learning epochs

For the prior learning configuration, prepare a separate input with
`--dropout 0.5`. This changes the numerical-comparison contract, so use a new
input/output directory:

```bash
uv run --no-sync -- python -m "$MODULE.hardware" prepare --dataset ogbn-arxiv \
  --data-dir "$DATA" --hidden 128 --layers 3 --fanout 10 --dropout 0.5 \
  --output "$WORK/learning-input"
```

A full epoch has ceil(90941/256)=356 batches, including a 61-node tail.
Every campaign epoch is followed by full-neighbor validation.

```bash
uv run --no-sync -- python -m cerebras.modelzoo.tools.benchmark_launcher \
  --name history-learning --output "$WORK/learning.launcher" -- \
  uv run --no-sync -- python -m "$MODULE.campaign" --backend CSX --mode tree_mean --epochs 10 \
  --input "$WORK/learning-input" --output "$WORK/learning" \
  --batch-size 256 --cluster-config "$WORK/cluster.json"
```

`campaign --dry-run` prints training, evaluation and (in history mode) refresh
commands. `--mode history` uses an immutable snapshot for one complete epoch,
then reloads its checkpoint to refresh all nodes before the next process.
Model and optimizer resume together. `campaign.json` includes process startup,
checkpoint, host refresh and validation time. No live online Worker feedback
or arbitrary MemoryX history-table access is assumed. Native CUDA modes
`snapshot32` and `online1` are not implemented as live CSX feedback.

Local CPU SDK execution and CUDA results do not establish CSX compilation,
physical numerical accuracy or speedup. Those remain the operator's real-data
hardware measurements.

## uv entrypoint

`benchmark_scripts/research/run.sh` sets the fork's import paths and calls
`uv run --no-sync --project FORK_ROOT`. For example:

```bash
benchmark_scripts/research/run.sh history_embedding.pilot --help
```

The driver also starts train/refresh/evaluation subprocesses through the same
uv project. Existing `pyproject.toml` and `uv.lock` own dependencies; provision
the site environment with the repository's uv setup before running jobs.
`--no-sync` prevents a job from changing the installed SDK or downloading
dependencies. Remote Workers still need the same site-provisioned SDK.
