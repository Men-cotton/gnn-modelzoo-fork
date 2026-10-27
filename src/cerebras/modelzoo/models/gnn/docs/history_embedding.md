# History and feature-mean comparisons on ALCF

After the usual `./setup.sh` dataset preparation, run from the checkout:

```bash
benchmark_scripts/cerebras/run_history_benchmark.sh \
  --dataset ogbn-arxiv --output model_dirs/history-window-01
```

The shell wrapper uses the existing `.venv`, catalog, source pinning, SDK
site authentication and tmux launcher. `--foreground` runs in the terminal;
`--dry-run` previews without submission. Preparation and runtime use the same
committed HEAD. All files must be on the existing ALCF shared filesystem.

One invocation prepares the graph bundle and initial histories, then executes
both `tree` and `tree_mean`. Compilation is part of SDK execution. Add
`--cases tree tree_mean history history-export` to include immutable history
reads and returned activations. `--repeats` repeats the matched cases.
Results are in `pilot.json` and each case's `result.json`; preparation and
case logs are stored beside them. Runtime modules remain under
`cerebras.modelzoo.models.gnn.tools.history_embedding`.

Defaults come from `configs/autotune/{arxiv,products}_w40.yaml`: H1024, three
layers, batch 4096, dropout 0.5, seed 42, 240 steps with 40 warmup steps, CSX
float16 with dynamic loss scaling, and native AdamW parameter groups and
settings (lr 0.003, eps 1e-6, betas 0.9/0.999, weight decay 0.0005 with no bias
decay). The backend and DataLoader worker settings also use that preset.
`--hidden`, `--layers`, `--batch-size`, `--dropout`, `--seed`, `--steps` and
`--warmup-steps` explicitly override these conditions for every compared case.

The same `BaseGraphDataSource` loads the full dataset, official splits and
native FP16-quantized features. Host aggregation and history refresh use FP32.
The experimental sampled graph uses the native root-neighbor sampling rule,
is frozen, and is shared by all cases. A single
context-free history per node requires the same sampled neighbors at every
depth, so its single fanout defaults to the native root fanout (15), rather
than the ordinary 15/10/5 depth-specific context. `--fanout` can override this
for all cases together. See [the method](history_method.md) for the approximation.

The benchmark can cross input passes to reach 240 steps. It reuses the native
`EpochBatchSampler` and seed+epoch target shuffle; tail roots remain padded and
masked. Histories stay immutable for the entire benchmark job. Exported root
IDs must be unique within each pass, and can recur in subsequent passes.
A checkpoint ending partway through a pass is not an epoch-resume checkpoint.

Each logical target batch is one SDK sample because local neighbor indexes
address that whole batch. Every step is captured to check root IDs and count
supervised targets. Timing includes client activation delivery and capture
I/O; whole-process time additionally includes compilation/startup/checkpoints.
Neither is isolated WSE time. These differences are shared by all cases.

For full-epoch learning, refresh and validation:

```bash
benchmark_scripts/cerebras/run_history_campaign.sh \
  --dataset ogbn-arxiv --mode tree_mean --epochs 10 \
  --output model_dirs/history-learning-01
```

`--mode history` refreshes all histories from the final checkpoint between
epochs. Model, optimizer and loss scaler resume together. Validation uses the
original full graph and current weights. The epoch campaign intentionally
uses full epochs, not the throughput benchmark's fixed 240-step window.

`--input <directory>` reuses a prepared bundle and its stored model/graph
conditions. Otherwise preparation is included in the launch. `--data-dir`
selects the ordinary datasets root. Missing data reports the existing
`download_datasets` command. Inputs are never downloaded implicitly.

Optional numerical diagnostics use a fresh `--dropout 0` input. `verify_sdk`
replays a captured run with the same SDK optimizer on CPU, including skipped
optimizer updates. Mixed-precision replay uses a tolerance, not bit equality.
For example, after a dropout-zero benchmark:

```bash
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.models.gnn.tools.history_embedding.verify_sdk \
  --input model_dirs/history-window-01/input \
  --epoch0 model_dirs/history-window-01/tree_mean-r0 \
  --output model_dirs/history-window-01/numerics.json
```

CPU development runs use `--backend CPU` on the Python drivers, FP32 and zero
DataLoader workers. Their results do not establish CSX correctness or speed.
