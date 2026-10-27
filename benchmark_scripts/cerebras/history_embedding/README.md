# History and exact feature means on ALCF

Use the existing shared checkout, provisioned `.venv`, SDK site authentication
and GNN processed datasets. Runtime code is under
`cerebras.modelzoo.models.gnn.tools.history_embedding`. It reuses native
GraphSAGE layers and masked classification loss. The experimental logic is
the fixed sampled graph, raw-feature preaggregation and immutable histories.
See [the method description](method.md) for the approximation and comparisons.

Complete the repository's `./setup.sh` dataset preparation before these
commands. It uses the existing GNN dataset downloader and PyG processing;
`prepare` builds the experiment inputs through the existing `BaseGraphDataSource`.
If raw data or splits are missing, it stops and prints the existing
`download_datasets` command with the selected data directory and dataset.
If ordinary GNN jobs use another `data_dir`, set `DATA` to that same datasets
root. The expected arxiv graph is
`$DATA/ogbn-arxiv/ogbn_arxiv/processed/geometric_data_processed.pt`.

```bash
export PYTHONPATH="$PWD/src"
export PATH="$PWD/.venv/bin:$PATH"
WORK="$PWD/model_dirs/history-arxiv-01"
MODULE=cerebras.modelzoo.models.gnn.tools.history_embedding
mkdir -p "$WORK"
# Use the same datasets root as the ordinary GNN jobs.
DATA="$PWD/src/cerebras/modelzoo/models/gnn/data/datasets"
uv run --no-sync -- python -m "$MODULE.hardware" prepare --dataset ogbn-arxiv --data-dir "$DATA" \
  --hidden 128 --layers 3 --fanout 10 --dropout 0 --output "$WORK/input"
uv run --no-sync -- python -m "$MODULE.preflight" \
  --input "$WORK/input" --batch-size 256 --output "$WORK/preflight.json"
```

Preparation reads the existing `models/gnn/data/datasets` directory by default;
`--data-dir` selects another existing dataset. It does not download or subset
the graph. The bundle contains original adjacency, official splits, fixed
sampled neighborhoods and initial model/history tensors. B256 logical roots
form one SDK sample, so SDK sample/s is not node/s.

The catalog uses existing source pinning and tmux. Each launch uses committed
HEAD; preparation and execution must use the same implementation. Source runs,
input, output and compile files must be on the ALCF shared filesystem. Worker
imports need only the pinned `src`, set by the existing source-run helper.
The backend reuses `configs/autotune/arxiv_w40.yaml`: one CSX, one streamer,
7200-second job limit, and SDK endpoint discovery. No cluster JSON is needed.

```bash
uv run --no-sync -- python -m cerebras.modelzoo.tools.benchmark_launcher jobs \
  --job csx-history-pilot -- --phase execute \
  --input "$WORK/input" --output "$WORK/window" \
  --batch-size 256 --steps 128 --warmup-steps 20 --repeats 3
```

The default pair is `tree` and `tree_mean`. Add
`--cases tree tree_mean history history-export` for history reads and returned
activations. `--phase compile` compiles without execution; execution can also
compile directly. `--dry-run` previews without creating or submitting work;
`--foreground` disables tmux. Compile files live under the experiment output.
Read `<output>-launcher/launcher.json`, `driver.log`, `pilot.json` and each
case's `result.json`. A detached launch is not job completion.

The 128-step window is not a complete epoch. Activation cadence is one for
all pilot cases. Client timing includes activation delivery and capture I/O;
whole-process timing includes startup and checkpoints. Neither isolates WSE
time. Returned histories are saved on the client and do not mutate Workers.

Numerical replay uses the same real input window and SDK optimizer:

```bash
uv run --no-sync -- python -m "$MODULE.verify_sdk" --input "$WORK/input" \
  --epoch0 "$WORK/window/tree_mean-r0" --atol 0.0002 --rtol 0.0001 \
  --output "$WORK/tree-mean-numerics.json"
uv run --no-sync -- python -m "$MODULE.evaluate" --bundle "$WORK/input/bundle.pt" \
  --checkpoint "$WORK/window/tree_mean-r0/checkpoint.mdl" \
  --output "$WORK/window-validation.json"
```

Evaluation uses the original full graph and current weights. It defaults to
validation; select the test checkpoint using validation first. Dropout-zero
replay is a numerical check, not a learning comparison.

For learning, prepare a fresh input with `--dropout 0.5`, then launch:

```bash
uv run --no-sync -- python -m cerebras.modelzoo.tools.benchmark_launcher jobs \
  --job csx-history -- --mode tree_mean --epochs 10 --batch-size 256 \
  --input "$WORK/learning-input" --output "$WORK/learning"
```

Each epoch covers all training roots and validates on the full graph. In
`--mode history`, a host process refreshes all histories from the final model
checkpoint before the next epoch. Model and optimizer resume together.
`campaign.json` includes startup, checkpoint, refresh and validation time.
There is no live Worker feedback or MemoryX history-table assumption. Native
CUDA `online1`/`snapshot32` modes remain local experiments.

Local SDK checks use the same modules with `--backend CPU`. CPU, CUDA and
`torch.export` results do not establish CSX compilation, numerical accuracy
or performance; those require completed ALCF runs.
