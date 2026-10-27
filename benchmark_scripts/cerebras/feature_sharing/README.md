# Shared tree features on ALCF

The dense and dictionary cases use the same native GraphSAGE, masked loss,
initial weights and first, second and tail sampled batches. Dictionary mode
reconstructs the tree with `torch.gather`. Capacity covers only this frozen
window; this is not online sampling or an epoch throughput experiment.

Run from the existing shared checkout with its provisioned `.venv` and SDK
site authentication. The management endpoint is discovered by the SDK, as in
ordinary GNN jobs. One CSX and one input streamer use the existing
`configs/autotune/arxiv_w40.yaml` backend settings. There is no cluster JSON.
The common source-run code binds Worker imports and shared mounts.

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
WORK="$PWD/model_dirs/shared-tree-01"
MODULE=cerebras.modelzoo.models.gnn.tools.feature_sharing
mkdir -p "$WORK"
# Use the same datasets root as the ordinary GNN jobs.
DATA="$PWD/src/cerebras/modelzoo/models/gnn/data/datasets"
uv run --no-sync -- python -m "$MODULE.prepare" --dataset arxiv --data-dir "$DATA" \
  --batch-size 128 --fanouts 15 10 5 --hidden 128 --dropout 0 --output "$WORK/input"
uv run --no-sync -- python -m "$MODULE.preflight" \
  --input "$WORK/input" --output "$WORK/preflight.json"
```

Preparation reads the existing GNN data directory; `--data-dir` can select
another existing shared dataset. It uses the native sampler directly.
A new window or batch size needs new inputs and compilation. Overflow is an
error. Paired comparisons require dropout zero.

Launch through the existing catalog; each invocation pins the current HEAD.
Use the same committed source for preparation and all paired phases. The
source/manifest and output directories must be shared, never client `/tmp`.

```bash
uv run --no-sync -- python -m cerebras.modelzoo.tools.benchmark_launcher jobs \
  --job csx-shared-tree -- --phase compile --precision bf16 \
  --input "$WORK/input" --output "$WORK/compile-check"

uv run --no-sync -- python -m cerebras.modelzoo.tools.benchmark_launcher jobs \
  --job csx-shared-tree -- --phase verify --precision bf16 \
  --input "$WORK/input" --output "$WORK/verify"

uv run --no-sync -- python -m cerebras.modelzoo.tools.benchmark_launcher jobs \
  --job csx-shared-tree -- --phase measure --precision bf16 \
  --input "$WORK/input" --output "$WORK/measure" --steps 30 --warmup-steps 3 \
  --verification "$WORK/verify/pilot.json"
```

Add `--dry-run` to preview the pinned launch; `--foreground` disables tmux.
Compilation files default to each experiment's `compile/`; `--compile-dir`
can reuse a shared cache. Different capture graphs may compile separately.
The jobs run sequentially. Inspect `<output>-launcher/launcher.json`,
`driver.log`, `<output>/pilot.json`, and each case's logs and `result.json`.
A detached launch is not completion. Compilation records
`compiled_not_executed`; numerical comparison records `passed` only after
checking losses, logits, final model and optimizer states. BF16 comparison
uses rtol/atol 0.01; FP32 defaults to exact comparison.

Timing counts valid targets after warmup at client activation receipts,
including capture overhead. It is not device time, network bandwidth or
end-to-end sampler throughput. Fewer input bytes do not establish a CSX
speedup or SRAM reduction. The compiler may materialize the expanded tree.

For a local SDK CPU check, call the same pilot module directly with
`--backend CPU --phase verify --precision fp32`. Preflight and CPU execution
do not establish CSX compilation or hardware correctness.
