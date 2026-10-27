# Shared tree features on ALCF

Run from the repository checkout after the usual `./setup.sh` dataset preparation:

```bash
benchmark_scripts/cerebras/run_shared_tree.sh
```

With no arguments, use arxiv and a new UTC-timestamped directory under the
pinned checkout's `model_dirs/shared_tree/` (bound to this run's outputs).
`--output` explicitly selects a different location; relative paths are resolved
from the caller's directory. The launcher reports its log/output location.
`--dataset` accepts both `arxiv`/`products` and their `ogbn-` names.

This uses the ordinary catalog, pinned source and detached tmux launcher.
`--foreground` runs in the terminal; `--dry-run` previews the commands. Each
launch uses committed HEAD, including input preparation. The checkout, data,
output and compile cache must be on the existing ALCF shared filesystem.
The provisioned `.venv` and SDK authentication are used by the shell wrapper.

One invocation prepares the frozen inputs, runs dense and dictionary jobs
sequentially, and records their results in `campaign.json`. The SDK compiles
as part of execution. No separate compile/verify/measure submissions or
verification receipt are required. Logs are in `prepare.log`, `dense.log`,
`dictionary.log`, and `<output>-launcher/driver.log`.

The defaults come from `configs/autotune/{arxiv,products}_w40.yaml`:

- H1024, three layers, batch 4096, dropout 0.5, fanouts 15/10/5, seed 42.
- CSX float16 with dynamic loss scaling, including scaler checkpoint state.
- The native AdamW configuration and parameter groups: lr 0.003, eps 1e-6,
  betas 0.9/0.999, weight decay 0.0005, and no decay on biases.
- 240 steps, with the first 40 excluded from the reported client rate.
- One CSX/input streamer, 40 DataLoader workers, prefetch 2, no persistent
  workers, pin memory enabled, and the existing 7200-second job limit.

Both representations use the same native GraphSAGE and masked loss, initial
weights, input window and settings. Dictionary mode reconstructs exactly the
same features with `torch.gather`. Preparation uses `BaseGraphDataSource` and
the native sampler. Missing data prints the existing `download_datasets`
command; this experiment does not download data.

`--input <directory>` reuses a previously prepared window and its stored model,
batch size, seed and fanouts. Preparation options (`--dataset`, `--data-dir`,
`--batch-size`, `--fanouts`, `--hidden`, `--dropout`, `--seed`, `--capacity`)
cannot be combined with `--input`; prepare a new window to change them.
`--data-dir` selects the same datasets root used by ordinary GNN jobs.
`--batch-size`, `--hidden`, `--fanouts`, `--dropout`, `--seed`, `--steps` and
`--warmup-steps` are explicit overrides applied equally to both cases.

Dropout is 0.5 by default, as in the throughput preset. The same seed does not
establish identical compiler RNG behavior between the two graphs, so these
runs record `comparison.status=not_checked` and completion as `executed`.
For a deterministic numerical comparison, use `--dropout 0` when preparing a
new window. The same invocation then compares every captured loss/logit and
final model, optimizer and scaler state. FP16 tolerance defaults to 0.01;
CPU checks are exact. `--rtol` and `--atol` apply only to this dropout-zero
comparison and must be finite and nonnegative. This check is reported
separately from execution. `--warmup-steps` must be positive and less than
`--steps`; the client rate excludes receipts before that boundary.

The remaining experimental differences from ordinary training are deliberate:
only the first, second and tail sampled batches are frozen and repeated;
each complete logical tree is one SDK sample; and every step has a client
activation receipt. Timing includes those receipts and capture overhead.
It is not a full epoch, online-sampling throughput, isolated device time or
network bandwidth. Fewer logical input bytes alone do not establish a speedup.

For a small local SDK check with an already prepared dropout-zero window:

```bash
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.models.gnn.tools.feature_sharing.pilot \
  --backend CPU --input /tmp/shared-input --output /tmp/shared-check \
  --steps 3 --warmup-steps 1
```

CPU checks use FP32 and zero DataLoader workers. They do not validate CSX
compilation, FP16 numerics or hardware performance.
