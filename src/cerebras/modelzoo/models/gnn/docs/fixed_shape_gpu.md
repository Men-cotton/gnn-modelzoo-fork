# Fixed-shape GraphSAGE on GPU

`fixed_shape_gpu.py` runs the ModelZoo GraphSAGE representation with native
PyTorch on a single GPU. It reuses `NeighborSamplingDataProcessor`,
`GraphSAGENeighborSamplerDataset`, `GraphSAGEBatch`, `GNNModel`, and its masked
loss. The existing PyG runner remains a separate benchmark path.

The CPU prepares the same deterministic sampled trees, feature tensors,
`node_masks`, `neighbor_masks`, labels, and `target_mask` as the CSX path.
The GPU receives complete batches, including padded targets in the last batch.
For batch size B, feature width F, and fanouts f1, f2, ...,
feature shapes are `[B, 1, F]`, `[B, f1, F]`, `[B, f1*f2, F]`, etc.
Aggregation and loss exclude masked entries. `cache_fraction` uses the shared
GraphCache on **CPU**, including when DataLoader workers are enabled.
`static_batch_cache_size` repeats precomputed batches just as in ModelZoo;
use it only for input-pipeline probes, not accuracy measurements.

## Run directly

Use the repository environment (`./setup.sh --target-env gpu` on Pegasus).
The Cerebras Python package is still an import dependency, but this runner does
not create a Cerebras backend or use its Trainer/executor.
From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.models.gnn.fixed_shape_gpu \
  --config src/cerebras/modelzoo/models/gnn/configs/params_graphsage_ogbn_arxiv_throughput_nocache.yaml \
  --output-dir model_dirs/fixed_shape_gpu/arxiv_run1
```

An existing output directory must be empty. The ModelZoo config's `model_dir`
is not used. `--num-workers N` overrides train and validation worker counts;
otherwise the shared config and CPU-affinity validation apply.
For a short check use `--max-steps 5 --warmup-steps 1`. This preserves the
configured tensor sizes; choose a smaller config explicitly if GPU memory is
insufficient. There is no automatic change of batch size or fanouts.

Execution is eager by default; `--compile` enables `torch.compile`.
`trainer.init.precision` selects FP32/FP16/BF16; `--precision fp32|fp16|bf16`
overrides it. FP16 uses dynamic gradient scaling. CPU correctness checks require
`--device cpu --precision fp32`; production runs fail if CUDA is unavailable.

This runner supports GraphSAGE, AdamW, `max_steps`, integer step-based
`eval_frequency`, and full validation passes when `compute_eval_metrics` is
true. Training cycles the sampler when exhausted. `steps_per_epoch` does not
truncate the sampler. The sampler's fixed ordering and neighborhoods are reused
on each pass, matching its current deterministic behavior.
It does not implement Trainer callbacks, checkpoint resume, scheduler policies,
gradient accumulation, DDP, or epoch-based stopping. It saves a final checkpoint.
Validation counts only real targets, including the final partial batch.

## Submit on Pegasus (NQSV)

```bash
benchmark_scripts/pegasus/submit_fixed_shape_gpu_nqsv.sh \
  --config src/cerebras/modelzoo/models/gnn/configs/params_graphsage_ogbn_arxiv_throughput_nocache.yaml \
  --dry-run
```

Remove `--dry-run` to submit one GPU job. Add `--compile` to compile the model.
The submitter sets the repository root as the submission directory and passes
an absolute config path. The PBS script uses the existing AC2/gpu queue and
requests one node for two hours. Each job writes a separate directory beneath
`model_dirs/fixed_shape_gpu/`, with timestamp and job ID in its name.
For interactive Pegasus runs, `run_fixed_shape_gpu.sh` accepts the Python
runner options and also supports `--dry-run`.

## Results and measurement boundary

Each output directory contains:

- `resolved_config.yaml`: merged YAML with CLI step/worker overrides.
- `metrics.jsonl`: runtime precision, device, cache placement, train windows,
  validation accuracy, and final measured totals. The same records go to stdout.
- `checkpoint.pt`: uncompiled model state, AdamW state, scaler state and step.

Warmup defaults to 40 steps. Windows synchronize CUDA at their boundaries and
include waiting for the host loader, batch transfer, forward, backward, and
optimizer work. Setup, warmup, validation, and checkpoint writing are excluded
from summary throughput. Logging occurs between measured windows. Workers may
prefetch across boundaries; this measures a running input pipeline, not isolated
sampling latency. Compilation after warmup, if any, remains part of the timing.

`seed_nodes_per_second` counts `target_mask` entries that are true;
`nominal_slots_per_second` includes padding. Keep the metric names when comparing
with PyG or CSX. Representation reuse does not imply identical kernels,
precision behavior, training schedules, or throughput across devices.

## Validation

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover \
  -s src/cerebras/modelzoo/models/gnn/tests -p test_fixed_shape_gpu.py
```

Tests use a tiny graph with an isolated node and a padded tail. They check
ModelZoo/native payload equality with zero and one worker, host caching,
padding-independent loss/gradients, train/eval counts, and checkpoint updates.
When CUDA is available they also check CPU/CUDA logits, loss and gradient parity,
and FP32/FP16 GPU training. These checks do not measure OGB or Pegasus performance.
