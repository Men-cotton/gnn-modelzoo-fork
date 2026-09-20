# GPU comparison policy (2026-09-20)

The CS-3 and Pegasus experiments share nominal learning settings. They compare
different implementations, with the exceptions below recorded explicitly. The
GPU path retains native PyTorch/PyG algorithms; numerical equivalence is not a
claim of this policy. Existing measurements must not be mixed with new runs.

## Research used before implementation

Primary sources were checked before changing the GPU policy:

- [PyG 2.7 neighbor sampling](https://pytorch-geometric.readthedocs.io/en/2.7.0/tutorial/neighbor_loader.html):
  ordinary NeighborLoader minibatches share sampled nodes, use random sampling,
  and may shuffle seed nodes each epoch. Retain these behaviors; do not copy the
  CS-3 sampler's deterministic cyclic neighbor windows or repeated epoch order.
- [PyG 2.7 SAGEConv](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.SAGEConv.html):
  retain the native layer, mean aggregation, root transform and additive bias.
  Do not add the CS-3 model's second trainable bias per layer.
- [PyTorch parameter groups](https://docs.pytorch.org/docs/2.14/optim.html#per-parameter-options):
  bias exclusion from weight decay is a documented native configuration.
  Use it to match the regularized parameter categories without importing SDK
  optimizer construction. Normalization parameters are also excluded.
- [PyTorch AdamW](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html):
  keep its update equation and automatic CUDA foreach selection. Fused execution
  remains available through the existing AdamW config, without a custom optimizer.
  Setting equal epsilon values does not equate the SDK/PyTorch equations.
- [PyTorch AMP](https://docs.pytorch.org/docs/2.14/amp.html):
  initial scale and growth interval are native knobs. Match their nominal values
  using the standard GradScaler, while retaining native overflow adaptation and
  avoiding SDK-specific scale clamps.
- [PyG compiled GNN guidance](https://pytorch-geometric.readthedocs.io/en/stable/tutorial/compile.html):
  use `torch.compile(dynamic=True)` for varying minibatch shapes. Eager execution
  remains explicitly selectable for diagnosis or a separately labeled baseline.

The online documentation describes the supported APIs, not a measured optimum
for this workload. API compatibility and host behavior were checked with the
installed Torch 2.4.0 / PyG 2.7.0. No dependency upgrade or hardware job is required
by this change. GPU cache, sparse execution, pinned input and TF32 capability are
retained; CPU/CSX input constraints do not determine the GPU execution strategy.

## Matched settings and retained differences

| Item | Policy |
| --- | --- |
| Architecture configuration | Same width/depth, mean aggregation, ReLU/dropout placement and separate classifier; retain PyG's one bias per SAGEConv versus two in CS-3 |
| Optimizer | Native AdamW; eps 1e-6 by default, same configured lr/weight_decay/betas; bias/norm parameters have zero decay |
| AdamW arithmetic | PyTorch puts eps after second-moment bias correction; SDK puts it before. Retain and disclose this difference |
| Precision | Same float16 AMP setting; initial scale 32768, growth interval 2000 by default, configurable with native-supported knobs |
| AMP adaptation | Native PyTorch behavior; no SDK min/max scale clamps or off-by-one growth rule emulation |
| Input | Same dataset/splits/fanouts/seed batch size/seed numbers; ordinary PyG random sampling, shared nodes and per-epoch shuffle |
| Epoch tail | Include it; actual seed counts, not 4096 nominal slots for every batch |
| Learning loss | Current-step mean over supervised seeds, matching the CS-3 progress-log definition; retain the ten-step mean separately |
| Validation | Full valid split with sampled neighborhoods; native PyG accuracy. Sampling randomness differs; no fabricated validation loss |
| Cache | Explicit 0.0 disables GPU caching; 1.0 caches on GPU. CS-3 full cache resides on CPU |
| Compilation | Explicit launcher choice; dynamic shapes when enabled; record the effective choice, not a fixed-shape-only CLI flag |

The plain GPU runner rejects unsupported configured schedulers, clipping, FP16
static loss scaling and SDK scale limits rather than silently ignoring them.
BF16/FP32 do not enable gradient scaling. Explicit optimizer epsilon is preserved.
CSX resolved throughput/learning configs now spell out the same existing FP16
initial scale and growth interval defaults; the CSX model and sampler are unchanged.

## Evidence saved for each run

`params.yaml` stores generated hyperparameters. `[GPU policy]` in the PyG log
records native optimizer identity/defaults, parameter names by decay group,
parameter count and resolved AMP options. `[Compile]` records the effective
compile switch and dynamic-shape setting. The campaign/study environment includes
package versions, source fingerprint, `NO_COMPILE` and host thread environment.
The fixed-shape GPU control saves the same optimizer/AMP policy in `gpu_policy.json`.

Version 2 `[Autotune]` records include cumulative optimizer updates, skipped
updates and loss scale. Update counting handles both ordinary GradScaler-skipped
calls and fused optimizers that suppress the update inside the kernel. Counting
does not introduce a host synchronization on every step.

The throughput collector records warmup endpoints and rejects skips within the
measured window; warmup skips alone do not invalidate a later full-update window.
Legacy version 1 logs remain readable with `optimizer_update_check` explicitly
marked unavailable. Missing old counters never imply zero skips. Learning curves
retain the update/skip counts and full-precision accuracy without automatic
accuracy-based acceptance. Losses must remain finite.

Before combining results, compare policy records, effective compile mode, timing
window and input controls. Equal seed numbers do not promise equal initialization
or sampled subgraphs. Dataset file identity across machines still needs local
data provenance; names/version numbers alone do not prove identical files.

Use a fresh output directory for reruns. See [HPC Asia commands](hpcasia_campaign.md)
for the independent 30-run campaign and the 54-run worker study.
