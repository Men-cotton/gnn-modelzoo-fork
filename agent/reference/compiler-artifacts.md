# Compiler artifacts and SDK boundaries

Use this reference to inspect output collected by a current execution or to check an SDK workaround. Join compiler output, runtime records and the resolved configuration by execution job ID, compile ID and session before comparing them.

## Finding and interpreting artifacts

| Artifact | What to inspect |
| --- | --- |
| Job YAML, cluster details, host launcher log | Job role, execution mode, configuration and termination reason. Successful compilation establishes compilation only. |
| `cirh*.mlir` | Mask/count arithmetic, dtypes, source locations, global batch size and CSX count. Check the actual training and evaluation programs separately. |
| `ws_rt.mlir` | Layouts, buffer lifetimes, send/receive operations and synchronization. Payload and modeled cycle fields have different meanings. |
| `ws_km.mlir`, kernel annotations and graphs | Kernel structure, optimizer state, evaluation accumulators and source mapping. |
| `ws_opt_perf_summary.json` and context summaries | Compiler estimates of rate, bubbles, switches and memory slots. Preserve each field's units and scope. |
| `wio_flows.json`, `wio_report.txt`, floorplans, `cluster_cfg.json` | Static placement, ingress/egress flow allocation and topology. Allocation counts do not measure utilization. |
| Session `stream_stats.json` and command maps | Runtime counters, observation time, session identity and the associated compile. Verify units, wraparound and sentinels before subtraction. |

Mixed f16/f32 IR requires dtype-aware byte accounting. Power, temperature, achieved bandwidth, latency and arithmetic-unit utilization require measured counters. Compiler estimates and layout allocation cannot supply them. A profiler request or output-path message does not establish that a usable trace was collected.

Use the active environment's SDK tools and the [compilation overview](../../src/cerebras/modelzoo/models/gnn/docs/compile_pipeline.md). Keep raw compiler output and the exact SDK/tool identity together. A local lowering fixture does not establish physical-device numerical behavior or full-model performance.

## SparseMatMul support boundary

`TopKExpertsSparseLinear` in `src/cerebras/modelzoo/layers/SparseMoEBlock.py` uses an expert-selection axis. A graph adjacency matrix does not automatically satisfy that operator's shape contract. The GNN registry rejects GCN SparseMatMul; enabling it requires a supported graph representation and CPU and CSX validation for the intended shapes. Check the active SDK's non-CSX implementation before making memory-scaling claims.

## SDK mean-loss conversion counterexamples

These fixtures retain the reason for the explicit-count contract in [pipeline-contracts.md](pipeline-contracts.md). They were converted locally with `cerebras_pytorch=2.10.0`, `cerebras_appliance=2.10.0`, `torch=2.4.0+cu121`; `torch-cirh-opt` SHA-256 was `84c356e9690aa27af3d9b7ba8ea5d8e2c43e4f6d7c207e72bf13b7ceb71120df`.

- [sdk-mean-loss.mlir](fixtures/sdk-mean-loss.mlir) keeps 4×2 logits and 4 labels dynamic. Mean NLL with `ignore_index=-100` lowers forward and backward to division by fixed 4.
- [sdk-mean-total-weight.mlir](fixtures/sdk-mean-total-weight.mlir) exposes `total_weight`: the converted value is the sum of two constant ones, independent of labels. This is a separate mismatch from interpretation of the sparse cross-entropy operation.
- [masked-loss.mlir](fixtures/masked-loss.mlir) is the equivalent explicit-mask workaround; its loss and gradient preserve the runtime valid-count denominator. It is authored arithmetic, not the complete Python model export.

From the repository root, convert each fixture with the active SDK's tool; for example:

    .venv/lib/python3.11/site-packages/cerebras/pytorch/lib/torch-cirh-opt \
      --torch-to-cirh-pipeline agent/reference/fixtures/sdk-mean-total-weight.mlir -o -

The host oracle uses zero logits of shape `(4,2)` and labels `[0,-100,-100,-100]`:

```python
import torch
logits = torch.zeros(4, 2, requires_grad=True)
labels = torch.tensor([0, -100, -100, -100])
loss, total = torch.ops.aten.nll_loss_forward(
    logits.log_softmax(1), labels, None, 1, -100
)
loss.backward()
print(loss.item(), total.item(), logits.grad.tolist())
```

Expected PyTorch results are loss ≈ 0.69314718, total_weight 1 and gradient `[[-0.5,0.5],[0,0],[0,0],[0,0]]`. Complete generated CIRH has not been numerically replayed here. Its internal negative-label handling remains unproved by these fixtures; a claimed numerical loss/gradient scaling factor therefore requires that assumption or a numerical execution. Manually splitting a PyTorch batch and observing NaNs is not evidence of this compiler behavior.

## Capturing diagnostics

Locate tools from the active `cerebras.pytorch` installation. In SDK 2.10, `backend/ltc_backend.py` reads `CSTORCH_DEBUG` and passes it to `initialize(ir_debug=...)`. PyTorch's `torch/csrc/lazy/core/debug_util.h` documents `LTC_SAVE_TENSORS_FILE` as the destination when `SaveTensorsGraphInfo` is invoked. Set a run-specific path and verify the resulting file. IR capture, profiler enablement and collection of a usable runtime trace each require their own artifact.
