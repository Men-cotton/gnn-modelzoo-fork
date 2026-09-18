# Compiler artifacts and SDK boundaries

Use this reference to inspect saved compiler output or reconsider a current SDK workaround. `W/` means `../Wafer-GNN-manager/` from the repository root; `A/` means `W/artifacts/csctl_log_exports/`. Archive paths and the installed SDK behavior below were checked on 2026-09-18.

## Finding and interpreting artifacts

The former `log/20251101`, `log/20251102` and `log/20251106` exports now live under the corresponding dates in `A/`. These directories can be ignored by ordinary searches; use `rg --files --hidden --no-ignore <archive>` when locating them. A missing old path alone does not establish missing evidence.

Join records by job ID, compile ID and session before comparing them:

| Artifact | What to inspect |
| --- | --- |
| Job YAML, cluster details, host launcher log | Job role, execution mode, configuration and termination reason. A successful coordinator compile alone establishes compilation. |
| Top-level `cs_*` and role-specific `compiles/cs_*` | Compiler output; multiple locations can contain the same compile. |
| `cirh*.mlir` | Tensor operations, mask/count arithmetic, dtype conversions and `loc(...)` source sites. Read graph attributes such as `cs.batch_size` and `cs.num_csx` separately from later subbatch decisions. |
| `ws_rt.mlir` | Layouts, buffer allocation/release/finalization, host rx/tx/compute threads, send/recv, events and weight barriers. `io_bits` describes payload; cycle fields model execution. |
| `ws_km.mlir`, `kernel_annotation.json`, `kernel_graph.json`, `kernel_tree.json`, `ws_stack_llvm_stats.json` | Kernel structure and source mapping. Optimizer state versus accuracy accumulators helps identify train/eval programs. |
| `ws_opt_perf_summary.json`, context summaries/breakdowns/switch matrices | Compiler estimates of throughput, bubbles, context switches and memory slots. Retain the field's unit and scope. |
| `wio_flows.json`, `wio_report.txt`, floorplans, `cluster_cfg.json` | Static placement, flow allocation and topology. Establish runtime bottlenecks with corresponding measurements. |
| `<role>/sessions/00000*/` | `stream_stats.json`, `dbg_kid_to_command_*.json` and `.compile_artifact_location.out` connect session observations to commands and compile output. A queue snapshot requires its observation time. |

Mixed `f16`/`f32` IR requires dtype-aware byte accounting; multiplying every SRAM slot by four is unverified. Power/temperature, packet/flit latency, achieved memory bandwidth and arithmetic-unit utilization require their own measured counters. The cited IR and compiler summaries do not supply those measurements.

Further interpretation is maintained in the sibling [figure guide](../../../Wafer-GNN-manager/analyze/figures/README.md), [kernel-lowering reference](../../../Wafer-GNN-manager/context/fact/graphsage-ws-km-lowering-structure.md), [compiler-estimate reference](../../../Wafer-GNN-manager/context/fact/compiler-estimate-vs-measured-rate.md), [WIO guide](../../../Wafer-GNN-manager/analyze/scripts/WIO/README.md) and [evidence policy](../../../Wafer-GNN-manager/context/policy/01_writing/03_evidence_and_figures.md). The local [compilation overview](../../src/cerebras/modelzoo/models/gnn/docs/compile_pipeline.md) explains lowering, but its historical Python/package paths need adaptation to the active environment.

## Archived examples that prevent misleading comparisons

Each bundle below is `A/<archive-date>/log-export-wsjob-<job>-1e523ff4/`. Archive dates can differ from execution dates: `A/20251102/a2.txt:1–2,5–7,16–17,37–39` records a November 1 launch from a v2.6 checkout, GraphSAGE/ogbn-arxiv train-and-evaluate, random initialization, 2000 planned steps and reuse of the training compile cache. The three representative training/evaluation/November 6 `cirh-with-mask-ranges.mlir:2` files all declare `cs.batch_size=1024` and `cs.num_csx=1`.

| Archive date / job / compile | Interpretation and evidence |
| --- | --- |
| 20251101 / `gfeghe672j7visbaqfhdky` / `cs_5508846183343714573` | Training: optimizer state in `ws_km.mlir:22–25`; named dimension/global batch 1024 in `ws_rt.mlir:275–278`; memset constant `9.765625E-4` at line 477. |
| 20251101 / `it6afsqz4vdwxywthxx3pk` / `cs_12829142514842419922` | Evaluation: accuracy accumulators in `ws_km.mlir:30–31`, validation source site in `ws_rt.mlir:188`, named dimension 512 with global batch still 1024 at lines 189–192, memset 1.0 at line 398. |
| 20251102 / `ppcsgtrju8lbxdagl5n6sx` / `cs_5469441283859545208` and `bcz2lnpdz2knifw93rfdwd` / `cs_9246664174225285148` | Train/eval summaries repeat the preceding pair's values: estimated 264057.1766 / 344810.2466 samples/s, modeled context-switch 16.2629% / 50.1950%, subbatch 1024 / 512 (`ws_opt_perf_summary.json:16,29,33,52`). |
| 20251106 / `jxy2dzzgicat7flgf9c3an` / `cs_16392203599647071561` | Estimated 262564.3719 samples/s and modeled context-switch 19.1111%, global batch/subbatch 1024 (`ws_opt_perf_summary.json:16,29,33,52`). The job YAML records a successful coordinator job. |

Keep the training and evaluation programs as separate evidence. Their estimated rates, layouts and optimizer work differ; the larger estimate does not establish a faster implementation of the same training workload.

The following static values belong to those compile directories. The first two columns apply to the corresponding November 1/2 pairs; the last applies to `cs_16392203599647071561`. Preserve the compiler's field definitions when reusing them.

| Field / source | Training pair | Evaluation pair | November 6 |
| --- | ---: | ---: | ---: |
| `coreHeight × coreWidth`; `ws_opt_perf_summary.json:30–35` | 800×704 | 800×704 | 820×725 |
| `nActv / nShards`; same source | 21 / 21 | 21 / 21 | 21 / 15 |
| `act_rx_bubble_percent`; same file:6 | 38.4185 | 39.4894 | 29.4868 |
| `num_context_switches`; same file:37 | 12 | 26 | 13 |
| `num_act_wio_reconfigs / num_wgt_wio_reconfigs`; same file:36,43 | 0 / 2 | 0 / 0 | 0 / 2 |
| `recompute_cycles / recompute_percent`; same file:48–49 | 0 / 0 | 0 / 0 | 0 / 0 |
| Maximum `Active Memory Bound`; `ws_opt_context_summary.txt:6,18` | 10038, context 7 | 9216, context 1 | 9758, context 7 |
| WGT / GRD buffer columns, both edges combined; `wio_report.txt:30–35` | 6 / 8 | 10 / 4 | 4 / 8 |
| ACT / WGT WIO counts; same file:49,57 | 101 / 21 | 101 / 21 | 61 / 29 |
| Allocated WIOs / available; same file:70–74 | 123 / 124 | 123 / 124 | 91 / 124 |
| Left / right edge allocation; same source | 62/62 / 61/62 | 62/62 / 61/62 | 46/62 / 45/62 |

`Active Memory Bound` units have not been established as bytes. WIO allocation counts describe placement, not utilization over time. The November 6 geometry also corrects the former claim that every export used 800×704.

`A/20251102/a2.txt:155–157` records a secondary-rack configuration warning that 83 activation WIOs could be saturated versus 101 allocated, immediately after selecting subbatch 512. Preserve this as a configuration warning; the ordering alone establishes neither causation by the subbatch change nor measured bandwidth. Old SciPy/NumPy, OpProfiler serialization and kernel-module warnings belong to that historical environment, not the current SDK diagnosis. In particular, the November 1 `cc4ypvcbgcpwtgaydam5wz` bundle's `activation-0/dbg_act_0.out:5–7,17` records module/device lookup errors followed by successful server startup, so those errors alone do not establish job failure.

In `wio_flows.json`, `wio_flows_config.wio_flows` contains 246 records for each November 1 train/eval compile: 123 INGRESS and 123 EGRESS. Each direction has 62 records at `x_coord=0` and 61 at `x_coord=761`, with `txrx_mode='7:1'`. November 6 has 182 records: 91 per direction, distributed 46/45 across those x coordinates. The former description of 246 ingress flows counted both directions. The old Rxact/Txact=9:1 claim has no established counting definition and should not be reused.

Queue examples in `A/20251101/log-export-wsjob-cc4ypvcbgcpwtgaydam5wz-1e523ff4/<role>/sessions/<session>/stream_stats.json` preserve the raw fields below. Every listed `total_capacity` is 67108864. Their units, sampling times and ring wrap semantics need verification before converting differences to MiB occupancy or attributing a drain/idle phase.

| Role / session / domain / source lines | `head` | `tail` |
| --- | ---: | ---: |
| activation-1 / 000001 / 3 / 53–56 | 69696 | 67177152 |
| activation-1 / 000002 / 0 / 14–17 | 3088064 | 3136448 |
| activation-1 / 000002 / 1 / 27–30 | 3006208 | 3099840 |
| activation-1 / 000002 / 3 / 53–56 | 3068992 | 3099840 |
| activation-14 / 000002 / 3 / 53–56 | 81856 | 2668672 |

Three runtime/export checks remain useful:

- **Sentinel:** `A/20251101/log-export-wsjob-cc4ypvcbgcpwtgaydam5wz-1e523ff4/activation-1/sessions/000002/stream_stats.json:88,109` records `bytes_fetched=3735928559` (`0xDEADBEEF`). Treat it as a suspected sentinel pending its schema definition; exclude it from measured-byte sums.
- **Profiler collection:** `A/20251102/log-export-wsjob-yg9ufjrh6nuxkwpv6yxnrz-1e523ff4/activation-3/dbg_act_3.err:10` names `/n1/wsjob/workdir/job-operator/wsjob-yg9ufjrh6nuxkwpv6yxnrz/activation-3/opprofiler_flow.pb`. No `*opprofiler*` or `*.pb` file is present in that date's archive. Confirm the binary was collected before claiming trace coverage; configuration and output-path messages establish only the request.
- **Failure phase:** `A/20251106/log-export-wsjob-gfuh5yxfiigtpv9pbfqey3-1e523ff4/coordinator-0/dbg_crd_0.out:142–152` tries microbatch 128 during compile layout exploration and fails `!ioConfigs.empty()` in `OptimizeLayoutPass::initValidIOConfigurations` (`OptimizeLayout.cc:2235`). In the same date's `jyxfy2sl28ur9i5prwyx5e` bundle, coordinator output records an executor configured for 2500 iterations, iteration 2499, session close and `WS_RT_USER_TERMINATED`/SIGTERM; its job YAML:1851–1859 records cancellation by the client with the compile assertion as the reason. These records establish a compile failure and related cancellation. The former runtime-race/teardown explanation and count of two independent runtime assertions were unsupported.

## SparseMatMul support boundary

`TopKExpertsSparseLinear` in `src/cerebras/modelzoo/layers/SparseMoEBlock.py` uses a small expert-selection axis with weights `(hidden_out, num_experts, hidden_in)`. Mapping full-graph adjacency to that axis makes its extent the node count. The installed SDK's non-CSX implementation in `cerebras/pytorch/nn/functional.py:184–205` scatters to dense storage, performs `einsum("...MBN, KBN -> ...MBK", ...)`, then gathers selected slots.

The PubMed mapping has 19717 nodes and 172 fixed row slots. The findings at Git revision `f6cda4a:agent/findings.md` recorded a CPU allocation failure of 49,761,291,392 bytes; its CPU raw log and exact SDK version have not been recovered, so retain that number as a historical report. The CSX failure is independently preserved in `W/artifacts/model_dirs/PubMed_gcn_sparse_matmul/cerebras_logs/20260601_135814/run.log:49`: WAF conversion rejects

    (tensor<19717x172x1xf16>, tensor<19717x172xi64>, tensor<64x19717x1xf16>)
      -> tensor<19717x172x64xf16>

The current registry rejects this GCN architecture. Re-enabling it requires a supported graph-adjacency representation plus CPU and CSX validation for the intended shape; availability of the expert-selection operator alone is insufficient.

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

Historical full-graph evidence is in `W/artifacts/model_dirs/PubMed_gcn/cerebras_logs/20260515_065139/`: `executors/000001/cirh.mlir:46–47,89–98` divides masked loss and gradient by 19717; `run.log:13,91,105,145–146` records 60 supervised training nodes, step 1 loss 0.00335, 500 validation nodes and loss 0.02760. Their magnitudes are near `ln(3)*60/19717` and `ln(3)*500/19717`. This supports a normalization concern at every masked full-graph step; it is historical evidence rather than an exact logits/weights replay. The [paper reference](../research/hpcasia-revalidation.md) identifies the six GraphSAGE runs and the separate physical revalidation still required.

## Capturing fresh diagnostics

Locate tools from the active `cerebras.pytorch` installation; old Python 3.8 and `pip-packages/` examples are historical. In SDK 2.10, `backend/ltc_backend.py:122–128` reads `CSTORCH_DEBUG` (default 1) and passes it to `initialize(ir_debug=...)`. The installed PyTorch `torch/csrc/lazy/core/debug_util.h:34–36` documents `LTC_SAVE_TENSORS_FILE` as the destination when `SaveTensorsGraphInfo` is invoked. Set it to a run-specific path and verify the resulting file. The previous `LTC_IR_DEBUG`/`LTC_SAVE_TENSORS_FMT=dot` recipe and claim to force an executor performance trace have not been verified against this SDK. IR capture, profiler enablement and collection of a usable runtime trace each need their own artifact.
