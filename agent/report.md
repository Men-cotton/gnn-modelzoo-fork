# Wafer-Scale GNN Log Review

## Executive Overview
- Nov 2 exports capture cached train compile `cs_5469441283859545208` at ≈264 k samples/s with 16.3 % context-switch overhead alongside the fresh eval compile `cs_9246664174225285148` at ≈345 k samples/s with 50.2 % switch cost, each spanning 21 shards of the 800×704 mesh (`log/20251102/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/cs_5469441283859545208/ws_opt_perf_summary.json:2-59`, `log/20251102/log-export-wsjob-bcz2lnpdz2knifw93rfdwd-1e523ff4/log-export-wsjob-bcz2lnpdz2knifw93rfdwd-1e523ff4/cs_9246664174225285148/ws_opt_perf_summary.json:2-60`).
- The eval pass redistributes WIO buffers from 3×/4× columns for weights/grads to 5×/2× while the compiler still sees 101 activation flows, triggering the scheduler's “Non Plan-of-Record Secondary Rack” bandwidth warning once gradient accumulation dropped to 512 (`log/20251102/log-export-wsjob-yg9ufjrh6nuxkwpv6yxnrz-1e523ff4/cs_5469441283859545208/wio_report.txt:10-76`, `log/20251102/log-export-wsjob-yg9ufjrh6nuxkwpv6yxnrz-1e523ff4/cs_9246664174225285148/wio_report.txt:10-76`, `log/20251102/a2.txt:155-157`).
- Enabling `op_profiler_config` propagated `profile_bins` flags to every replica and attempted to emit `opprofiler_flow.pb`, but the export retains only console traces while the launcher floods Pydantic union warnings about the `OpProfiler` payload (`log/20251102/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/wsjob-ppcsgtrju8lbxdagl5n6sx.yaml:65-76`, `log/20251102/log-export-wsjob-yg9ufjrh6nuxkwpv6yxnrz-1e523ff4/wsjob-yg9ufjrh6nuxkwpv6yxnrz.yaml:96-113`, `log/20251102/log-export-wsjob-yg9ufjrh6nuxkwpv6yxnrz-1e523ff4/activation-3/dbg_act_3.err:1-12`, `log/20251102/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/coordinator-0/dbg_crd_0.err:8050-8073`, `log/20251102/b2.txt:7-29`).

## 2025-11-06 Run Analysis
- **Performance:** Successful job (`log-export-wsjob-jxy2dzzgicat7flgf9c3an-1e523ff4`) achieved ≈262.5 k samples/s with 19.11 % context-switch overhead and 1024 batch size (`ws_opt_perf_summary.json:2-62`).
- **Runtime Failure:** Two jobs failed with `Internal assertion error: !ioConfigs.empty()`. The larger failed job ran for ≈2500 iterations, indicating a runtime race condition or teardown issue rather than a static configuration error, despite `wio_flows.json` being fully populated.
- **Telemetry Deep Dive:**
    - **Memory:** Peak usage in Context 7 at ≈9.7 k slots. Inferred as ≈39 KB (assuming 4-byte slots based on `ws_rt.mlir` `f32` allocations).
    - **NoC/IO:** High WIO utilization (91/124) with left-edge pressure.
    - **Core Utilization:** 9:1 Rxact/Txact ratio confirms ingress-heavy workload.


## Nov 2 Findings
- Context memory ceilings remain at ≈10 k slots for the cached training compile but fall to ≤9.2 k across contexts 0–7 in the eval compile, easing the single-context hotspot seen previously (`log/20251102/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/cs_5469441283859545208/ws_opt_context_summary.txt:1-18`, `log/20251102/log-export-wsjob-bcz2lnpdz2knifw93rfdwd-1e523ff4/log-export-wsjob-bcz2lnpdz2knifw93rfdwd-1e523ff4/cs_9246664174225285148/ws_opt_context_summary.txt:1-18`).
- The training bundle still performs two weight-side WIO reconfigs while the eval run removes them and halves gradient accumulation (1024→512), matching the host log’s micro-batch selection during the second compile (`log/20251102/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/cs_5469441283859545208/ws_opt_perf_summary.json:33-59`, `log/20251102/log-export-wsjob-bcz2lnpdz2knifw93rfdwd-1e523ff4/log-export-wsjob-bcz2lnpdz2knifw93rfdwd-1e523ff4/cs_9246664174225285148/ws_opt_perf_summary.json:33-60`, `log/20251102/a2.txt:155-162`).
- Activation-side logs confirm `profile_bins` hooks fired on-device, but no `.pb` artifact was bundled, so the trace must be collected from the remote path referenced in the runtime logs (`log/20251102/log-export-wsjob-yg9ufjrh6nuxkwpv6yxnrz-1e523ff4/activation-3/dbg_act_3.err:1-12`).

### 2025-11-02 Performance Snapshot

| compile_id | roles captured | samples/s | context switches | grad micro-batch | notes |
| --- | --- | --- | --- | --- | --- |
| `cs_5469441283859545208` | activation, weight, chief, worker | ≈264 k | 12 (16.3 % time) | 1024 | 2 weight-side WIO reconfigs; cached compile reused (`log/20251102/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/cs_5469441283859545208/ws_opt_perf_summary.json:36-59`, `log/20251102/a2.txt:37-39`) |
| `cs_9246664174225285148` | activation, weight, chief, worker | ≈345 k | 26 (50.2 % time) | 512 | 0 WIO reconfigs; weight buffers widen while grad buffers shrink (`log/20251102/log-export-wsjob-bcz2lnpdz2knifw93rfdwd-1e523ff4/log-export-wsjob-bcz2lnpdz2knifw93rfdwd-1e523ff4/cs_9246664174225285148/ws_opt_perf_summary.json:33-59`, `log/20251102/log-export-wsjob-yg9ufjrh6nuxkwpv6yxnrz-1e523ff4/cs_9246664174225285148/wio_report.txt:10-35`) |

### 2025-11-01 Performance Snapshot

| compile_id | roles captured | throughput (samples/s) | dominant bubble | context switches | grad micro-batch | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `cs_5508846183343714573` | activation, weight, chief, worker | ≈264 k (`ws_opt_perf_summary.json:6,16,29`) | 38.4 % activation receive | 12 (16.3 % runtime) | 1024 / 21 shards | No activation WIO rebinds; west-edge ingress fully utilized |
| `cs_12829142514842419922` | activation, weight, chief, worker | ≈345 k (`ws_opt_perf_summary.json:8,29,33,37,52`) | 39.5 % activation receive | 26 (50.2 % runtime) | 512 / 21 shards | Higher throughput but increased context churn and halved accumulation |

## Deep Dive: Topology, Fabric, & Hardware

### Topology & Placement
- Each role separates into `compiles/cs_*` and `sessions/00000*`, with placement artifacts (`ws_rt.mlir`, `cluster_cfg.json`, `wio_*`) in the compile trees and runtime telemetry (`dbg_kid_to_command_*.json`, `stream_stats.json`) in the sessions.
- Final floorplans confirm a consistent `core_height` 800 × `core_width` 704 footprint across all exports (`log-export-wsjob-cc4ypvcbgcpwtgaydam5wz-1e523ff4/.../final-floorplans.json:3`; analogous entries in the other bundles).
- `kernel_annotation.json`, `kernel_graph.json`, `kernel_tree.json`, and `ws_stack_llvm_stats.json` map the fused-kernel DAG; diffs between compile IDs highlight the shift from optimizer-heavy training to eval-style execution.

### Fabric Scheduling & Runtime
- Optimizer context summaries highlight context 7 peaking close to 10 k active SRAM slots with oscillations between contexts 0↔7 and 4↔7 (`ws_opt_context_summary.txt:1`, `ws_opt_context_switch_matrix.txt:1`). Context breakdowns (`ws_opt_context_breakdown.txt:362,382,414`) repeatedly list `rxact` ingress kernels and `ews_dynamic` staging code, reinforcing ingress as the bottleneck.
- WIO telemetry (`wio_flows.json:1`) enumerates 246 ingress flows—62 of them at `x_coord=0` with `txrx_mode` `7:1`—and `wio_report.txt:44-76` shows 123/124 slots engaged, with the right edge at 61/62.
- Session snapshots capture the transient queue depth shift: `activation-1/sessions/000001/stream_stats.json:55-57` records ≈64 MiB ring buffers (pre-drain export), while `000002` idles near 0.03–0.09 MiB (`activation-1/sessions/000002/stream_stats.json:14-57`). Moments of ≈2.5 MiB accumulation (`activation-14/sessions/000002/stream_stats.json:14-57`) align with west-edge pressure.
- Weight-stream statistics retain sentinel `0xDEADBEEF` values for `bytes_fetched` (`activation-1/sessions/000002/stream_stats.json:81-110`); treat these as placeholders when reconciling host transfer logs.

### Hardware Counters
- Baseline compile (`cs_5508846183343714573`) sustains ≈264 k samples/s with 38.4 % activation receive bubbles and 16.3 % cycle loss across 12 context switches; gradient accumulation spans 1024 micro-batch items over 21 activation shards (`ws_opt_perf_summary.json:6,16,29,34,37,52`).
- Alternative compile (`cs_12829142514842419922`) reaches ≈345 k samples/s but pays for it with 26 context switches (50.2 % runtime) and a reduced 512-item accumulation (`ws_opt_perf_summary.json:8,29,33,37,52`).
- Both passes avoid activation WIO rebinds, rely on ≤2 weight-side reconfigurations, and leave recompute disabled (`ws_opt_perf_summary.json:36,43,48` in both directories).

## Log Inspection Workflow & Artifacts

### Log Inspection Workflow
- Step 1 – Discover exports: list bundles under `log/` with `ls log` to identify the relevant `log-export-wsjob-*` directories plus host logs (`log:1-6`).
- Step 2 – Enumerate roles: for a given export, run `ls log/log-export-.../<bundle>/` to confirm available activation/weight/chief/worker captures (`log-export-wsjob-cc4ypvcbgcpwtgaydam5wz-1e523ff4/log-export-wsjob-cc4ypvcbgcpwtgaydam5wz-1e523ff4:1-47`).
- Step 3 – Review compile artifacts: inspect `compiles/cs_*` for each role (for example `ls .../activation-0/compiles`) and open `ws_rt.mlir`, `ws_km.mlir`, and `wio_flows.json` to understand placement and kernel wiring (`activation-0/compiles:1-4`).
- Step 4 – Check runtime telemetry: traverse `sessions/00000*` to read `stream_stats.json`, `dbg_kid_to_command_*.json`, and `.compile_artifact_location.out` pointers (`activation-1/sessions/000001:1-6`, `chief-0/sessions/000002:1-7`).
- Step 5 – Summarize performance: load `ws_opt_perf_summary.json` with `python -m json.tool` or inline scripts to extract throughput, bubbles, and context switches (`cs_5508846183343714573/ws_opt_perf_summary.json` dump in analysis).
- Step 6 – Correlate WIO pressure: consult `wio_report.txt` and `wio_flows.json` for slot usage, ingress lanes, and flow distribution (`wio_report.txt:44-76`; `wio_flows.json:1-35`).
- Step 7 – Validate host context: tail `a.txt` for launcher chronology and job IDs, and `b.txt` for dependency warnings (`log/a.txt:1-210`, `log/b.txt:1-200`).
- Step 8 – Capture diffs: use `diff -u` on MLIR and kernel annotations across compile IDs to highlight functional changes (`kernel_annotation.json` diff excerpt near line 13721).

### Artifact Inventory
- `log-export-wsjob-cc4ypvcbgcpwtgaydam5wz-1e523ff4` — full multi-role export (activation/weight/chief/worker) with both compile IDs mirrored across `compiles/` and `sessions/`. Weight bootstrap errors in `activation-0/dbg_act_0.out:1` point to missing kernel modules when the bundle was captured offline.
- `log-export-wsjob-gfeghe672j7visbaqfhdky-1e523ff4` — coordinator-focused capture retaining optimizer context dumps (`ws_opt_context_summary.txt:1`, `ws_opt_context_switch_matrix.txt:1`) for `cs_5508846183343714573`.
- `log-export-wsjob-it6afsqz4vdwxywthxx3pk-1e523ff4` — complementary coordinator bundle emphasizing `cs_12829142514842419922`; topology manifests (`fabric.json`, `cluster-details-config-map.json`) detail wafer-to-host wiring.
- `a.txt`, `b.txt` — host-side Python launcher output and dependency warnings, establishing provenance and runtime chronology.

### Host & Environment Signals
- `a.txt:1` logs the Nov 1 2025 Python launcher invoking `run.py` for GraphSAGE on `ogbn-arxiv` with 2 000 planned steps, no warm start, and no checkpoint reuse—handy for aligning host expectations with wafer captures.
- `b.txt:1` raises SciPy/NumPy incompatibilities (`NumPy >=1.25.2`) plus deprecation warnings.

## Gaps & Recommendations

### Telemetry Coverage Gaps
- **Power & Thermal:** The exports do not include wafer power draw, efficiency, density, or thermal telemetry. All `power` hits correspond to optimizer tensor names rather than energy metrics (for example `log-export-wsjob-cc4ypvcbgcpwtgaydam5wz-1e523ff4/.../weight-16/dbg_wgt_16.out:11-51`), and the coordinator/activation logs expose zero-copy memory throttle flags only (`log-export-wsjob-cc4ypvcbgcpwtgaydam5wz-1e523ff4/.../activation-9/dbg_act_9.out:1-10`) with no temperature or throttling counters.
- **Fabric / NoC micro-metrics:** Available artifacts (`wio_flows.json`, `wio_report.txt`, `stream_stats.json`) report topology and queue occupancy (for example `log-export-wsjob-cc4ypvcbgcpwtgaydam5wz-1e523ff4/.../activation-1/sessions/000001/stream_stats.json:48-64`), but there are no packet/flit latency, congestion scores, bisection bandwidth, or virtual-channel utilization summaries.
- **Memory bandwidth hierarchy:** Context summaries capture SRAM slot bounds (`log-export-wsjob-gfeghe672j7visbaqfhdky-1e523ff4/.../ws_opt_context_summary.txt:1-16`) yet omit achieved SRAM/DRAM bandwidth, contention rates, or cache statistics; the only bandwidth indicators are boolean flags such as `kernel_bandwidth_bound_accurate_directions` (`log-export-wsjob-cc4ypvcbgcpwtgaydam5wz-1e523ff4/.../activation-11/dbg_act_11.out:1-10`).
- **Compute-unit utilization:** Performance summaries list sample/s throughput, bubble percentages, and context-switch overheads (`log-export-wsjob-cc4ypvcbgcpwtgaydam5wz-1e523ff4/.../cs_5508846183343714573/ws_opt_perf_summary.json`) but provide no arithmetic-unit utilization, achieved TOPS/FLOPS, or stall-cycle breakdown beyond aggregate bubble figures.

### Outstanding Gaps
- Weight-stream bootstrap errors surface in `activation-0/dbg_act_0.out:1`; confirm kernel modules before re-running live compiles.
- Fabric counters expose persistent west-edge pressure with minimal right-edge headroom; consider redistributing ingress load if possible.
- Sentinel metrics (`0xDEADBEEF`) and missing kernel modules limit confidence in byte-accurate throughput comparisons.
- Power/thermal, detailed NoC, memory-bandwidth, and core-utilization telemetry are absent from the captured bundles; capturing those counters on the next run would enable the requested analyses.

### Recommended Actions
1. Pull the `opprofiler_flow.pb` traces from the remote runtime paths or adjust the export pipeline to bundle them so the newly enabled profiler yields usable data (`log/20251102/log-export-wsjob-yg9ufjrh6nuxkwpv6yxnrz-1e523ff4/activation-3/dbg_act_3.err:1-12`, `log/20251102/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/log-export-wsjob-ppcsgtrju8lbxdagl5n6sx-1e523ff4/coordinator-0/dbg_crd_0.err:8050-8073`).
2. Decide whether the eval-style 512 micro-batch compile or the 1024 training compile should remain canonical, then prune the redundant artifacts accordingly.
3. Pair `ws_opt_perf_summary.json` with context telemetry to quantify ingress bottlenecks before queuing another hardware submission.
4. Repair the Python environment (`uv` + NumPy/SciPy versions) to stop launcher retries and reproduce the higher-throughput compile deterministically.
5. Augment future exports with device power/thermal counters, detailed fabric/NOC metrics, memory-bandwidth stats, and core-utilization logs to close the telemetry gaps.
6. After analysis, clear transient artifacts (`executors/`, `model_dir_gnn/ir_debug/`) so future exports remain concise and auditable.
