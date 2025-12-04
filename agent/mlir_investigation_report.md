# MLIR Investigation Report

## Overview
This report summarizes the findings from analyzing two key MLIR files generated during the compilation of the GraphSAGE model: `cirh-with-mask-ranges.mlir` and `ws_rt.mlir`. These files represent different stages of the compilation pipeline, from high-level graph representation to low-level runtime execution instructions.

## 1. High-Level Graph: `cirh-with-mask-ranges.mlir`
This file represents the computation graph at a high level, likely after some initial lowering and optimization but before mapping to the specific hardware runtime.

### Key Findings:
- **Model Structure:** The graph corresponds to a GraphSAGE model. It includes typical operations such as:
    - `cirh.MatMul`: Matrix multiplications for linear layers.
    - `cirh.Relu`: Activation functions.
    - `cirh.L2Norm`: Normalization.
    - `cirh.Reduce`, `cirh.BroadcastInDim`: Tensor manipulation for aggregation.
    - `cirh.SparseSoftmaxCrossEntropyWithLogits`: Loss calculation.
- **Mixed Precision:** The model uses a mix of `f16` (half-precision) and `f32` (single-precision) types. This is evident in operation signatures and cast operations (`cirh.Cast`), indicating optimization for performance on the hardware.
- **Graph Attributes:**
    - `cs.batch_size = 1024`: The batch size is explicitly defined.
    - `cs.num_csx = 1`: The compilation is targeting a single CSX system.
- **Source Mapping:** Operations are tagged with `loc(...)` attributes that trace back to the original PyTorch source code (e.g., `torch/nn/functional.py`, `cerebras/modelzoo/trainer/trainer.py`), facilitating debugging and performance analysis.

## 2. Low-Level Runtime: `ws_rt.mlir`
This file contains the "Wafer-Scale Runtime" (ws_rt) instructions. It is a low-level representation that dictates how the host (controller) interacts with the wafer and manages execution.

### Key Findings:
- **Memory Layouts:** The file defines hundreds of `#layout` attributes (e.g., `#layout0` to `#layout247`...). These specify precise memory organizations, including shapes, reshapes, and replication strategies, crucial for efficient data movement on the wafer.
- **Host Orchestration:** The execution is organized into concurrent threads on the host:
    - `ws_rt.host.iter_thread @rx`: Handles receiving data (likely results or gradients) from the wafer.
    - `ws_rt.host.iter_thread @tx`: Handles sending data (inputs, weights) to the wafer.
    - `ws_rt.host.iter_thread @compute`: Manages host-side computations and synchronization.
- **Explicit Memory Management:** The runtime uses explicit allocation and deallocation:
    - `ws_rt.acth.alloc`: Allocates activation buffers.
    - `ws_rt.acth.release`: Frees buffers when they are no longer needed.
    - `ws_rt.acth.finalize`: Marks buffers as ready or completed.
- **Communication Primitives:**
    - `ws_rt.acth.send` / `ws_rt.acth.recv`: Instructions for data transfer between host and wafer.
    - Attributes like `cycles_with_io` and `io_bits` suggest that the compiler models the cost and bandwidth of these transfers.
- **Synchronization:**
    - `ws_rt.event.declare`, `ws_rt.event.set`, `ws_rt.event.get`: Event-based synchronization is used to coordinate between the `@rx`, `@tx`, and `@compute` threads.
    - `ws_rt.wgth.host_aggregate_barrier`: Barriers for synchronizing weight updates or reductions.
- **Weight Handling:** Specific operations for weight management (`ws_rt.wgth...`) such as `l2norm`, `add` (for updates), and `sum_all_reduce_start` (for gradient aggregation) are present, showing how the optimizer step is offloaded or managed.

## Conclusion
The investigation reveals a sophisticated compilation stack. `cirh-with-mask-ranges.mlir` captures the semantic intent of the model with hardware-agnostic optimizations, while `ws_rt.mlir` lowers this into a highly detailed execution plan. The runtime MLIR highlights the complexity of orchestrating a wafer-scale engine, involving precise memory layout management, multi-threaded host control, and fine-grained synchronization and communication scheduling.

## 3. Additional MLIR Insights & Diffs

### MLIR Insights
`cirh-with-mask-ranges.mlir` confirms high-level GraphSAGE structure with mixed precision. `ws_rt.mlir` reveals explicit runtime memory management and multi-threaded host orchestration (`@rx`, `@tx`, `@compute`).

### MLIR Diff Highlights (Action 1)
- Micro-batch tiling shrank from 1024 nodes to 512: training compile keeps `#ws.named_dim` entries at 1024 (`log/log-export-wsjob-gfeghe672j7visbaqfhdky-1e523ff4/log-export-wsjob-gfeghe672j7visbaqfhdky-1e523ff4/cs_5508846183343714573/ws_rt.mlir:275-277`), while the follow-up compile rewrites them to 512 and overhauls the associated layouts to 512-wide shards (`log/log-export-wsjob-it6afsqz4vdwxywthxx3pk-1e523ff4/log-export-wsjob-it6afsqz4vdwxywthxx3pk-1e523ff4/cs_12829142514842419922/ws_rt.mlir:1-5`, `log/log-export-wsjob-it6afsqz4vdwxywthxx3pk-1e523ff4/log-export-wsjob-it6afsqz4vdwxywthxx3pk-1e523ff4/cs_12829142514842419922/ws_rt.mlir:189-191`).
- Runtime initialization diverges: the training pass seeds accumulation buffers with a 9.77e‑4 scaling factor (`log/log-export-wsjob-gfeghe672j7visbaqfhdky-1e523ff4/log-export-wsjob-gfeghe672j7visbaqfhdky-1e523ff4/cs_5508846183343714573/ws_rt.mlir:477`), whereas the second compile writes ones into the same buffers before ingress (`log/log-export-wsjob-it6afsqz4vdwxywthxx3pk-1e523ff4/log-export-wsjob-it6afsqz4vdwxywthxx3pk-1e523ff4/cs_12829142514842419922/ws_rt.mlir:398`).
- Graph composition shifts from optimizer-heavy training to eval-style inference: the first `ws_km` loads nine optimizer state tensors per layer alongside parameter norms (`log/log-export-wsjob-gfeghe672j7visbaqfhdky-1e523ff4/log-export-wsjob-gfeghe672j7visbaqfhdky-1e523ff4/cs_5508846183343714573/ws_km.mlir:22-70`), while the second `ws_km` drops all optimizer loads and instead pulls accuracy metric accumulators plus the raw model weights (`log/log-export-wsjob-it6afsqz4vdwxywthxx3pk-1e523ff4/log-export-wsjob-it6afsqz4vdwxywthxx3pk-1e523ff4/cs_12829142514842419922/ws_km.mlir:20-31`).
- Call-stack metadata mirrors the mode change: training traces reference gradient norm hooks (`log/log-export-wsjob-gfeghe672j7visbaqfhdky-1e523ff4/log-export-wsjob-gfeghe672j7visbaqfhdky-1e523ff4/cs_5508846183343714573/ws_rt.mlir:272-274`), whereas the alternate compile pivots to accuracy-metric bookkeeping (`log/log-export-wsjob-it6afsqz4vdwxywthxx3pk-1e523ff4/log-export-wsjob-it6afsqz4vdwxywthxx3pk-1e523ff4/cs_12829142514842419922/ws_rt.mlir:184-188`).
