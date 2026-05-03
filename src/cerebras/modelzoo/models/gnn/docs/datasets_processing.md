# GraphSAGE Data Processing Details

This document outlines the data processing strategies for GraphSAGE implementations in `src/cerebras/modelzoo/models/gnn`.

## 1. Implementations

- **CSX (ModelZoo)**: `src/cerebras/modelzoo/models/gnn/data_processing/processor.py` (delegates to `pipelines/neighbor_sampling.py`)
- **GPU (PyG)**: `src/cerebras/modelzoo/models/gnn/reference/pyg/train.py` (uses `pyg_gnn/data.py` for loading)

## 2. Processing Policies

### Directionality (Directed vs. Undirected)
Both implementations standardize on **Undirected** graphs for the supported datasets.

- **CSX**:
  - In `pipelines/common.py`, `to_undirected` is explicitly applied to the `edge_index` during `prepare_graph_components`.
- **GPU**:
  - In `pyg_gnn/data.py`, `to_undirected` is applied to OGB datasets (`ogbn-*`).
  - For Planetoid and Reddit, the raw datasets are typically treated as undirected symmetric graphs in standard PyG usage.

### Self-loops
- **CSX**:
  - `pipelines/neighbor_sampling.py` builds a neighbor table from the given `edge_index`. It does **not** explicitly add self-loops during component preparation.
  - If self-loops are required for the aggregation step (e.g. including central node features), they are expected to be handled by the model architecture (separating central node from neighbors) rather than adding explicit edges to the graph topology.
- **GPU**:
  - `pyg_gnn/data.py` does not explicitly call `add_self_loops`.
  - `NeighborLoader` samples based on the provided `edge_index`.
  - Similar to CSX, the GraphSAGE operator typically handles the central node's feature ($x_u$) separately from the neighbor aggregation ($x_{\mathcal{N}(u)}$).

### Heterogeneous Graphs
Both implementations explicitly convert supported heterogeneous datasets to **Homogeneous** graphs.

- **Target Strategy**: Focuses on the primary node type (e.g., "paper" for `ogbn-mag`).
- **CSX**:
  - `pipelines/common.py` checks for `ogbn-mag`.
  - Extracts "paper" node features.
  - Extracts "paper-to-paper" (or "paper-cites-paper") edges and makes them undirected.
  - Raises `NotImplementedError` for other heterogeneous datasets.
- **GPU**:
  - `pyg_gnn/data.py` (`load_dataset`) performs the exact same reduction for `ogbn-mag`: selects "paper" nodes/labels and "paper-paper" edges, then converts to undirected.

### Label & Evaluation Targets
- **CSX**:
  - Uses `split_masks` (boolean tensors) derived from the dataset's split indices.
  - Targets are filtered based on these masks during batch construction (`GraphSAGENeighborSamplerDataset`).
- **GPU**:
  - Uses `split_idx` (dictionary of indices) to define `input_nodes` for the `NeighborLoader`.
  - For `ogbn-mag`, effectively predicts on "paper" nodes using "paper" labels.

### Node Structural Features
Degree-derived node features are not part of the main GNN data pipeline.

A local PubMed experiment evaluated a single additional structural feature:
`standardize(log1p(degree))` concatenated to the input node feature matrix. The
run used fresh model directories and CPU `uv run cszoo fit` executions with
evaluation metrics enabled.

| Model | Baseline final validation accuracy | With degree feature | Delta |
| :--- | ---: | ---: | ---: |
| GATv2 | 0.7419999838 | 0.7540000081 | +0.0120 |
| Graph Transformer | 0.2020000070 | 0.3880000114 | +0.1860 |

The GATv2 improvement is small enough to be treated as inconclusive without
multi-seed confirmation. The Graph Transformer final-step value improved, but
the baseline already reached similar intermediate validation accuracy around
0.42, so this result does not establish a stable structural-feature benefit.

For the HPC-oriented path, the extra feature plumbing is intentionally omitted.
The additional configuration surface, `n_feat` bookkeeping, data-processing
branches, and CPU/GPU/CSX parity checks would broaden the implementation
without directly advancing the main objectives: static-shape execution,
throughput, memory movement, scaling, and compile/runtime behavior. Learned
degree embeddings would require still more model-side surface area, so they
should remain out of scope unless a repeated ablation shows a robust accuracy
gain that is large enough to justify that complexity.

## Summary Table

| Feature | CSX (`data_processing`) | GPU (`pyg_gnn`) |
| :--- | :--- | :--- |
| **Graph Type** | Undirected (Converted via `to_undirected`) | Undirected (Converted for OGB, implicit for others) |
| **Self-loops** | Not added to graph topology | Not added to graph topology |
| **Heterogeneous** | Converted to Homogeneous (e.g. MAG -> Paper-Paper) | Converted to Homogeneous (e.g. MAG -> Paper-Paper) |
| **Targets** | Mask-based selection | Index-based selection |
