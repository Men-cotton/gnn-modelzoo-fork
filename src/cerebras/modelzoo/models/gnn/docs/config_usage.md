# Config Usage Investigation Report

This document outlines how the configuration file `src/cerebras/modelzoo/models/gnn/configs/params_graphsage_ogbn_arxiv.yaml` is used to instantiate the `GraphSAGE` architecture defined in `src/cerebras/modelzoo/models/gnn/architectures/graphsage.py`.

## Overview

The configuration flow involves an intermediate model wrapper class `GNNModel` (defined in `src/cerebras/modelzoo/models/gnn/model.py`). The parameters from the YAML file are first parsed into a `GNNModelConfig` object, which is then used by `GNNModel` to instantiate the underlying `GraphSAGE` architecture with the correct arguments.

## 1. Configuration Source (`params_graphsage_ogbn_arxiv.yaml`)

The relevant section in the YAML file is under `trainer.init.model`. These parameters define the model architecture.

```yaml
model:
  name: graphsage

  # Dataset-specific dimensions for ogbn-arxiv
  n_feat: 128
  n_class: 40

  # ... other params ...
  core_architecture: GraphSAGE
  graphsage_hidden_dim: 256
  graphsage_num_layers: 2
  graphsage_dropout: 0.5
  graphsage_aggregator: mean
```

## 2. Intermediate Parsing (`model.py`)

The class `GNNModel` in `src/cerebras/modelzoo/models/gnn/model.py` acts as the point of entry. It uses `GNNModelConfig` to validate and structure the parameters.

### `GNNModelConfig`
This class defines the schema for the configuration, matching the keys in the YAML file:

```python
class GNNModelConfig(GNNArchConfig):
    # ...
    n_feat: int  # Inherited from GNNArchConfig
    n_class: int # Inherited from GNNArchConfig
    
    graphsage_hidden_dim: int = 128 # Code Default
    graphsage_num_layers: int = 2   # Code Default
    graphsage_dropout: float = 0.5
    graphsage_aggregator: str = "mean"
    # ...
```

### `GNNModel.build_model`
The `build_model` method uses these configuration values to initialize the `GraphSAGE` class.

```python
        elif architecture == "graphsage":
            core = GraphSAGE(
                input_dim=model_config.n_feat,
                hidden_dim=model_config.graphsage_hidden_dim,
                num_layers=model_config.graphsage_num_layers,
                dropout=model_config.graphsage_dropout,
                aggregator=model_config.graphsage_aggregator,
                num_classes=model_config.n_class,
            )
```

## 3. Architecture Initialization (`graphsage.py`)

Finally, `src/cerebras/modelzoo/models/gnn/architectures/graphsage.py` receives the arguments in its `__init__` method.

```python
class GraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        aggregator: str,
        num_classes: int,
    ):
        # ...
```

## Parameter Mapping Summary

| YAML Key (`params_graphsage_ogbn_arxiv.yaml`) | `GNNModelConfig` Attribute | Default (if missing) | `GraphSAGE` Argument | Value (ogbn-arxiv) |
| :--- | :--- | :--- | :--- | :--- |
| `n_feat` | `n_feat` | - | `input_dim` | 128 |
| `graphsage_hidden_dim` | `graphsage_hidden_dim` | 128 | `hidden_dim` | 256 |
| `graphsage_num_layers` | `graphsage_num_layers` | 2 | `num_layers` | 2 |
| `graphsage_dropout` | `graphsage_dropout` | `dropout` | 0.5 |
| `graphsage_aggregator` | `graphsage_aggregator` | `aggregator` | "mean" |
| `n_class` | `n_class` | `num_classes` | 40 |

## 4. Mathematical Formulation & Matrix Dimensions

The `GraphSAGE` implementation in `architectures/graphsage.py` uses the **Sum-based Aggregation** (often referred to as *Mean Aggregator* with separate linear transformations for self and neighbor features). Note that standard PyTorch `nn.Linear` layers **include bias** by default.

### 4.1. Mathematical Formulation

This implementation uses **Sum-based Aggregation with Masking** (which corresponds to `mean` aggregator via normalization). While the original GraphSAGE paper (Hamilton et al. 2017) defines the operation using **Concatenation**, the formulation used here is mathematically equivalent (isomorphic) provided the weights are split.

*   **Original GraphSAGE (Concat)**: $h_v^{(k)} = \sigma(W \cdot \text{CONCAT}(h_v^{(k-1)}, h_{\mathcal{N}(v)}^{(k-1)}))$
*   **This Implementation (Sum/Add)**: $h_v^{(k)} = \sigma(W_{\text{self}} h_v^{(k-1)} + W_{\text{neighbor}} h_{\mathcal{N}(v)}^{(k-1)})$

By defining $W = [W_{\text{self}} | W_{\text{neighbor}}]$, the operations are identical. The implementation uses separate linear layers ($W_{\text{self}}$ and $W_{\text{neighbor}}$) to process self and neighbor features independently before summing.

For a given layer $k$ and node $v$, the update rule is:

1.  **Masked Aggregation**:
    $$
    h_{\mathcal{N}(v)}^{(k-1)} = \frac{\sum_{u \in \mathcal{N}(v)} (h_u^{(k-1)} \cdot M_{v,u})}{\max(1, \sum_{u \in \mathcal{N}(v)} M_{v,u})}
    $$
    *Where $M_{v,u}$ is the mask (1 if valid neighbor, 0 if padding).*

2.  **State Update**:
    $$
    h_v^{(k)} = \sigma \left( \text{Dropout} \left( W_{\text{self}}^{(k)} h_v^{(k-1)} + b_{\text{self}}^{(k)} + W_{\text{neighbor}}^{(k)} h_{\mathcal{N}(v)}^{(k-1)} + b_{\text{neighbor}}^{(k)} \right) \right)
    $$

    *Note on Bias: The implementation uses two `nn.Linear` layers (`self_linear` and `neighbor_linear`), both with `bias=True` by default. This results in two bias vectors being added ($b_{\text{self}} + b_{\text{neighbor}}$). This is mathematically redundant (equivalent to a single bias) and consumes additional memory bandwidth. A simpler implementation would disable bias on the linear layers and add a single bias vector after the sum.*

### 4.2. Matrix Dimensions

The dimensions of the weight matrices depend on the layer index, specifically differing between the first layer (projection) and subsequent layers.

#### Layer 1 (Input/Projection Layer)
Transforms input features (`n_feat=128`) to hidden state (`hidden_dim=256`).
*   **Weight Matrices**:
    *   $W_{\text{self}}^{(1)}$: Shape `[256, 128]` (Input: 128, Output: 256) - *PyTorch `nn.Linear` stores weights as `[Out, In]`, so internal shape is `[256, 128]`. Math operations are $y = xW^T + b$.*
    *   $W_{\text{neighbor}}^{(1)}$: Shape `[256, 128]`
*   **Bias**: Vector of size `[256]`

#### Layer 2 (Hidden Layer)
Transforms hidden state to hidden state.
*   **Weight Matrices**:
    *   $W_{\text{self}}^{(2)}$: Shape `[256, 256]`
    *   $W_{\text{neighbor}}^{(2)}$: Shape `[256, 256]`
*   **Bias**: Vector of size `[256]`

#### Classifier Head (Post-Processing)
This is an independent `nn.Linear` layer applied **after** the GraphSAGE message passing loops are completed. It operates only on the final embeddings of the target nodes.
*   **Matrix**: Shape `[40, 256]` (Input: 256, Output: `n_class=40`)

## 5. Batch Size & Receptive Field Analysis

In GNN sampling, `batch_size` only defines the number of root nodes for loss calculation. The actual computational load is determined by the `fanouts` configuration, which defines the **Receptive Field**.

*   **Configured Batch Size**: `1024` (Seed Nodes)
*   **Configured Fanouts**: `[25, 15]` (from YAML `dataset_profiles`)
    *   Hop 1 (Neighbors of Seeds): 25 neighbors per node.
    *   Hop 2 (Neighbors of Neighbors): 15 neighbors per node (Leaf level).

### 5.1. Effective Node Count (Tree Expansion vs. Subgraph)
The implementation uses a **Tree Expansion** strategy to ensure static shapes. This differs significantly from standard **Subgraph Sampling** (e.g., PyG's `NeighborLoader`) which deduplicates nodes.

1.  **Depth 0 (Target Nodes)**: `1024`
2.  **Depth 1 (Hop 1 Neighbors)**: $1024 \times 25 = 25,600$
3.  **Depth 2 (Hop 2/Leaves)**: $25,600 \times 15 = 384,000$

**Total Input Features Loaded:** 384,000 nodes $\times$ 128 float32 dimensions.

*   **Redundancy**: `384,000` is the number of nodes in the expanded computation tree. If a node appears multiple times in the neighborhood of the batch, it is duplicated in the input tensor. This redundancy allows for fully static execution (no indirect memory addressing on device) but increases the data volume compared to deduplicated subgraphs.

## 6. Static Shapes & Padding (Cerebras Optimization)

The implementation in `data_processing/pipelines/neighbor_sampling.py` enforces **Static Shapes**, which is critical for maximizing performance on Cerebras hardware (and other accelerators).

*   **Padding**:
    *   The `GraphSAGENeighborSamplerDataset` creates tensors of fixed size `[num_parents, fanout]`.
    *   If a node has fewer neighbors than `fanout`, the remaining slots are filled with a `pad_id` (default 0).
    *   Example: If a node has 5 neighbors but fanout is 25, then 20 slots are padded.

*   **Masking**:
    *   Attributes `node_masks` and `neighbor_masks` are generated alongside features.
    *   The `_aggregate_neighbors` function in `graphsage.py` uses these masks to zero out the contribution of padded nodes before the `sum` or `mean` operation.

*   **Performance Implication (Effective vs Hardware FLOPs)**:
    *   **Hardware FLOPs**: The hardware measures matrix products for the full dense shapes, including padded regions.
    *   **Effective FLOPs**: The "useful" work depends on the actual degree of the graph. On sparse graphs, the ratio of Effective/Hardware FLOPs may be low because operations on padded zeros contribute nothing to the result. This is the trade-off for eliminating control flow divergence and dynamic memory access.

## 7. Arithmetic Intensity & Bottleneck Analysis

A Roofline analysis requires considering both the **Host-side data preparation** and the **Device-side execution**.

### 7.1. Aggregation Phase (Data Transfer & Aggregation)
*   **Location**: The random-access "Gather" of feature vectors happens on the **Host (CPU)** during the `Dataset.__getitem__` call.
*   **Device Operation**: The Accelerator receives fully materialized, expanded tensors. It performs a reduction (Sum/Mean) over the `fanout` dimension.
*   **Bottleneck**:
    *   **I/O Bandwidth**: Transferring the expanded tree data (384,000 nodes per batch) from Host to Device is likely the primary bottleneck. The volume is significantly larger than a deduplicated subgraph.
    *   **Compute efficiency**: The aggregation on device involves summing padded zeros, which consumes memory bandwidth and cycles without increasing effective throughput.

### 7.2. Linear Transformation Phase (Compute Bound)
*   **Operation**: Dense Matrix Multiplication ($Y = XW^T$).
*   **Intensity**: $\approx 131,072 / 512 \approx 256$ FLOPs/Byte (for Layer 1).
*   **Characteristics**: High intensity, likely to utilize Tensor Cores / Matrix Engines efficiently.

### 7.3. Conclusion
The model's wall-clock performance is likely dominated by **Data Transfer (Host-to-Device)** and **Preprocessing (CPU Gather)** due to the Tree Expansion strategy. While the dense matrix multiplications on the device are efficient, the overhead of moving redundant, expanded data limits the overall throughput. The architectural choice of Static Shapes trades increased I/O volume for guaranteed utilization of the massive compute core.

## 8. Configuration Details: `sampler_seed`

*   **Value**: `42`
*   **Effect**: The `sampler_seed` ensures determinism in the data loading and sampling process. Specifically, it controls:
    1.  **Batch Composition**: If `shuffle: True`, this seed is used to initialize the random number generator that shuffles the order of target nodes (`_order_targets` in `neighbor_sampling.py`).
    2.  **Neighbor Selection**: It is used in the `_deterministic_choice` method to deterministically select a subset of neighbors when the number of neighbors exceeds the specified `fanout`.

## 9. Investigation of `pyg_gnn` Scripts (Comparison with `cszoo`)

The directory `src/cerebras/modelzoo/models/gnn/pyg_gnn` contains a separate set of scripts (entry point: `graphsage_pyg.py`) intended for running GraphSAGE using standard PyTorch Geometric (PyG) implementations, likely for benchmarking or reference purposes.

This section highlights the key differences between the standard `cszoo` flow (investigated above) and this `pyg_gnn` script flow.

### 9.1. Model Architecture Differences

*   **`cszoo` Flow**:
    *   **Class**: `src/cerebras/modelzoo/models/gnn/architectures/graphsage.py` (`GraphSAGE`)
    *   **Implementation**: Cerebras custom implementation of GraphSAGE.
    *   **Config Usage**: `graphsage_num_layers` from the YAML is correctly passed to the model constructor.

*   **`pyg_gnn` Flow (`graphsage_pyg.py`)**:
    *   **Class**: `src/cerebras/modelzoo/models/gnn/pyg_gnn/model.py` (`GraphSAGEWrapper`) wrapping `torch_geometric.nn.models.GraphSAGE`.
    *   **Implementation**: Standard PyTorch Geometric implementation.
    *   **Config Usage**:
        *   `n_feat`, `graphsage_hidden_dim`, `n_class`, `graphsage_dropout` are used.
        *   `graphsage_num_layers` is used to determine `num_layers`.

### 9.2. Data Loading Differences

*   **`cszoo` Flow**:
    *   **Loader**: `src/cerebras/modelzoo/models/gnn/data_processing/pipelines/neighbor_sampling.py` (`GraphSAGENeighborSamplerDataset`).
    *   **Mechanism**: Custom implementation designed to yield `GraphSAGEBatch` objects compatible with the Cerebras execution model.
    *   **Logic**: Manually handles sampling logic and determinism.

*   **`pyg_gnn` Flow**:
    *   **Loader**: `src/cerebras/modelzoo/models/gnn/pyg_gnn/data.py` uses `torch_geometric.loader.NeighborLoader` (or `DistNeighborLoader` for distributed).
    *   **Mechanism**: Standard PyG loader.
    *   **Seed**: The `sampler_seed` from config is used to seed a `torch.Generator`, which is passed to the `NeighborLoader`. This controls shuffling and sampling randomness in the standard PyG way.

### 9.3. Summary of Differences

| Feature | `cszoo` Flow | `pyg_gnn` Flow |
| :--- | :--- | :--- |
| **Entry Point** | `cszoo fit` | `graphsage_pyg.py` |
| **Model Implementation** | Custom `architectures/graphsage.py` | PyG `torch_geometric.nn.models.GraphSAGE` |
| **Data Loader** | Custom `GraphSAGENeighborSamplerDataset` | PyG `NeighborLoader` |
