# Running GraphsAGE with PyG-GNN

This document describes how to execute the distributed GraphSAGE training.

## Usage

### Single Process (Development / Debugging)
You can run the script normally with `uv run`. This uses a single GPU (or CPU).

```bash
uv run src/cerebras/modelzoo/models/gnn/graphsage_pyg.py \
    --config src/cerebras/modelzoo/models/gnn/configs/params_graphsage_ogbn_arxiv.yaml
```

### Multi-GPU (Distributed Data Parallel - DDP)
To execute on multiple GPUs, you must use `torchrun`. This spawns multiple processes (one per GPU).

**Example: Run on 2 GPUs**

```bash
uv run torchrun --nproc_per_node=2 src/cerebras/modelzoo/models/gnn/graphsage_pyg.py \
    --config src/cerebras/modelzoo/models/gnn/configs/params_graphsage_ogbn_arxiv.yaml
```

**Note on `uv run torchrun`**:
`uv run` ensures the command is executed within the project's virtual environment. `torchrun` is the PyTorch-provided utility to manage distributed workers.

## Arguments
- `--nproc_per_node=<N>`: Number of GPUs to use on the current node.
- `--config`: Path to the YAML configuration file.

## CAGNET (Communication-Avoiding SpMM)

This model uses optimized 1.5D, 2D, or 3D algorithms for Sparse Matrix-Matrix Multiplication (SpMM) to reduce communication overhead on multi-GPU setups.
This is the default mode of operation (equivalent to baseline OFFSET-GNN).

### Configuration

Specify the process grid dimensions to match your multi-GPU setup. If not specified, it defaults to 1x1 (local execution).

**Example: 2 GPUs (1x2 Grid)**
```bash
uv run torchrun --nproc_per_node=2 src/cerebras/modelzoo/models/gnn/graphsage_pyg.py \
    --config src/cerebras/modelzoo/models/gnn/configs/params_graphsage_ogbn_arxiv.yaml \
    --cagnet-rows 1 \
    --cagnet-cols 2
```

### CAGNET Arguments

- `--cagnet-rows <R>`: Number of row groups (R).
- `--cagnet-cols <C>`: Number of column groups (C).
- `--cagnet-rep <K>`: Replication factor (usually 1 for 1.5D/2D).

**Constraints**:
- The total number of GPUs (world size) must equal `rows * cols * rep`.
- For 1.5D/2D decomposition (recommended), keep `rep=1` and ensure `rows * cols == N_GPUs`.
 
### Performance Optimization (Custom Kernels)
 
To achieve maximum performance (matching OFFSET-GNN baseline), it is highly recommended to install the custom CUDA kernel for sparse matrix operations.
Since this model always uses the CAGNET implementation path (`CagnetSAGE`), it relies on these kernels for efficient execution even on a single GPU.
Without this kernel, the model will fallback to a slower PyTorch implementation (torch.sparse.mm).
 
```bash
cd src/cerebras/modelzoo/models/gnn/pyg_gnn/ops
uv run setup.py install
```
