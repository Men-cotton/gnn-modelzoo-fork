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

You can enable CAGNET-optimized distributed training using CLI arguments. This mode uses optimized 1.5D, 2D, or 3D algorithms for Sparse Matrix-Matrix Multiplication (SpMM) to reduce communication overhead on multi-GPU setups.

### Enabling CAGNET

Simply add the `--use-cagnet` flag and specify the process grid dimensions.

**Example: 2 GPUs (1x2 Grid)**
```bash
uv run torchrun --nproc_per_node=2 src/cerebras/modelzoo/models/gnn/graphsage_pyg.py \
    --config src/cerebras/modelzoo/models/gnn/configs/params_graphsage_ogbn_arxiv.yaml \
    --use-cagnet \
    --cagnet-rows 1 \
    --cagnet-cols 2
```

### CAGNET Arguments

- `--use-cagnet`: Enable the CAGNET backend.
- `--cagnet-rows <R>`: Number of row groups (R).
- `--cagnet-cols <C>`: Number of column groups (C).
- `--cagnet-rep <K>`: Replication factor (usually 1 for 1.5D/2D).

**Constraints**:
- The total number of GPUs (world size) must equal `rows * cols * rep`.
- For 1.5D/2D decomposition (recommended), keep `rep=1` and ensure `rows * cols == N_GPUs`.
