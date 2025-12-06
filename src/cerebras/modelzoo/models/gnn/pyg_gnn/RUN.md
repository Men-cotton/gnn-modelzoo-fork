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
