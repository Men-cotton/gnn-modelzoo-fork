from __future__ import annotations

import cerebras.pytorch as cstorch
import torch


def validate_single_streamer() -> None:
    # Each item in these loaders is already a complete logical GNN batch.
    # Supporting multiple streamers requires global-batch sharding, including
    # the SDK's per-CSX subdivision and epoch/evaluation ordering contracts.
    if cstorch.use_cs() and cstorch.distributed.num_streamers() > 1:
        raise ValueError(
            "GNN data loaders currently support one CSX input streamer only; "
            "configure num_csx=1 and num_workers_per_csx=1. Multiple streamers "
            "would duplicate graph batches. This limit does not restrict the "
            "PyTorch DataLoader num_workers setting."
        )


def to_dense_adjacency(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    num_nodes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=dtype)
    if edge_index.numel() > 0:
        edge_index_long = edge_index.to(dtype=torch.long)
        adjacency.index_put_(
            (edge_index_long[0], edge_index_long[1]),
            edge_weight.to(dtype=dtype),
            accumulate=True,
        )
    return adjacency.unsqueeze(0)


__all__ = ["to_dense_adjacency", "validate_single_streamer"]
