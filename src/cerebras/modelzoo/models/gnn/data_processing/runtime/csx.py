from __future__ import annotations

import torch


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


__all__ = ["to_dense_adjacency"]
