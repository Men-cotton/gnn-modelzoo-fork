from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cerebras.pytorch as cstorch
import torch


@dataclass(frozen=True)
class EdgeIndexAdjacency:
    """Container for edge_index / edge_weight tensors."""

    edge_index: torch.Tensor
    edge_weight: torch.Tensor

    def to(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "EdgeIndexAdjacency":
        target_dtype = torch.int32 if cstorch.use_cs() else torch.long
        edge_index = self.edge_index
        if edge_index.dtype != target_dtype or (
            device is not None and edge_index.device != device
        ):
            edge_index = edge_index.to(
                device=device or edge_index.device, dtype=target_dtype
            )
        edge_weight = self.edge_weight
        if device is not None or dtype is not None:
            edge_weight = edge_weight.to(
                device=device, dtype=dtype or edge_weight.dtype
            )
        return EdgeIndexAdjacency(edge_index=edge_index, edge_weight=edge_weight)


def to_edge_adjacency(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> EdgeIndexAdjacency:
    return EdgeIndexAdjacency(edge_index=edge_index, edge_weight=edge_weight)


__all__ = ["EdgeIndexAdjacency", "to_edge_adjacency"]
