from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping

import torch
from torch import Tensor

from cerebras.pytorch.utils.nest import register_visit_fn


@dataclass
class GraphSAGEBatch:
    """Container for GraphSAGE mini-batches with static shapes."""

    node_features: List[Tensor]
    node_masks: List[Tensor]
    neighbor_masks: List[Tensor]
    labels: Tensor
    target_mask: Tensor

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GraphSAGEBatch":
        return cls(
            node_features=list(payload["node_features"]),
            node_masks=list(payload["node_masks"]),
            neighbor_masks=list(payload["neighbor_masks"]),
            labels=payload["labels"],
            target_mask=payload["target_mask"],
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> "GraphSAGEBatch":
        return GraphSAGEBatch(
            node_features=[
                feat.to(device, non_blocking=non_blocking) for feat in self.node_features
            ],
            node_masks=[
                mask.to(device, non_blocking=non_blocking) for mask in self.node_masks
            ],
            neighbor_masks=[
                mask.to(device, non_blocking=non_blocking) for mask in self.neighbor_masks
            ],
            labels=self.labels.to(device, non_blocking=non_blocking),
            target_mask=self.target_mask.to(device, non_blocking=non_blocking),
        )


@register_visit_fn(GraphSAGEBatch)
def _visit_graphsage_batch(batch: "GraphSAGEBatch"):
    # Ensure Cerebras dataloader utilities see all embedded tensors so they can
    # infer batch sizes and move data across devices correctly.
    for idx, tensor in enumerate(batch.node_features):
        yield ["node_features", str(idx)], tensor
    for idx, tensor in enumerate(batch.node_masks):
        yield ["node_masks", str(idx)], tensor
    for idx, tensor in enumerate(batch.neighbor_masks):
        yield ["neighbor_masks", str(idx)], tensor
    yield ["labels"], batch.labels
    yield ["target_mask"], batch.target_mask


__all__ = ["GraphSAGEBatch"]
