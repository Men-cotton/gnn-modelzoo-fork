from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple, Union

import torch
from torch import Tensor

from cerebras.pytorch.utils.nest import register_visit_fn

AdjacencyPayload = Union[
    Tensor,
    Tuple[Tensor, Tensor],
    Dict[str, Tensor],
]


@dataclass
class FullGraphBatch:
    """Container for full-graph batches with one graph per sample."""

    features: Tensor
    adjacency: AdjacencyPayload
    labels: Tensor
    target_mask: Tensor

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FullGraphBatch":
        return cls(
            features=payload["features"],
            adjacency=payload["adjacency"],
            labels=payload["labels"],
            target_mask=payload["target_mask"],
        )

    @classmethod
    def from_legacy_tuple(cls, payload) -> "FullGraphBatch":
        (features, adjacency), (labels, mask) = payload
        return cls(
            features=features,
            adjacency=adjacency,
            labels=labels,
            target_mask=mask,
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> "FullGraphBatch":
        adjacency = self.adjacency
        if hasattr(adjacency, "edge_index") and hasattr(adjacency, "edge_weight"):
            adjacency = adjacency.to(device=device)
        elif torch.is_tensor(adjacency):
            adjacency = adjacency.to(device, non_blocking=non_blocking)
        elif isinstance(adjacency, tuple):
            adjacency = tuple(
                tensor.to(device, non_blocking=non_blocking) for tensor in adjacency
            )
        elif isinstance(adjacency, dict):
            adjacency = {
                key: tensor.to(device, non_blocking=non_blocking)
                for key, tensor in adjacency.items()
            }

        return FullGraphBatch(
            features=self.features.to(device, non_blocking=non_blocking),
            adjacency=adjacency,
            labels=self.labels.to(device, non_blocking=non_blocking),
            target_mask=self.target_mask.to(device, non_blocking=non_blocking),
        )


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
                feat.to(device, non_blocking=non_blocking)
                for feat in self.node_features
            ],
            node_masks=[
                mask.to(device, non_blocking=non_blocking) for mask in self.node_masks
            ],
            neighbor_masks=[
                mask.to(device, non_blocking=non_blocking)
                for mask in self.neighbor_masks
            ],
            labels=self.labels.to(device, non_blocking=non_blocking),
            target_mask=self.target_mask.to(device, non_blocking=non_blocking),
        )


@register_visit_fn(FullGraphBatch)
def _visit_full_graph_batch(batch: "FullGraphBatch"):
    yield ["features"], batch.features
    if hasattr(batch.adjacency, "edge_index") and hasattr(
        batch.adjacency, "edge_weight"
    ):
        yield ["adjacency", "edge_index"], batch.adjacency.edge_index
        yield ["adjacency", "edge_weight"], batch.adjacency.edge_weight
    elif torch.is_tensor(batch.adjacency):
        yield ["adjacency"], batch.adjacency
    elif isinstance(batch.adjacency, tuple):
        for idx, tensor in enumerate(batch.adjacency):
            yield ["adjacency", str(idx)], tensor
    elif isinstance(batch.adjacency, dict):
        for key, tensor in batch.adjacency.items():
            yield ["adjacency", str(key)], tensor
    yield ["labels"], batch.labels
    yield ["target_mask"], batch.target_mask


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


__all__ = ["AdjacencyPayload", "FullGraphBatch", "GraphSAGEBatch"]
