from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple, Union

import cerebras.pytorch as cstorch
import torch

from ..data_processing.batches import AdjacencyPayload, FullGraphBatch, GraphSAGEBatch

LegacyFullGraphPayload = Tuple[
    Tuple[torch.Tensor, AdjacencyPayload],
    Tuple[torch.Tensor, torch.Tensor],
]
GNNBatch = Union[
    FullGraphBatch,
    GraphSAGEBatch,
    LegacyFullGraphPayload,
    Mapping[str, Any],
]


@dataclass(frozen=True)
class AdaptedBatch:
    model_args: Tuple[Any, ...]
    labels: torch.Tensor
    target_mask: torch.Tensor


def coerce_gnn_batch(batch: GNNBatch) -> Union[FullGraphBatch, GraphSAGEBatch]:
    if isinstance(batch, (FullGraphBatch, GraphSAGEBatch)):
        return batch
    if isinstance(batch, Mapping):
        if "features" in batch and "adjacency" in batch:
            return FullGraphBatch.from_payload(batch)
        return GraphSAGEBatch.from_payload(batch)
    return FullGraphBatch.from_legacy_tuple(batch)


def adapt_graphsage_batch(
    batch: GraphSAGEBatch,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
) -> AdaptedBatch:
    batch = batch.to(device)
    if any(feat.dtype != model_dtype for feat in batch.node_features):
        batch = GraphSAGEBatch(
            node_features=[feat.to(model_dtype) for feat in batch.node_features],
            node_masks=batch.node_masks,
            neighbor_masks=batch.neighbor_masks,
            labels=batch.labels,
            target_mask=batch.target_mask,
        )
    return AdaptedBatch(
        model_args=(batch,),
        labels=batch.labels,
        target_mask=batch.target_mask.to(torch.bool),
    )


def adapt_full_graph_batch(
    batch: FullGraphBatch,
    *,
    architecture: str,
    device: torch.device,
) -> AdaptedBatch:
    batch = batch.to(device)
    features = batch.features
    labels = batch.labels
    mask = batch.target_mask
    if architecture.lower() == "gcn":
        if cstorch.use_cs() and features.dim() == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        if features.dtype != torch.float32:
            features = features.to(torch.float32)

    if mask.dim() == 2 and mask.size(0) == 1:
        mask = mask.squeeze(0)
    if labels.dim() == 2 and labels.size(0) == 1:
        labels = labels.squeeze(0)

    return AdaptedBatch(
        model_args=(features, batch.adjacency),
        labels=labels.to(torch.long),
        target_mask=mask.to(torch.bool),
    )


def adapt_gnn_batch(
    batch: GNNBatch,
    *,
    architecture: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> AdaptedBatch:
    batch = coerce_gnn_batch(batch)
    if isinstance(batch, GraphSAGEBatch):
        return adapt_graphsage_batch(
            batch,
            device=device,
            model_dtype=model_dtype,
        )
    if architecture.lower() in {"graphsage", "graphtransformer"}:
        raise ValueError(
            f"{architecture} requires neighbor-sampling GraphSAGEBatch inputs."
        )
    return adapt_full_graph_batch(
        batch,
        architecture=architecture,
        device=device,
    )


__all__ = [
    "AdaptedBatch",
    "GNNBatch",
    "LegacyFullGraphPayload",
    "adapt_gnn_batch",
    "coerce_gnn_batch",
]
