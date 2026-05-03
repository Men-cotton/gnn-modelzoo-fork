from __future__ import annotations

import torch
import torch.nn as nn

from ..data_processing.batches import GraphSAGEBatch
from .graphsage import GraphSAGELayer


class GraphTransformerSageLayer(nn.Module):
    """
    Combines a local GraphSAGE message-passing step with global self-attention
    over the sampled node set.
    """

    def __init__(
        self,
        channels: int,
        graphsage_layer: nn.Module,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.graphsage = graphsage_layer
        self.attn = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, self_feats: torch.Tensor, aggregated_neighbors: torch.Tensor
    ) -> torch.Tensor:
        h_local = self.graphsage(self_feats, aggregated_neighbors)
        h = self.norm1(self_feats + self.dropout(h_local))

        h_attn, _ = self.attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        h = self.norm2(h + self.dropout(h_attn.squeeze(0)))

        h_ffn = self.ffn(h)
        return h + self.dropout(h_ffn)


class GraphTransformer(nn.Module):
    """GraphTransformer operating on the same neighbor-sampling batches as GraphSAGE."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        heads: int = 4,
        dropout: float = 0.5,
        aggregator: str = "mean",
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("GraphTransformer requires num_layers > 0.")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator.lower()
        self.num_classes = num_classes

        self.embedding = (
            nn.Linear(self.in_dim, self.hidden_dim)
            if self.in_dim != self.hidden_dim
            else nn.Identity()
        )

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            graphsage_layer = GraphSAGELayer(
                self_in_dim=self.hidden_dim,
                neighbor_in_dim=self.hidden_dim,
                out_dim=self.hidden_dim,
                dropout=0.0,
                apply_activation=False,
            )
            self.layers.append(
                GraphTransformerSageLayer(
                    channels=self.hidden_dim,
                    graphsage_layer=graphsage_layer,
                    heads=heads,
                    dropout=self.dropout,
                )
            )

        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        self.positional_encoding = None

    def _aggregate_neighbors(
        self, neighbor_feats: torch.Tensor, neighbor_mask: torch.Tensor
    ) -> torch.Tensor:
        if neighbor_mask.dim() != 2:
            raise ValueError(
                f"Neighbor mask must be rank-2; got shape {neighbor_mask.shape}."
            )
        num_nodes, fanout = neighbor_mask.shape
        if neighbor_feats.size(0) != num_nodes * fanout:
            raise ValueError(
                "Neighbor features do not align with mask. "
                f"Expected first dimension {num_nodes * fanout}, "
                f"got {neighbor_feats.size(0)}."
            )

        neighbor_feats = neighbor_feats.view(num_nodes, fanout, -1)
        mask_bool = neighbor_mask.bool()

        if self.aggregator in ("mean", "sum"):
            mask_float = mask_bool.unsqueeze(-1).to(neighbor_feats.dtype)
            summed = (neighbor_feats * mask_float).sum(dim=1)
            if self.aggregator == "mean":
                denom = mask_float.sum(dim=1).clamp_min(1.0)
                aggregated = summed / denom
            else:
                aggregated = summed
        elif self.aggregator == "max":
            fill_value = torch.finfo(neighbor_feats.dtype).min
            masked = neighbor_feats.masked_fill(~mask_bool.unsqueeze(-1), fill_value)
            aggregated = masked.max(dim=1).values
            valid = mask_bool.any(dim=1).unsqueeze(-1)
            aggregated = torch.where(valid, aggregated, 0.0)
        else:
            raise ValueError(
                f"Unsupported GraphTransformer aggregator '{self.aggregator}'. "
                "Choose from {'mean', 'sum', 'max'}."
            )
        return aggregated

    def forward(self, batch: GraphSAGEBatch) -> torch.Tensor:
        if len(batch.node_features) != len(batch.node_masks):
            raise ValueError(
                "GraphSAGEBatch node_features and node_masks length mismatch."
            )
        if len(batch.node_features) != self.num_layers + 1:
            raise ValueError(
                f"Expected {self.num_layers + 1} node feature tensors, "
                f"got {len(batch.node_features)}."
            )
        if len(batch.neighbor_masks) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} neighbor masks, got {len(batch.neighbor_masks)}."
            )

        flattened_node_features = []
        flattened_node_masks = []
        for feats, mask in zip(batch.node_features, batch.node_masks):
            flattened_node_features.append(feats.reshape(-1, feats.size(-1)))
            flattened_node_masks.append(mask.reshape(-1))

        flattened_neighbor_masks = [
            mask.reshape(-1, mask.size(-1)) for mask in batch.neighbor_masks
        ]

        states = [
            self.embedding(feats) * mask.view(-1, 1).to(dtype=feats.dtype)
            for feats, mask in zip(flattened_node_features, flattened_node_masks)
        ]

        if self.positional_encoding is not None:
            pass

        for layer_idx, layer in enumerate(self.layers):
            prev_states = states
            next_states = list(prev_states)
            max_depth = self.num_layers - layer_idx

            for depth in range(max_depth - 1, -1, -1):
                aggregated_neighbors = self._aggregate_neighbors(
                    prev_states[depth + 1], flattened_neighbor_masks[depth]
                )
                updated = layer(prev_states[depth], aggregated_neighbors)
                parent_mask = flattened_node_masks[depth].view(-1, 1).to(updated.dtype)
                next_states[depth] = updated * parent_mask

            states = next_states

        return self.classifier(states[0])


__all__ = ["GraphTransformer", "GraphTransformerSageLayer"]
