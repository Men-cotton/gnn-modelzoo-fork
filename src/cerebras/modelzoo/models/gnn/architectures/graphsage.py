from __future__ import annotations

import torch
import torch.nn as nn

from ..batches import GraphSAGEBatch


class GraphSAGELayer(nn.Module):
    """Single GraphSAGE aggregation layer."""

    def __init__(
        self,
        self_in_dim: int,
        neighbor_in_dim: int,
        out_dim: int,
        dropout: float,
        apply_activation: bool,
    ):
        super().__init__()
        self.self_linear = nn.Linear(self_in_dim, out_dim)
        self.neighbor_linear = nn.Linear(neighbor_in_dim, out_dim)
        self.activation = nn.ReLU() if apply_activation else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self, self_feats: torch.Tensor, neighbor_feats: torch.Tensor
    ) -> torch.Tensor:
        combined = self.self_linear(self_feats) + self.neighbor_linear(neighbor_feats)
        combined = self.activation(combined)
        return self.dropout(combined)


class GraphSAGE(nn.Module):
    """GraphSAGE implementation operating on GraphSAGEBatch inputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        aggregator: str,
        num_classes: int,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("GraphSAGE requires num_layers > 0.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator.lower()
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            is_first_layer = layer_idx == 0
            is_last_layer = layer_idx == self.num_layers - 1
            self_in_dim = self.input_dim if is_first_layer else self.hidden_dim
            neighbor_in_dim = self.input_dim if is_first_layer else self.hidden_dim
            apply_activation = not is_last_layer
            dropout_p = self.dropout if apply_activation else 0.0
            self.layers.append(
                GraphSAGELayer(
                    self_in_dim=self_in_dim,
                    neighbor_in_dim=neighbor_in_dim,
                    out_dim=self.hidden_dim,
                    dropout=dropout_p,
                    apply_activation=apply_activation,
                )
            )

        self.final_dropout = (
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        )
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
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
            else:  # sum
                aggregated = summed
        elif self.aggregator == "max":
            fill_value = torch.finfo(neighbor_feats.dtype).min
            masked = neighbor_feats.masked_fill(~mask_bool.unsqueeze(-1), fill_value)
            aggregated = masked.max(dim=1).values
            valid = mask_bool.any(dim=1).unsqueeze(-1)
            aggregated = torch.where(valid, aggregated, 0.0)
        else:
            raise ValueError(
                f"Unsupported GraphSAGE aggregator '{self.aggregator}'. "
                "Choose from {'mean', 'sum', 'max'}."
            )
        return aggregated

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
            if feats.dim() == 3:
                flat_feats = feats.reshape(-1, feats.size(-1))
            else:
                flat_feats = feats.reshape(-1, feats.size(-1))

            if mask.dim() > 1:
                flat_mask = mask.reshape(-1)
            else:
                flat_mask = mask.view(-1)

            flattened_node_features.append(flat_feats)
            flattened_node_masks.append(flat_mask)

        flattened_neighbor_masks = []
        for mask in batch.neighbor_masks:
            if mask.dim() == 3:
                flattened_neighbor_masks.append(mask.reshape(-1, mask.size(-1)))
            else:
                flattened_neighbor_masks.append(mask)

        states = [
            feats * mask.view(-1, 1).to(dtype=feats.dtype)
            for feats, mask in zip(flattened_node_features, flattened_node_masks)
        ]

        for layer_idx, layer in enumerate(self.layers):
            prev_states = states
            next_states = list(prev_states)
            max_depth = self.num_layers - layer_idx

            # Update nodes from deepest to shallowest for this layer
            for depth in range(max_depth - 1, -1, -1):
                self_feats = prev_states[depth]
                neighbor_feats = prev_states[depth + 1]
                aggregated_neighbors = self._aggregate_neighbors(
                    neighbor_feats, flattened_neighbor_masks[depth]
                )
                updated = layer(self_feats, aggregated_neighbors)
                parent_mask = flattened_node_masks[depth].view(-1, 1).to(updated.dtype)
                next_states[depth] = updated * parent_mask

            states = next_states

        root_embeddings = states[0]
        root_embeddings = self.final_dropout(root_embeddings)
        logits = self.classifier(root_embeddings)
        return logits


__all__ = ["GraphSAGE", "GraphSAGELayer"]
