from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import cerebras.pytorch as cstorch

from ..data_processing.runtime.torch import EdgeIndexAdjacency
from .ops import segment_softmax

try:
    from torch.cuda.amp import autocast as cuda_autocast
except ImportError:
    cuda_autocast = None


class GATv2Layer(nn.Module):
    """A single Graph Attention Network v2 layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        activation: nn.Module = nn.Identity(),
        use_bias: bool = True,
        concat: bool = True,
        dropout_rate: float = 0.0,
        share_weights: bool = False,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.share_weights = share_weights
        self.negative_slope = negative_slope

        self.weight_l = nn.Parameter(torch.empty(in_features, heads * out_features))
        if share_weights:
            self.weight_r = self.weight_l
        else:
            self.weight_r = nn.Parameter(torch.empty(in_features, heads * out_features))

        self.att = nn.Parameter(torch.empty(1, heads, out_features))

        if use_bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * out_features))
        elif use_bias and not concat:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_l)
        if not self.share_weights:
            nn.init.xavier_uniform_(self.weight_r)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: Union[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
            Dict[str, torch.Tensor],
            EdgeIndexAdjacency,
        ],
    ) -> torch.Tensor:
        output_dtype = features.dtype
        if features.dim() == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        num_nodes = features.size(0)
        edge_index = self._prepare_edge_index(adjacency, features.device)

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        h_l = torch.matmul(features, self.weight_l).view(
            -1, self.heads, self.out_features
        )
        h_r = torch.matmul(features, self.weight_r).view(
            -1, self.heads, self.out_features
        )

        x_i = h_l.index_select(0, target_nodes)
        x_j = h_r.index_select(0, source_nodes)

        attention_input = F.leaky_relu(
            x_i + x_j,
            negative_slope=self.negative_slope,
        )
        scores = (attention_input * self.att).sum(dim=-1)

        alpha = segment_softmax(scores, target_nodes, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        messages = x_j * alpha.unsqueeze(-1)

        out = features.new_zeros((num_nodes, self.heads, self.out_features))
        out.index_add_(0, target_nodes, messages)

        if self.concat:
            out = out.reshape(-1, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        out = self.activation(out)
        if out.dtype != output_dtype:
            out = out.to(output_dtype)
        return out

    def _prepare_edge_index(
        self,
        adjacency: Union[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
            Dict[str, torch.Tensor],
            EdgeIndexAdjacency,
        ],
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(adjacency, EdgeIndexAdjacency):
            edge_index = adjacency.edge_index
        elif isinstance(adjacency, dict):
            edge_index = adjacency.get("edge_index")
            if edge_index is None:
                raise KeyError("Expected 'edge_index' in adjacency dictionary.")
        elif isinstance(adjacency, (tuple, list)):
            if len(adjacency) != 2:
                raise ValueError(
                    "Adjacency tuple must contain (edge_index, edge_weight)."
                )
            edge_index = adjacency[0]
        elif isinstance(adjacency, torch.Tensor):
            if adjacency.dim() == 3 and adjacency.size(0) == 1:
                adjacency = adjacency.squeeze(0)
            if adjacency.is_sparse:
                edge_index = adjacency.coalesce().indices()
            else:
                edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
        else:
            raise TypeError(
                "Unsupported adjacency type. Expected tuple, dict, EdgeIndexAdjacency, "
                "or torch.Tensor."
            )

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E]; received {tuple(edge_index.shape)}"
            )
        index_dtype = torch.int32 if cstorch.use_cs() else torch.long
        return edge_index.to(device=device, dtype=index_dtype)


class GATv2(nn.Module):
    """Standard two-layer Graph Attention Network v2."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
        dropout_rate: float,
        activation_hidden: nn.Module,
        activation_output: nn.Module,
        use_bias: bool,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("GATv2 requires num_heads > 0.")
        self.gat1 = GATv2Layer(
            in_features=in_dim,
            out_features=hidden_dim,
            heads=num_heads,
            activation=activation_hidden,
            use_bias=use_bias,
            concat=True,
            dropout_rate=dropout_rate,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.gat2 = GATv2Layer(
            in_features=hidden_dim * num_heads,
            out_features=num_classes,
            heads=1,
            activation=activation_output,
            use_bias=use_bias,
            concat=False,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        features: torch.Tensor,
        adjacency: Union[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
            Dict[str, torch.Tensor],
            EdgeIndexAdjacency,
        ],
    ) -> torch.Tensor:
        ctx = (
            cuda_autocast(enabled=False)
            if cuda_autocast is not None and torch.cuda.is_available()
            else nullcontext()
        )
        with ctx:
            hidden = self.gat1(features, adjacency)
            hidden = self.dropout(hidden)
            logits = self.gat2(hidden, adjacency)
        return logits


__all__ = ["GATv2", "GATv2Layer"]
