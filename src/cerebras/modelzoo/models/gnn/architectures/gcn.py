from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple, Union

import cerebras.pytorch as cstorch
import torch
import torch.nn as nn

from ..pipelines.common import EdgeIndexAdjacency

try:
    from torch.cuda.amp import autocast as cuda_autocast
except ImportError:
    cuda_autocast = None


class GCNLayer(nn.Module):
    """A single Graph Convolutional Network layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        use_bias: bool,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
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
        """
        Args:
            features: Node features of shape [num_nodes, in_features].
            adjacency: One of
                * Tuple(edge_index, edge_weight)
                * Dict with keys 'edge_index' and optional 'edge_weight'
                * Dense adjacency matrix (torch.Tensor)
                * Sparse COO tensor (torch.Tensor)
        """
        output_dtype = features.dtype
        support = torch.matmul(features, self.weight)

        if cstorch.use_cs():
            adjacency_dense = self._prepare_dense_adjacency(
                adjacency,
                device=support.device,
                dtype=support.dtype,
                num_nodes=support.size(0),
            )
        else:
            edge_index, edge_weight = self._prepare_edge_tensors(
                adjacency,
                device=support.device,
                dtype=support.dtype,
            )

        ctx = (
            cuda_autocast(enabled=False)
            if cuda_autocast is not None and torch.cuda.is_available()
            else nullcontext()
        )
        with ctx:
            if cstorch.use_cs():
                output = self._propagate_dense(support, adjacency_dense)
            else:
                output = self._propagate(support, edge_index, edge_weight)

        if self.bias is not None:
            output = output + self.bias
        output = self.activation(output)
        if output.dtype != output_dtype:
            output = output.to(output_dtype)
        return output

    def _prepare_edge_tensors(
        self,
        adjacency: Union[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
            Dict[str, torch.Tensor],
            EdgeIndexAdjacency,
        ],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(adjacency, EdgeIndexAdjacency):
            edge_index = adjacency.edge_index
            edge_weight = adjacency.edge_weight
        elif isinstance(adjacency, dict):
            edge_index = adjacency.get("edge_index")
            if edge_index is None:
                raise KeyError("Expected 'edge_index' in adjacency dictionary.")
            edge_weight = adjacency.get("edge_weight")
        elif isinstance(adjacency, (tuple, list)):
            if len(adjacency) != 2:
                raise ValueError(
                    "Adjacency tuple must contain (edge_index, edge_weight)."
                )
            edge_index, edge_weight = adjacency
        elif isinstance(adjacency, torch.Tensor):
            if adjacency.is_sparse:
                coalesced = adjacency.coalesce()
                edge_index = coalesced.indices()
                edge_weight = coalesced.values()
            else:
                indices = adjacency.nonzero(as_tuple=False)
                edge_index = indices.t().contiguous()
                edge_weight = adjacency[indices[:, 0], indices[:, 1]]
        else:
            raise TypeError(
                "Unsupported adjacency type. Expected tuple, dict, or torch.Tensor."
            )

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E]; received {tuple(edge_index.shape)}"
            )

        index_dtype = torch.int32 if cstorch.use_cs() else torch.long
        edge_index = edge_index.to(device=device, dtype=index_dtype)
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1),
                device=device,
                dtype=dtype,
            )
        else:
            edge_weight = edge_weight.to(device=device, dtype=dtype)

        if edge_index.size(1) == 0:
            empty_index = torch.empty(
                (2, 0), device=device, dtype=index_dtype
            )
            empty_weight = torch.empty(
                (0,), device=device, dtype=dtype
            )
            return empty_index, empty_weight

        return edge_index, edge_weight

    def _prepare_dense_adjacency(
        self,
        adjacency: Union[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
            Dict[str, torch.Tensor],
            EdgeIndexAdjacency,
        ],
        device: torch.device,
        dtype: torch.dtype,
        num_nodes: int,
    ) -> torch.Tensor:
        if not isinstance(adjacency, torch.Tensor):
            raise TypeError(
                "Dense adjacency tensor is required when running on Cerebras hardware."
            )
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        if adjacency.dim() != 3 or adjacency.size(0) != 1:
            raise ValueError(
                "Dense adjacency must have shape [1, N, N] when running on Cerebras hardware."
            )
        if adjacency.size(1) != adjacency.size(2):
            raise ValueError(
                f"Dense adjacency must be square; received shape {tuple(adjacency.shape)}."
            )
        if adjacency.size(1) != num_nodes:
            raise ValueError(
                "Dense adjacency dimension mismatch with feature tensor."
            )
        return adjacency.to(device=device, dtype=dtype)

    def _propagate(
        self,
        support: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = support.size(0)
        out = support.new_zeros(num_nodes, support.size(1))
        # Use index_select to retrieve row/col without direct slicing (unsupported on int tensors on WSE).
        row_selector = edge_index.new_tensor([0])
        col_selector = edge_index.new_tensor([1])
        row = torch.index_select(edge_index, 0, row_selector).squeeze(0)
        col = torch.index_select(edge_index, 0, col_selector).squeeze(0)
        device = support.device
        row = row.to(device=device)
        col = col.to(device=device)
        messages = support.index_select(0, col)
        messages = messages * edge_weight.unsqueeze(-1)
        out.index_add_(0, row, messages)
        return out

    def _propagate_dense(
        self,
        support: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        support_batched = support.unsqueeze(0)
        propagated = torch.matmul(adjacency, support_batched)
        return propagated.squeeze(0)


class GCN(nn.Module):
    """Standard two-layer Graph Convolutional Network."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout_rate: float,
        activation_hidden: nn.Module,
        activation_output: nn.Module,
        use_bias: bool,
    ):
        super().__init__()
        self.gc1 = GCNLayer(
            in_features=in_dim,
            out_features=hidden_dim,
            activation=activation_hidden,
            use_bias=use_bias,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.gc2 = GCNLayer(
            in_features=hidden_dim,
            out_features=num_classes,
            activation=activation_output,
            use_bias=use_bias,
        )

    def forward(
        self,
        features: torch.Tensor,
        adjacency: Union[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
            Dict[str, torch.Tensor],
        ],
    ) -> torch.Tensor:
        ctx = (
            cuda_autocast(enabled=False)
            if cuda_autocast is not None and torch.cuda.is_available()
            else nullcontext()
        )
        with ctx:
            hidden = self.gc1(features, adjacency)
            hidden = self.dropout(hidden)
            logits = self.gc2(hidden, adjacency)
        return logits


__all__ = ["GCN", "GCNLayer"]
