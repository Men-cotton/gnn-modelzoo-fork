from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from annotated_types import Ge, Le
from cerebras.modelzoo.config import ModelConfig
from typing_extensions import Annotated


class GNNArchConfig(ModelConfig):
    """Base configuration for GNN architecture parameters."""

    n_feat: int
    n_class: int

    n_hid: int = 16
    dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.5
    activation_fn_hidden: Literal["relu", "none"] = "relu"
    activation_fn_output: Literal["relu", "none"] = "none"
    use_bias: bool = True


@dataclass(frozen=True)
class GCNConfig:
    """Normalized GCN architecture configuration."""

    n_feat: int
    n_class: int
    n_hid: int = 16
    dropout_rate: float = 0.5
    activation_fn_hidden: Literal["relu", "none"] = "relu"
    activation_fn_output: Literal["relu", "none"] = "none"
    use_bias: bool = True
    core_architecture: Literal["GCN"] = "GCN"


@dataclass(frozen=True)
class GATv2Config:
    """Normalized GATv2 architecture configuration."""

    n_feat: int
    n_class: int
    n_hid: int = 16
    num_heads: int = 8
    dropout_rate: float = 0.5
    activation_fn_hidden: Literal["relu", "none"] = "relu"
    activation_fn_output: Literal["relu", "none"] = "none"
    use_bias: bool = True
    core_architecture: Literal["GATv2"] = "GATv2"


@dataclass(frozen=True)
class GraphSAGEConfig:
    """Normalized GraphSAGE architecture configuration."""

    n_feat: int
    n_class: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    aggregator: Literal["mean", "sum", "max"] = "mean"
    core_architecture: Literal["GraphSAGE"] = "GraphSAGE"


@dataclass(frozen=True)
class GraphTransformerConfig:
    """Normalized GraphTransformer architecture configuration."""

    n_feat: int
    n_class: int
    hidden_dim: int = 128
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.5
    aggregator: Literal["mean", "sum", "max"] = "mean"
    core_architecture: Literal["GraphTransformer"] = "GraphTransformer"


class GNNModelConfig(GNNArchConfig):
    """Compatibility configuration for trainer-facing GNN model aliases."""

    name: Literal["gatv2", "gcn", "graphsage", "graph_transformer"] = "gcn"
    to_float16: bool = False
    disable_log_softmax: bool = False
    compute_eval_metrics: bool = True

    core_architecture: Literal["GATv2", "GCN", "GraphSAGE", "GraphTransformer"] = "GCN"
    gatv2_num_heads: int = 8
    graphsage_hidden_dim: int = 128
    graphsage_num_layers: int = 2
    graphsage_dropout: Annotated[float, Ge(0), Le(1)] = 0.5
    graphsage_aggregator: Literal["mean", "sum", "max"] = "mean"
    graph_transformer_hidden_dim: int = 128
    graph_transformer_num_layers: int = 2
    graph_transformer_heads: int = 4
    graph_transformer_dropout: Annotated[float, Ge(0), Le(1)] = 0.5

    @property
    def architecture_config(self) -> Union[
        GATv2Config,
        GCNConfig,
        GraphSAGEConfig,
        GraphTransformerConfig,
    ]:
        if self.core_architecture.lower() == "graphsage":
            return GraphSAGEConfig(
                n_feat=self.n_feat,
                n_class=self.n_class,
                hidden_dim=self.graphsage_hidden_dim,
                num_layers=self.graphsage_num_layers,
                dropout=self.graphsage_dropout,
                aggregator=self.graphsage_aggregator,
            )
        if self.core_architecture.lower() == "graphtransformer":
            return GraphTransformerConfig(
                n_feat=self.n_feat,
                n_class=self.n_class,
                hidden_dim=self.graph_transformer_hidden_dim,
                num_layers=self.graph_transformer_num_layers,
                heads=self.graph_transformer_heads,
                dropout=self.graph_transformer_dropout,
                aggregator=self.graphsage_aggregator,
            )
        if self.core_architecture.lower() == "gatv2":
            return GATv2Config(
                n_feat=self.n_feat,
                n_class=self.n_class,
                n_hid=self.n_hid,
                num_heads=self.gatv2_num_heads,
                dropout_rate=self.dropout_rate,
                activation_fn_hidden=self.activation_fn_hidden,
                activation_fn_output=self.activation_fn_output,
                use_bias=self.use_bias,
            )
        return GCNConfig(
            n_feat=self.n_feat,
            n_class=self.n_class,
            n_hid=self.n_hid,
            dropout_rate=self.dropout_rate,
            activation_fn_hidden=self.activation_fn_hidden,
            activation_fn_output=self.activation_fn_output,
            use_bias=self.use_bias,
        )

    @property
    def __model_cls__(self):
        from cerebras.modelzoo.models.gnn.model import (
            GATv2Model,
            GCNModel,
            GraphTransformerModel,
            GraphSAGEModel,
        )

        if self.name == "graphsage":
            return GraphSAGEModel
        if self.name == "graph_transformer":
            return GraphTransformerModel
        if self.name == "gatv2":
            return GATv2Model
        return GCNModel


__all__ = [
    "GATv2Config",
    "GCNConfig",
    "GNNArchConfig",
    "GNNModelConfig",
    "GraphSAGEConfig",
    "GraphTransformerConfig",
]
