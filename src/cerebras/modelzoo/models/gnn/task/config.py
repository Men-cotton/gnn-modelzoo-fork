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
class GraphSAGEConfig:
    """Normalized GraphSAGE architecture configuration."""

    n_feat: int
    n_class: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    aggregator: Literal["mean", "sum", "max"] = "mean"
    core_architecture: Literal["GraphSAGE"] = "GraphSAGE"


class GNNModelConfig(GNNArchConfig):
    """Compatibility configuration for trainer-facing GNN model aliases."""

    name: Literal["gcn", "graphsage"] = "gcn"
    to_float16: bool = False
    disable_log_softmax: bool = False
    compute_eval_metrics: bool = True

    core_architecture: Literal["GCN", "GraphSAGE"] = "GCN"
    graphsage_hidden_dim: int = 128
    graphsage_num_layers: int = 2
    graphsage_dropout: Annotated[float, Ge(0), Le(1)] = 0.5
    graphsage_aggregator: Literal["mean", "sum", "max"] = "mean"

    @property
    def architecture_config(self) -> Union[GCNConfig, GraphSAGEConfig]:
        if self.core_architecture.lower() == "graphsage":
            return GraphSAGEConfig(
                n_feat=self.n_feat,
                n_class=self.n_class,
                hidden_dim=self.graphsage_hidden_dim,
                num_layers=self.graphsage_num_layers,
                dropout=self.graphsage_dropout,
                aggregator=self.graphsage_aggregator,
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
        from cerebras.modelzoo.models.gnn.model import GCNModel, GraphSAGEModel

        if self.name == "graphsage":
            return GraphSAGEModel
        return GCNModel


__all__ = ["GCNConfig", "GNNArchConfig", "GNNModelConfig", "GraphSAGEConfig"]
