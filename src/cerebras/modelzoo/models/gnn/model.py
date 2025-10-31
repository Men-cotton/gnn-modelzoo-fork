from __future__ import annotations

import logging
from typing import Dict, Literal, Tuple, Type, Union

import cerebras.pytorch as cstorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from annotated_types import Ge, Le
from cerebras.modelzoo.config import ModelConfig
from cerebras.pytorch.metrics import AccuracyMetric
from typing_extensions import Annotated

from .architectures import GCN, GraphSAGE
from .batches import GraphSAGEBatch
from .pipelines.common import EdgeIndexAdjacency

logger = logging.getLogger(__name__)


_ACTIVATION_FN_MAP: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "none": nn.Identity,
}

AdjacencyPayload = Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Dict[str, torch.Tensor],
    EdgeIndexAdjacency,
]


class GNNArchConfig(ModelConfig):
    """Base configuration for GNN architecture parameters."""

    n_feat: int
    n_class: int

    n_hid: int = 16
    dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.5
    activation_fn_hidden: Literal["relu", "none"] = "relu"
    activation_fn_output: Literal["relu", "none"] = "none"
    use_bias: bool = True


class GNNModelConfig(GNNArchConfig):
    """Configuration for the GNNModel wrapper."""

    name: Literal["gnn", "graphsage"] = "gnn"
    to_float16: bool = False
    disable_log_softmax: bool = False
    compute_eval_metrics: bool = True

    core_architecture: Literal["GCN", "GraphSAGE"] = "GCN"
    graphsage_hidden_dim: int = 128
    graphsage_num_layers: int = 2
    graphsage_dropout: Annotated[float, Ge(0), Le(1)] = 0.5
    graphsage_aggregator: Literal["mean", "sum", "max"] = "mean"

    @property
    def __model_cls__(self):
        return GNNModel


class GNNModel(nn.Module):
    """Top-level model wrapper that selects the desired GNN architecture."""

    def __init__(self, config: GNNModelConfig):
        super().__init__()
        if isinstance(config, dict):
            model_dict = config.get("model", config)
            if not isinstance(model_dict, dict):
                raise TypeError("Expected model configuration dictionary.")
            self.config = GNNModelConfig(**model_dict)
        else:
            self.config = config

        self.model = self.build_model(self.config)
        self.loss_fn = nn.NLLLoss(ignore_index=-100)
        self.accuracy_metric = (
            AccuracyMetric(name="eval/masked_accuracy")
            if self.config.compute_eval_metrics
            else None
        )

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------
    def build_model(self, model_config: GNNModelConfig) -> nn.Module:
        architecture = model_config.core_architecture.lower()
        if architecture == "gcn":
            activation_hidden = _ACTIVATION_FN_MAP[model_config.activation_fn_hidden]()
            activation_output = _ACTIVATION_FN_MAP[model_config.activation_fn_output]()
            core = GCN(
                in_dim=model_config.n_feat,
                hidden_dim=model_config.n_hid,
                num_classes=model_config.n_class,
                dropout_rate=model_config.dropout_rate,
                activation_hidden=activation_hidden,
                activation_output=activation_output,
                use_bias=model_config.use_bias,
            )
        elif architecture == "graphsage":
            core = GraphSAGE(
                input_dim=model_config.n_feat,
                hidden_dim=model_config.graphsage_hidden_dim,
                num_layers=model_config.graphsage_num_layers,
                dropout=model_config.graphsage_dropout,
                aggregator=model_config.graphsage_aggregator,
                num_classes=model_config.n_class,
            )
        else:
            raise ValueError(
                f"Unsupported core architecture '{model_config.core_architecture}'."
            )

        return core

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Union[
            GraphSAGEBatch,
            Tuple[
                Tuple[torch.Tensor, AdjacencyPayload],
                Tuple[torch.Tensor, torch.Tensor],
            ],
        ],
    ) -> torch.Tensor:
        if isinstance(batch, dict):
            batch = GraphSAGEBatch.from_payload(batch)

        if isinstance(batch, GraphSAGEBatch):
            param = next(self.parameters())
            device = param.device
            model_dtype = param.dtype
            batch = batch.to(device)
            if any(feat.dtype != model_dtype for feat in batch.node_features):
                # Align GraphSAGE feature tensors with the model dtype (e.g., float16).
                batch = GraphSAGEBatch(
                    node_features=[
                        feat.to(model_dtype) for feat in batch.node_features
                    ],
                    node_masks=batch.node_masks,
                    neighbor_masks=batch.neighbor_masks,
                    labels=batch.labels,
                    target_mask=batch.target_mask,
                )
            logits = self.model(batch)
            labels = batch.labels
            mask = batch.target_mask.to(torch.bool)
        else:
            (features, adjacency), (labels, mask) = batch
            if self.config.core_architecture.lower() == "gcn":
                if cstorch.use_cs() and features.dim() == 3 and features.size(0) == 1:
                    features_for_model = features.squeeze(0)
                else:
                    features_for_model = features
                if features_for_model.dtype != torch.float32:
                    features_for_model = features_for_model.to(torch.float32)
            else:
                features_for_model = features
            logits = self.model(features_for_model, adjacency)
            if self.config.core_architecture.lower() == "gcn":
                if logits.dtype != torch.float32:
                    logits = logits.to(torch.float32)
            if mask.dim() == 2 and mask.size(0) == 1:
                mask = mask.squeeze(0)
            mask = mask.to(torch.bool)
            if labels.dim() == 2 and labels.size(0) == 1:
                labels = labels.squeeze(0)
            labels = labels.to(torch.long)

        if not self.config.disable_log_softmax:
            log_probs = F.log_softmax(logits, dim=1)
        else:
            log_probs = logits

        labels_long = labels.to(torch.long)
        ignore_filled = torch.full_like(
            labels_long, self.loss_fn.ignore_index
        )
        labels_with_ignore = torch.where(mask, labels_long, ignore_filled)
        loss = self.loss_fn(log_probs, labels_with_ignore)

        if (
            not self.training
            and self.accuracy_metric is not None
        ):
            predictions = log_probs.argmax(dim=-1).to(labels_long.dtype).detach()
            weights = mask.to(log_probs.dtype)
            self.accuracy_metric(
                labels=labels_long.clone().detach(),
                predictions=predictions,
                weights=weights,
            )

        return loss


class GraphSAGEModel(GNNModel):
    """Alias model registered separately for GraphSAGE experiments."""

    pass


__all__ = ["GNNModel", "GNNModelConfig", "GraphSAGEModel", "GraphSAGEBatch"]
