from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from cerebras.pytorch.metrics import AccuracyMetric

from ..architectures.registry import get_architecture_class
from .adapters import GNNBatch, adapt_gnn_batch
from .config import GATv2Config, GCNConfig, GNNModelConfig, GraphSAGEConfig

_ACTIVATION_FN_MAP: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "none": nn.Identity,
}


class GNNTaskWrapper(nn.Module):
    """Trainer-facing GNN wrapper for architecture adapters, loss, and metrics."""

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
        self.nll_loss_fn = nn.NLLLoss(ignore_index=-100)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.accuracy_metric = (
            AccuracyMetric(name="eval/masked_accuracy")
            if self.config.compute_eval_metrics
            else None
        )

    def build_model(self, model_config: GNNModelConfig) -> nn.Module:
        architecture_config = model_config.architecture_config
        if isinstance(architecture_config, GCNConfig):
            activation_hidden = _ACTIVATION_FN_MAP[
                architecture_config.activation_fn_hidden
            ]()
            activation_output = _ACTIVATION_FN_MAP[
                architecture_config.activation_fn_output
            ]()
            return get_architecture_class(architecture_config.core_architecture)(
                in_dim=architecture_config.n_feat,
                hidden_dim=architecture_config.n_hid,
                num_classes=architecture_config.n_class,
                dropout_rate=architecture_config.dropout_rate,
                activation_hidden=activation_hidden,
                activation_output=activation_output,
                use_bias=architecture_config.use_bias,
            )
        if isinstance(architecture_config, GATv2Config):
            activation_hidden = _ACTIVATION_FN_MAP[
                architecture_config.activation_fn_hidden
            ]()
            activation_output = _ACTIVATION_FN_MAP[
                architecture_config.activation_fn_output
            ]()
            return get_architecture_class(architecture_config.core_architecture)(
                in_dim=architecture_config.n_feat,
                hidden_dim=architecture_config.n_hid,
                num_classes=architecture_config.n_class,
                num_heads=architecture_config.num_heads,
                dropout_rate=architecture_config.dropout_rate,
                activation_hidden=activation_hidden,
                activation_output=activation_output,
                use_bias=architecture_config.use_bias,
            )
        if isinstance(architecture_config, GraphSAGEConfig):
            return get_architecture_class(architecture_config.core_architecture)(
                input_dim=architecture_config.n_feat,
                hidden_dim=architecture_config.hidden_dim,
                num_layers=architecture_config.num_layers,
                dropout=architecture_config.dropout,
                aggregator=architecture_config.aggregator,
                num_classes=architecture_config.n_class,
            )
        raise ValueError(
            f"Unsupported core architecture '{model_config.core_architecture}'."
        )

    def forward(self, batch: GNNBatch) -> torch.Tensor:
        param = next(self.parameters())
        adapted = adapt_gnn_batch(
            batch,
            architecture=self.config.core_architecture,
            device=param.device,
            model_dtype=param.dtype,
        )
        logits = self.model(*adapted.model_args)
        if (
            self.config.core_architecture.lower() == "gcn"
            and logits.dtype != torch.float32
        ):
            logits = logits.to(torch.float32)

        labels_long = adapted.labels.to(torch.long)
        mask = adapted.target_mask.to(torch.bool)
        ignore_filled = torch.full_like(labels_long, self.nll_loss_fn.ignore_index)
        labels_with_ignore = torch.where(mask, labels_long, ignore_filled)
        if not self.config.disable_log_softmax:
            log_probs = F.log_softmax(logits, dim=1)
            loss = self.nll_loss_fn(log_probs, labels_with_ignore)
        else:
            log_probs = logits
            loss = self.ce_loss_fn(logits, labels_with_ignore)

        if not self.training and self.accuracy_metric is not None:
            predictions = log_probs.argmax(dim=-1).to(labels_long.dtype).detach()
            weights = mask.to(log_probs.dtype)
            self.accuracy_metric(
                labels=labels_long.clone().detach(),
                predictions=predictions,
                weights=weights,
            )

        return loss


__all__ = ["GNNTaskWrapper"]
