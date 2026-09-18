from __future__ import annotations

import torch
import torch.nn as nn
from cerebras.pytorch.metrics import AccuracyMetric

from ..architectures.spec import ArchitectureSpec
from .adapters import GNNBatch
from .config import GNNModelConfig
from .loss import masked_classification_loss


class GNNTaskWrapper(nn.Module):
    """Trainer-facing GNN wrapper for architecture adapters, loss, and metrics."""

    def __init__(self, config: GNNModelConfig):
        super().__init__()
        # The registry imports task adapters while registering architectures.
        from ..architectures.registry import get_architecture_spec_for_config

        if isinstance(config, dict):
            model_dict = config.get("model", config)
            if not isinstance(model_dict, dict):
                raise TypeError("Expected model configuration dictionary.")
            self.config = GNNModelConfig(**model_dict)
        else:
            self.config = config

        self.architecture_config = self.config.architecture_config
        self.architecture_spec = get_architecture_spec_for_config(
            self.architecture_config
        )
        self.model = self.build_model(self.architecture_spec, self.architecture_config)
        self.accuracy_metric = (
            AccuracyMetric(name="eval/masked_accuracy")
            if self.config.compute_eval_metrics
            else None
        )

    def build_model(
        self,
        architecture_spec: ArchitectureSpec,
        architecture_config,
    ) -> nn.Module:
        return architecture_spec.build_model(architecture_config)

    def forward(self, batch: GNNBatch) -> torch.Tensor:
        param = next(self.parameters())
        adapted = self.architecture_spec.adapt_batch(
            batch,
            param.device,
            param.dtype,
            self.architecture_spec.name,
        )
        logits = self.model(*adapted.model_args)
        logits = self.architecture_spec.postprocess_logits(logits)

        labels_long = adapted.labels.to(torch.long)
        mask = adapted.target_mask.to(torch.bool) & (labels_long != -100)
        loss = masked_classification_loss(
            logits,
            labels_long,
            mask,
            disable_log_softmax=self.config.disable_log_softmax,
        )

        if not self.training and self.accuracy_metric is not None:
            predictions = logits.argmax(dim=-1).to(labels_long.dtype).detach()
            weights = mask.to(torch.float32)
            self.accuracy_metric(
                labels=labels_long.clone().detach(),
                predictions=predictions,
                weights=weights,
            )

        return loss


__all__ = ["GNNTaskWrapper"]
