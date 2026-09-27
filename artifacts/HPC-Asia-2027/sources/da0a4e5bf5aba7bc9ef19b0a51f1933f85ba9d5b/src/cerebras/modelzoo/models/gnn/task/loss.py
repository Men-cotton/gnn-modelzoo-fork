from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    disable_log_softmax: bool = False,
) -> torch.Tensor:
    """Average node losses over supervised targets, including padded batches.

    Keep the FP32 numerator and target count explicit: SDK mean-loss lowering
    with ignored labels can divide by the fixed number of slots instead.
    """
    mask = target_mask.to(torch.bool) & (labels != -100)
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels)).to(torch.long)
    logits = logits.float()
    if disable_log_softmax:
        losses = F.cross_entropy(logits, safe_labels, reduction="none")
    else:
        losses = F.nll_loss(F.log_softmax(logits, dim=1), safe_labels, reduction="none")
    weights = mask.to(torch.float32)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)
