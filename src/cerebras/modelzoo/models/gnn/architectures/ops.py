from __future__ import annotations

import torch


def segment_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Compute softmax over rows grouped by segment id."""
    if src.dim() != 2:
        raise ValueError(f"Expected src to have shape [E, H], got {src.shape}.")
    if index.dim() != 1 or index.size(0) != src.size(0):
        raise ValueError("index must be a rank-1 tensor aligned with src.")

    max_per_segment = src.new_full((num_segments, src.size(1)), float("-inf"))
    max_per_segment.scatter_reduce_(
        0,
        index.unsqueeze(-1).expand_as(src),
        src,
        reduce="amax",
        include_self=False,
    )
    src = src - max_per_segment.index_select(0, index)

    exp_src = torch.exp(src)
    sum_per_segment = src.new_zeros((num_segments, src.size(1)))
    sum_per_segment.index_add_(0, index, exp_src)
    return exp_src / sum_per_segment.index_select(0, index).clamp_min(1e-16)


__all__ = ["segment_softmax"]
