from __future__ import annotations

import logging

from cerebras.modelzoo.trainer.callbacks import register_global_callback

from .callbacks.compute_time import ComputeTimeCallback
from .data_processing.batches import FullGraphBatch, GraphSAGEBatch
from .task.config import GNNArchConfig, GNNModelConfig
from .task.wrapper import GNNTaskWrapper

# Register compute time callback globally once (module-level).
# This preserves the legacy import side effect while callbacks own the code.
_COMPUTE_TIME_CB = ComputeTimeCallback()
register_global_callback(_COMPUTE_TIME_CB)

logger = logging.getLogger(__name__)


class GATv2Model(GNNTaskWrapper):
    """Alias model registered separately for GATv2 experiments."""

    pass


class GCNModel(GNNTaskWrapper):
    """Alias model registered separately for GCN experiments."""

    pass


class GraphSAGEModel(GNNTaskWrapper):
    """Alias model registered separately for GraphSAGE experiments."""

    pass


class GraphTransformerModel(GNNTaskWrapper):
    """Alias model registered separately for GraphTransformer experiments."""

    pass


__all__ = [
    "FullGraphBatch",
    "GATv2Model",
    "GCNModel",
    "GNNArchConfig",
    "GNNModelConfig",
    "GraphSAGEBatch",
    "GraphSAGEModel",
    "GraphTransformerModel",
]
