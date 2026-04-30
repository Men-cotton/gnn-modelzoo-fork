from typing import Literal

import torch
import torch.nn as nn
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.config import ModelConfig


class SdkBenchmarkRowColBroadcastModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/row-col-broadcast"]
    """Model registry name."""

    height: Annotated[int, Ge(1)] = 8
    width: Annotated[int, Ge(1)] = 8
    pe_length: Annotated[int, Ge(1)] = 16
    broadcast_axis: Literal["row", "column"] = "row"

    @property
    def __model_cls__(self):
        return SdkBenchmarkRowColBroadcastModel


class SdkBenchmarkRowColBroadcastModel(nn.Module):
    """Static tensor analogue of row/column H2D broadcast.

    The CSL sample reduces host traffic by using row/column broadcast memcpy
    APIs. This PyTorch graph cannot exercise those APIs; it checks the
    fixed-shape broadcast semantics with `expand`.
    """

    def __init__(self, config: SdkBenchmarkRowColBroadcastModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = SdkBenchmarkRowColBroadcastModelConfig(**config)
        super().__init__()
        self.height = config.height
        self.width = config.width
        self.broadcast_axis = config.broadcast_axis

    def forward(self, batch):
        source, reference = batch
        if self.broadcast_axis == "row":
            output = source.expand(-1, self.height, self.width, -1)
        else:
            output = source.expand(-1, self.height, self.width, -1)

        error = output - reference
        loss = (error * error).mean()
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_abs_error": error.abs().amax(),
        }
