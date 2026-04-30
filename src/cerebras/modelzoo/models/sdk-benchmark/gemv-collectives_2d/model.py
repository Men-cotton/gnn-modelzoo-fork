from typing import Literal

import torch
import torch.nn as nn

from cerebras.modelzoo.config import ModelConfig


class GemvCollectives2dModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/gemv-collectives_2d"] = (
        "sdk_benchmark/gemv-collectives_2d"
    )
    matrix_rows: int = 32
    matrix_cols: int = 16
    kernel_rows: int = 4
    kernel_cols: int = 4

    @property
    def __model_cls__(self):
        return GemvCollectives2dModel


class GemvCollectives2dModel(nn.Module):
    def __init__(self, config: GemvCollectives2dModelConfig):
        if isinstance(config, dict):
            config = GemvCollectives2dModelConfig(**config)
        super().__init__()
        self.matrix_rows = config.matrix_rows
        self.matrix_cols = config.matrix_cols

    def forward(self, batch):
        a = batch["A"]
        x = batch["x"]
        b = batch["b"]
        reference = batch["reference"]

        output = torch.matmul(a, x) + b
        error = output - reference
        squared_error = error * error
        loss = squared_error.mean()
        max_abs_error = torch.amax(torch.abs(error))
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_abs_error": max_abs_error,
        }
