from typing import Literal

import torch
import torch.nn as nn

from cerebras.modelzoo.config import ModelConfig


class GemvCheckerboardPatternModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/gemv-checkerboard-pattern"] = (
        "sdk_benchmark/gemv-checkerboard-pattern"
    )
    matrix_rows: int = 32
    matrix_cols: int = 16
    kernel_rows: int = 4
    kernel_cols: int = 4

    @property
    def __model_cls__(self):
        return GemvCheckerboardPatternModel


class GemvCheckerboardPatternModel(nn.Module):
    def __init__(self, config: GemvCheckerboardPatternModelConfig):
        if isinstance(config, dict):
            config = GemvCheckerboardPatternModelConfig(**config)
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
        squared_error = error.float() * error.float()
        loss = squared_error.mean()
        max_abs_error = torch.amax(torch.abs(error.float()))
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_abs_error": max_abs_error,
        }
