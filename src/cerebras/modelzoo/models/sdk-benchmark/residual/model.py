from typing import Literal

import torch
import torch.nn as nn

from cerebras.modelzoo.config import ModelConfig


class SdkBenchmarkResidualConfig(ModelConfig):
    name: Literal["sdk_benchmark/residual"]
    rows: int = 6
    cols: int = 4

    @property
    def __model_cls__(self):
        return SdkBenchmarkResidualModel


class SdkBenchmarkResidualModel(nn.Module):
    def __init__(self, config: SdkBenchmarkResidualConfig):
        super().__init__()
        if isinstance(config, dict):
            config = config.get("model", config)
            config = SdkBenchmarkResidualConfig(**config)
        self.rows = config.rows
        self.cols = config.cols

    def forward(self, batch):
        matrix, vector, rhs, reference_norm = batch
        residual = rhs - torch.matmul(matrix, vector)
        output = torch.amax(torch.abs(residual.float()), dim=(1, 2))
        error = output - reference_norm
        loss = (error * error).mean()
        return {
            "loss": loss,
            "output": output,
            "reference": reference_norm,
            "residual": residual,
            "max_abs_error": torch.amax(torch.abs(error)),
        }
