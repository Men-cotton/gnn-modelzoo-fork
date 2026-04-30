from typing import Literal

import torch
from torch import nn

from cerebras.modelzoo.config import ModelConfig


class CholeskyBenchmarkModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/cholesky"]
    matrix_size: int = 40

    @property
    def __model_cls__(self):
        return CholeskyBenchmarkModel


class CholeskyBenchmarkModel(nn.Module):
    def __init__(self, config: CholeskyBenchmarkModelConfig):
        if isinstance(config, dict):
            config = config.get("model", config)
            config = CholeskyBenchmarkModelConfig(**config)
        super().__init__()

    def forward(self, batch):
        output = torch.linalg.cholesky(batch["matrix"])
        reference = batch["reference"]
        factor_error = output - reference
        reconstructed = torch.matmul(output, output.transpose(-1, -2))
        residual = reconstructed - batch["matrix"]
        loss = (factor_error * factor_error).mean()
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_factor_error": factor_error.abs().max(),
            "max_residual": residual.abs().max(),
        }
