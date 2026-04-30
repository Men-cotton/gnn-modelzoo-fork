from typing import Literal

import torch
import torch.nn as nn

from cerebras.modelzoo.config import ModelConfig


class SdkBenchmarkSpmvHypersparseConfig(ModelConfig):
    name: Literal["sdk_benchmark/spmv-hypersparse"]
    rows: int = 16
    cols: int = 16

    @property
    def __model_cls__(self):
        return SdkBenchmarkSpmvHypersparseModel


class SdkBenchmarkSpmvHypersparseModel(nn.Module):
    def __init__(self, config: SdkBenchmarkSpmvHypersparseConfig):
        super().__init__()
        if isinstance(config, dict):
            config = config.get("model", config)
            config = SdkBenchmarkSpmvHypersparseConfig(**config)
        self.rows = config.rows
        self.cols = config.cols

    def forward(self, batch):
        matrix, vector, reference = batch
        output = torch.matmul(matrix, vector).squeeze(-1)
        error = output - reference
        loss = (error.float() * error.float()).mean()
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_abs_error": torch.amax(torch.abs(error.float())),
        }
