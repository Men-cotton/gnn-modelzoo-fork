from typing import Literal

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class SdkBenchmarkResidualDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmarkResidualDataProcessor"]
    batch_size: int = 1
    sample_count: int = 10
    rows: int = 6
    cols: int = 4


class SdkBenchmarkResidualDataProcessor:
    def __init__(self, config: SdkBenchmarkResidualDataProcessorConfig):
        if isinstance(config, dict):
            config = SdkBenchmarkResidualDataProcessorConfig(**config)
        self.config = config
        self.dtype = cstorch.amp.get_floating_point_dtype()

    def create_dataloader(self):
        cfg = self.config
        matrix = torch.arange(cfg.batch_size * cfg.rows * cfg.cols)
        matrix = matrix.reshape(cfg.batch_size, cfg.rows, cfg.cols)
        matrix = matrix.to(dtype=self.dtype)
        vector = torch.arange(cfg.batch_size * cfg.cols)
        vector = vector.reshape(cfg.batch_size, cfg.cols, 1)
        vector = vector.to(dtype=self.dtype) + 100.0
        rhs = torch.arange(cfg.batch_size * cfg.rows)
        rhs = rhs.reshape(cfg.batch_size, cfg.rows, 1)
        rhs = rhs.to(dtype=self.dtype) + 200.0
        residual = rhs - torch.matmul(matrix, vector)
        reference_norm = torch.amax(torch.abs(residual.float()), dim=(1, 2))
        return SampleGenerator(
            data=(matrix, vector, rhs, reference_norm),
            sample_count=cfg.sample_count,
        )
