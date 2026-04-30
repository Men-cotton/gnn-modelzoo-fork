from typing import Literal

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class SdkBenchmarkSpmvHypersparseDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmarkSpmvHypersparseDataProcessor"]
    batch_size: int = 1
    sample_count: int = 10
    rows: int = 16
    cols: int = 16


class SdkBenchmarkSpmvHypersparseDataProcessor:
    def __init__(self, config: SdkBenchmarkSpmvHypersparseDataProcessorConfig):
        if isinstance(config, dict):
            config = SdkBenchmarkSpmvHypersparseDataProcessorConfig(**config)
        self.config = config
        self.dtype = cstorch.amp.get_floating_point_dtype()

    def create_dataloader(self):
        cfg = self.config
        row_ids = torch.arange(cfg.rows).reshape(cfg.rows, 1)
        col_ids = torch.arange(cfg.cols).reshape(1, cfg.cols)
        mask = ((row_ids * 7 + col_ids * 11) % 13 == 0).to(self.dtype)
        values = ((row_ids + 1) * (col_ids + 3)).to(self.dtype) / 100.0
        matrix = (mask * values).unsqueeze(0).repeat(cfg.batch_size, 1, 1)
        vector = torch.arange(cfg.batch_size * cfg.cols)
        vector = vector.reshape(cfg.batch_size, cfg.cols, 1)
        vector = vector.to(dtype=self.dtype) / 10.0 + 1.0
        reference = torch.matmul(matrix, vector).squeeze(-1)
        return SampleGenerator(
            data=(matrix, vector, reference),
            sample_count=cfg.sample_count,
        )
