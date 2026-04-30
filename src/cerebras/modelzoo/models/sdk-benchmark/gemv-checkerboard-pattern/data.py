from typing import Literal

import torch

from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class GemvCheckerboardPatternDataConfig(DataConfig):
    data_processor: Literal["GemvCheckerboardPatternDataProcessor"] = (
        "GemvCheckerboardPatternDataProcessor"
    )
    batch_size: int = 1
    sample_count: int = 5
    matrix_rows: int = 32
    matrix_cols: int = 16


class GemvCheckerboardPatternDataProcessor:
    def __init__(self, config: GemvCheckerboardPatternDataConfig):
        if isinstance(config, dict):
            config = GemvCheckerboardPatternDataConfig(**config)
        self.config = config

    def create_dataloader(self):
        cfg = self.config
        dtype = torch.float16
        a = torch.arange(
            cfg.matrix_rows * cfg.matrix_cols, dtype=torch.float32
        ).reshape(cfg.matrix_rows, cfg.matrix_cols)
        a = (a / float(cfg.matrix_rows * cfg.matrix_cols)).to(dtype)
        x = torch.linspace(0.125, 1.0, cfg.matrix_cols, dtype=torch.float32)
        x = x.reshape(cfg.matrix_cols, 1).to(dtype)
        b = torch.linspace(-0.25, 0.25, cfg.matrix_rows, dtype=torch.float32)
        b = b.reshape(cfg.matrix_rows, 1).to(dtype)
        reference = torch.matmul(a, x) + b

        data = {
            "A": a.unsqueeze(0).repeat(cfg.batch_size, 1, 1),
            "x": x.unsqueeze(0).repeat(cfg.batch_size, 1, 1),
            "b": b.unsqueeze(0).repeat(cfg.batch_size, 1, 1),
            "reference": reference.unsqueeze(0).repeat(cfg.batch_size, 1, 1),
        }
        return SampleGenerator(data=data, sample_count=cfg.sample_count)
