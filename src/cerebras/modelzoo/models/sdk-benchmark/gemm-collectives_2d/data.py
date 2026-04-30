from typing import Literal

import torch

from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class GemmCollectives2dDataConfig(DataConfig):
    data_processor: Literal["GemmCollectives2dDataProcessor"] = (
        "GemmCollectives2dDataProcessor"
    )
    batch_size: int = 1
    sample_count: int = 5
    p: int = 4
    mt: int = 14
    kt: int = 14
    nt: int = 14


class GemmCollectives2dDataProcessor:
    def __init__(self, config: GemmCollectives2dDataConfig):
        if isinstance(config, dict):
            config = GemmCollectives2dDataConfig(**config)
        self.config = config

    def create_dataloader(self):
        cfg = self.config
        m = cfg.p * cfg.mt
        k = cfg.p * cfg.kt
        n = cfg.p * cfg.nt

        a = torch.arange(m * k, dtype=torch.float32).reshape(m, k)
        a = a / float(m * k)
        b = torch.arange(k * n, dtype=torch.float32).reshape(k, n)
        b = (b / float(k * n)) - 0.25
        reference = torch.matmul(a, b)

        data = {
            "A": a.unsqueeze(0).repeat(cfg.batch_size, 1, 1),
            "B": b.unsqueeze(0).repeat(cfg.batch_size, 1, 1),
            "reference": reference.unsqueeze(0).repeat(cfg.batch_size, 1, 1),
        }
        return SampleGenerator(data=data, sample_count=cfg.sample_count)
