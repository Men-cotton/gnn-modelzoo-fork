from typing import Literal

import torch

from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class SingleTileMatvecDataConfig(DataConfig):
    data_processor: Literal["SingleTileMatvecDataProcessor"] = (
        "SingleTileMatvecDataProcessor"
    )
    batch_size: int = 1
    sample_count: int = 5
    width: int = 2
    height: int = 2
    tile_size: int = 25
    iters: int = 1


class SingleTileMatvecDataProcessor:
    def __init__(self, config: SingleTileMatvecDataConfig):
        if isinstance(config, dict):
            config = SingleTileMatvecDataConfig(**config)
        self.config = config

    def create_dataloader(self):
        cfg = self.config
        n = cfg.tile_size
        tile_count = cfg.height * cfg.width

        a = torch.arange(tile_count * n * n, dtype=torch.float32)
        a = a.reshape(cfg.height, cfg.width, n, n)
        a = a / float(tile_count * n * n)
        x = torch.linspace(0.125, 1.0, n, dtype=torch.float32)
        x = x.reshape(1, 1, n).repeat(cfg.height, cfg.width, 1)
        single_iter = torch.matmul(a, x.unsqueeze(-1)).squeeze(-1)
        reference = single_iter * float(cfg.iters)

        data = {
            "A": a.unsqueeze(0).repeat(cfg.batch_size, 1, 1, 1, 1),
            "x": x.unsqueeze(0).repeat(cfg.batch_size, 1, 1, 1),
            "reference": reference.unsqueeze(0).repeat(cfg.batch_size, 1, 1, 1),
        }
        return SampleGenerator(data=data, sample_count=cfg.sample_count)
