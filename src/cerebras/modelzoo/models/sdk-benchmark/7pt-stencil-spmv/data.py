from typing import Literal

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig

from .model import apply_7pt_stencil


class SdkBenchmark7ptStencilSpmvDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmark7ptStencilSpmvDataProcessor"]
    batch_size: int = 1
    sample_count: int = 10
    height: int = 4
    width: int = 4
    z_dim: int = 8


class SdkBenchmark7ptStencilSpmvDataProcessor:
    def __init__(self, config: SdkBenchmark7ptStencilSpmvDataProcessorConfig):
        if isinstance(config, dict):
            config = SdkBenchmark7ptStencilSpmvDataProcessorConfig(**config)
        self.config = config
        self.dtype = cstorch.amp.get_floating_point_dtype()

    def create_dataloader(self):
        cfg = self.config
        shape = (cfg.batch_size, cfg.height, cfg.width, cfg.z_dim)
        x = torch.arange(cfg.batch_size * cfg.height * cfg.width * cfg.z_dim)
        x = x.reshape(shape).to(dtype=self.dtype) + 100.0

        coeff_values = torch.tensor(
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 6.0],
            dtype=self.dtype,
        )
        coeff = coeff_values.reshape(1, 1, 1, 7).repeat(
            cfg.batch_size, cfg.height, cfg.width, 1
        )
        reference = apply_7pt_stencil(x, coeff)
        return SampleGenerator(
            data=(x, coeff, reference),
            sample_count=cfg.sample_count,
        )
