from typing import Literal

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig

from .model import run_25pt_stencil


class SdkBenchmark25PtStencilDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmark25PtStencilDataProcessor"]
    batch_size: int = 1
    sample_count: int = 10
    size: int = 10
    z_dim: int = 10
    iterations: int = 10
    dx: int = 20


def _gaussian_source(iterations, dtype):
    tau = torch.tensor(1.0, dtype=dtype)
    scale = torch.tensor(8.0, dtype=dtype)
    mscale = torch.tensor(-8.0, dtype=dtype)
    fmax = torch.tensor(25.0, dtype=dtype)
    dt = torch.tensor(0.001610153, dtype=dtype)
    sigma = torch.tensor(0.6, dtype=dtype) * fmax
    t = torch.arange(iterations, dtype=dtype) * dt
    power = (sigma * t - tau) * (sigma * t - tau)
    expf = torch.exp(power * mscale)
    return -2.0 * scale * sigma * (sigma - 2.0 * sigma * scale * power) * expf


class SdkBenchmark25PtStencilDataProcessor:
    def __init__(self, config: SdkBenchmark25PtStencilDataProcessorConfig):
        if isinstance(config, dict):
            config = SdkBenchmark25PtStencilDataProcessorConfig(**config)
        self.config = config
        self.dtype = cstorch.amp.get_floating_point_dtype()

    def create_dataloader(self):
        cfg = self.config
        vp = torch.full(
            (cfg.batch_size, cfg.size, cfg.size, cfg.z_dim),
            10.3703699112,
            dtype=self.dtype,
        )
        source_terms = torch.zeros(
            (cfg.batch_size, cfg.iterations, cfg.size, cfg.size, cfg.z_dim),
            dtype=self.dtype,
        )
        source = _gaussian_source(cfg.iterations, self.dtype)
        src_x = cfg.size // 2 - 5
        src_y = cfg.size // 2 - 5
        src_z = cfg.z_dim // 2 - 5
        source_terms[:, :, src_y, src_x, src_z] = source.reshape(1, -1)
        reference = run_25pt_stencil(
            vp, source_terms, cfg.iterations, cfg.dx
        )
        return SampleGenerator(
            data=(vp, source_terms, reference),
            sample_count=cfg.sample_count,
        )
