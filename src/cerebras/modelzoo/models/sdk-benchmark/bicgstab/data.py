from typing import Literal

import torch
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class BiCGSTABDataProcessorConfig(DataConfig):
    data_processor: Literal["BiCGSTABDataProcessor"]
    batch_size: Annotated[int, Ge(1)] = 1
    sample_count: Annotated[int, Ge(1)] = 8
    height: Annotated[int, Ge(1)] = 5
    width: Annotated[int, Ge(1)] = 5
    z_dim: Annotated[int, Ge(2)] = 5
    iterations: Annotated[int, Ge(1)] = 4
    epsilon: float = 1.0e-6


def _stencil(x, center):
    zero_w = x[:, :, :1, :] * 0.0
    west = torch.cat((zero_w, x[:, :, :-1, :]), dim=2)
    east = torch.cat((x[:, :, 1:, :], zero_w), dim=2)
    zero_h = x[:, :1, :, :] * 0.0
    north = torch.cat((zero_h, x[:, :-1, :, :]), dim=1)
    south = torch.cat((x[:, 1:, :, :], zero_h), dim=1)
    zero_z = x[:, :, :, :1] * 0.0
    bottom = torch.cat((zero_z, x[:, :, :, :-1]), dim=3)
    top = torch.cat((x[:, :, :, 1:], zero_z), dim=3)
    return center * x - west - east - south - north - bottom - top


def _run_bicgstab(x, b, center, iterations, epsilon):
    r0 = b - _stencil(x, center)
    r = r0
    p = r0
    rho = (r0 * r).sum(dim=(1, 2, 3), keepdim=True)
    for _ in range(iterations):
        v = _stencil(p, center)
        r0_dot_v = (r0 * v).sum(dim=(1, 2, 3), keepdim=True)
        alpha = rho / (r0_dot_v + epsilon)
        s = r - alpha * v
        t = _stencil(s, center)
        t_dot_s = (t * s).sum(dim=(1, 2, 3), keepdim=True)
        t_dot_t = (t * t).sum(dim=(1, 2, 3), keepdim=True)
        omega = t_dot_s / (t_dot_t + epsilon)
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho_next = (r0 * r).sum(dim=(1, 2, 3), keepdim=True)
        beta = (rho_next / (rho + epsilon)) * (alpha / (omega + epsilon))
        p = r + beta * (p - omega * v)
        rho = rho_next
    return x


class BiCGSTABDataProcessor:
    def __init__(self, config: BiCGSTABDataProcessorConfig):
        if isinstance(config, dict):
            config = BiCGSTABDataProcessorConfig(**config)
        self.config = config

    def create_dataloader(self):
        cfg = self.config
        shape = (1, cfg.height, cfg.width, cfg.z_dim)
        x = torch.zeros(shape, dtype=torch.float32)
        b = torch.arange(cfg.height * cfg.width * cfg.z_dim, dtype=torch.float32)
        b = b.reshape(shape) + 1.0
        columns = torch.arange(cfg.width, dtype=torch.float32)
        center = 6.0 + columns.reshape(1, 1, cfg.width, 1)
        center = center.expand(1, cfg.height, cfg.width, 1).clone()
        reference = _run_bicgstab(x, b, center, cfg.iterations, cfg.epsilon)
        data = (
            x.repeat(cfg.batch_size, 1, 1, 1),
            b.repeat(cfg.batch_size, 1, 1, 1),
            reference.repeat(cfg.batch_size, 1, 1, 1),
        )
        return SampleGenerator(data=data, sample_count=cfg.sample_count)
