from typing import Literal

import torch
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class PowerMethodDataProcessorConfig(DataConfig):
    data_processor: Literal["PowerMethodDataProcessor"]
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
    return center * x - west - 2.0 * east - 3.0 * south - 4.0 * north - 5.0 * bottom - 6.0 * top


def _run_power(x, center, iterations, epsilon):
    x = x / (torch.linalg.vector_norm(x, dim=(1, 2, 3), keepdim=True) + epsilon)
    for _ in range(iterations):
        y = _stencil(x, center)
        x = y / (
            torch.linalg.vector_norm(y, dim=(1, 2, 3), keepdim=True) + epsilon
        )
    return x


class PowerMethodDataProcessor:
    def __init__(self, config: PowerMethodDataProcessorConfig):
        if isinstance(config, dict):
            config = PowerMethodDataProcessorConfig(**config)
        self.config = config

    def create_dataloader(self):
        cfg = self.config
        shape = (1, cfg.height, cfg.width, cfg.z_dim)
        x = torch.arange(cfg.height * cfg.width * cfg.z_dim, dtype=torch.float32)
        x = x.reshape(shape) + 100.0
        center = torch.full((1, cfg.height, cfg.width, 1), 6.0)
        reference = _run_power(x, center, cfg.iterations, cfg.epsilon)
        data = (
            x.repeat(cfg.batch_size, 1, 1, 1),
            reference.repeat(cfg.batch_size, 1, 1, 1),
        )
        return SampleGenerator(data=data, sample_count=cfg.sample_count)
