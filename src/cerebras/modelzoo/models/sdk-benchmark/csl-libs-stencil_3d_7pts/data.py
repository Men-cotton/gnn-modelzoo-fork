from typing import Literal

import torch
import torch.nn.functional as F

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class CslLibsStencil3d7ptsDataProcessorConfig(DataConfig):
    data_processor: Literal["CslLibsStencil3d7ptsDataProcessor"]
    batch_size: int = 1
    sample_count: int = 10
    depth: int = 8
    height: int = 8
    width: int = 8


class CslLibsStencil3d7ptsDataProcessor:
    def __init__(self, config: CslLibsStencil3d7ptsDataProcessorConfig):
        if isinstance(config, dict):
            config = CslLibsStencil3d7ptsDataProcessorConfig(**config)

        self.config = config
        self.batch_size = get_streaming_batch_size(config.batch_size)
        self.dtype = cstorch.amp.get_floating_point_dtype()

    def create_dataloader(self):
        cfg = self.config
        values = torch.arange(
            self.batch_size * cfg.depth * cfg.height * cfg.width,
            dtype=torch.float32,
        ).reshape(self.batch_size, cfg.depth, cfg.height, cfg.width)
        values = ((values % 29.0) - 14.0) / 17.0
        values = values.to(self.dtype)

        coefficients = torch.tensor(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0],
            dtype=self.dtype,
        ).reshape(1, 7, 1, 1, 1)
        coefficients = coefficients.expand(self.batch_size, -1, -1, -1, -1)

        reference = self._stencil(values, coefficients)
        return SampleGenerator(
            {
                "input": values,
                "coefficients": coefficients,
                "reference": reference,
            },
            cfg.sample_count,
        )

    @staticmethod
    def _stencil(x, coeff):
        west = F.pad(x[..., :-1], (1, 0, 0, 0, 0, 0))
        east = F.pad(x[..., 1:], (0, 1, 0, 0, 0, 0))
        south = F.pad(x[:, :, :-1, :], (0, 0, 1, 0, 0, 0))
        north = F.pad(x[:, :, 1:, :], (0, 0, 0, 1, 0, 0))
        bottom = F.pad(x[:, :-1, :, :], (0, 0, 0, 0, 1, 0))
        top = F.pad(x[:, 1:, :, :], (0, 0, 0, 0, 0, 1))
        return (
            coeff[:, 0] * west
            + coeff[:, 1] * east
            + coeff[:, 2] * south
            + coeff[:, 3] * north
            + coeff[:, 4] * bottom
            + coeff[:, 5] * top
            + coeff[:, 6] * x
        )
