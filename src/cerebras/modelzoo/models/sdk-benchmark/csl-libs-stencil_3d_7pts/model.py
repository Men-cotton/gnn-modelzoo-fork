from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from cerebras.modelzoo.config import ModelConfig


class CslLibsStencil3d7ptsConfig(ModelConfig):
    name: Literal["sdk_benchmark/csl-libs-stencil_3d_7pts"]
    depth: int = 8
    height: int = 8
    width: int = 8

    @property
    def __model_cls__(self):
        return CslLibsStencil3d7ptsModel


class CslLibsStencil3d7ptsModel(nn.Module):
    def __init__(self, config: CslLibsStencil3d7ptsConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = CslLibsStencil3d7ptsConfig(**config)

        super().__init__()
        self.depth = config.depth
        self.height = config.height
        self.width = config.width

    def forward(self, batch):
        x = batch["input"]
        coeff = batch["coefficients"]
        reference = batch["reference"]

        west = F.pad(x[..., :-1], (1, 0, 0, 0, 0, 0))
        east = F.pad(x[..., 1:], (0, 1, 0, 0, 0, 0))
        south = F.pad(x[:, :, :-1, :], (0, 0, 1, 0, 0, 0))
        north = F.pad(x[:, :, 1:, :], (0, 0, 0, 1, 0, 0))
        bottom = F.pad(x[:, :-1, :, :], (0, 0, 0, 0, 1, 0))
        top = F.pad(x[:, 1:, :, :], (0, 0, 0, 0, 0, 1))

        output = (
            coeff[:, 0] * west
            + coeff[:, 1] * east
            + coeff[:, 2] * south
            + coeff[:, 3] * north
            + coeff[:, 4] * bottom
            + coeff[:, 5] * top
            + coeff[:, 6] * x
        )
        error = (output - reference).abs()
        loss = (error * error).mean()
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_error": error.max(),
        }
