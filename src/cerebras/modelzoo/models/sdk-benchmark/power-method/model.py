from typing import Literal

import torch
import torch.nn as nn
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.config import ModelConfig


class PowerMethodModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/power-method"]
    height: Annotated[int, Ge(1)] = 5
    width: Annotated[int, Ge(1)] = 5
    z_dim: Annotated[int, Ge(2)] = 5
    iterations: Annotated[int, Ge(1)] = 4
    epsilon: float = 1.0e-6

    @property
    def __model_cls__(self):
        return PowerMethodModel


class PowerMethodModel(nn.Module):
    def __init__(self, config: PowerMethodModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = PowerMethodModelConfig(**config)
        super().__init__()
        self.iterations = config.iterations
        self.epsilon = config.epsilon
        self.register_buffer(
            "center",
            torch.full((1, config.height, config.width, 1), 6.0),
            persistent=False,
        )

    def _stencil(self, x):
        zero_w = x[:, :, :1, :] * 0.0
        west = torch.cat((zero_w, x[:, :, :-1, :]), dim=2)
        east = torch.cat((x[:, :, 1:, :], zero_w), dim=2)
        zero_h = x[:, :1, :, :] * 0.0
        north = torch.cat((zero_h, x[:, :-1, :, :]), dim=1)
        south = torch.cat((x[:, 1:, :, :], zero_h), dim=1)
        zero_z = x[:, :, :, :1] * 0.0
        bottom = torch.cat((zero_z, x[:, :, :, :-1]), dim=3)
        top = torch.cat((x[:, :, :, 1:], zero_z), dim=3)
        return (
            self.center * x
            - west
            - 2.0 * east
            - 3.0 * south
            - 4.0 * north
            - 5.0 * bottom
            - 6.0 * top
        )

    def forward(self, batch):
        x, reference = batch
        norm = torch.linalg.vector_norm(x, dim=(1, 2, 3), keepdim=True)
        x = x / (norm + self.epsilon)

        for _ in range(self.iterations):
            y = self._stencil(x)
            norm = torch.linalg.vector_norm(y, dim=(1, 2, 3), keepdim=True)
            x = y / (norm + self.epsilon)

        y = self._stencil(x)
        eigenvalue = (x * y).sum(dim=(1, 2, 3)).mean()
        diff = x - reference
        loss = (diff * diff).mean()
        return {
            "loss": loss,
            "output": x,
            "reference": reference,
            "max_error": diff.abs().amax(),
            "eigenvalue": eigenvalue,
        }
