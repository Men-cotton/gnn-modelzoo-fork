from typing import Literal

import torch
import torch.nn as nn
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.config import ModelConfig


class ConjugateGradientModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/conjugate-gradient"]
    height: Annotated[int, Ge(1)] = 5
    width: Annotated[int, Ge(1)] = 5
    z_dim: Annotated[int, Ge(2)] = 5
    iterations: Annotated[int, Ge(1)] = 4
    epsilon: float = 1.0e-6

    @property
    def __model_cls__(self):
        return ConjugateGradientModel


class ConjugateGradientModel(nn.Module):
    def __init__(self, config: ConjugateGradientModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = ConjugateGradientModelConfig(**config)
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
            - east
            - south
            - north
            - bottom
            - top
        )

    def forward(self, batch):
        x, b, reference = batch
        r = b - self._stencil(x)
        p = r
        rho = (r * r).sum(dim=(1, 2, 3), keepdim=True)

        for _ in range(self.iterations):
            w = self._stencil(p)
            eta = (p * w).sum(dim=(1, 2, 3), keepdim=True)
            alpha = rho / (eta + self.epsilon)
            x = x + alpha * p
            r = r - alpha * w
            rho_next = (r * r).sum(dim=(1, 2, 3), keepdim=True)
            beta = rho_next / (rho + self.epsilon)
            p = r + beta * p
            rho = rho_next

        diff = x - reference
        loss = (diff * diff).mean()
        return {
            "loss": loss,
            "output": x,
            "reference": reference,
            "max_error": diff.abs().amax(),
            "residual_norm_sq": rho.mean(),
        }
