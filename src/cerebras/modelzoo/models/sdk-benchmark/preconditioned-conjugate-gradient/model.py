from typing import Literal

import torch
import torch.nn as nn
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.config import ModelConfig


class PreconditionedConjugateGradientModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/preconditioned-conjugate-gradient"]
    height: Annotated[int, Ge(1)] = 5
    width: Annotated[int, Ge(1)] = 5
    z_dim: Annotated[int, Ge(2)] = 5
    iterations: Annotated[int, Ge(1)] = 4
    epsilon: float = 1.0e-6

    @property
    def __model_cls__(self):
        return PreconditionedConjugateGradientModel


class PreconditionedConjugateGradientModel(nn.Module):
    def __init__(self, config: PreconditionedConjugateGradientModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = PreconditionedConjugateGradientModelConfig(**config)
        super().__init__()
        self.iterations = config.iterations
        self.epsilon = config.epsilon
        columns = torch.arange(config.width, dtype=torch.float32)
        center = 6.0 + columns.reshape(1, 1, config.width, 1)
        self.register_buffer(
            "center",
            center.expand(1, config.height, config.width, 1).clone(),
            persistent=False,
        )
        self.register_buffer(
            "inv_diagonal", 1.0 / self.center, persistent=False
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
        z = self.inv_diagonal * r
        p = z
        rho = (r * z).sum(dim=(1, 2, 3), keepdim=True)
        xi = (r * r).sum(dim=(1, 2, 3), keepdim=True)

        for _ in range(self.iterations):
            w = self._stencil(p)
            eta = (p * w).sum(dim=(1, 2, 3), keepdim=True)
            alpha = rho / (eta + self.epsilon)
            x = x + alpha * p
            r = r - alpha * w
            z = self.inv_diagonal * r
            rho_next = (r * z).sum(dim=(1, 2, 3), keepdim=True)
            beta = rho_next / (rho + self.epsilon)
            p = z + beta * p
            rho = rho_next
            xi = (r * r).sum(dim=(1, 2, 3), keepdim=True)

        diff = x - reference
        loss = (diff * diff).mean()
        return {
            "loss": loss,
            "output": x,
            "reference": reference,
            "max_error": diff.abs().amax(),
            "residual_norm_sq": xi.mean(),
        }
