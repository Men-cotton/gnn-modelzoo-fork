from typing import Literal

import torch
import torch.nn as nn
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.config import ModelConfig


class BiCGSTABModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/bicgstab"]
    height: Annotated[int, Ge(1)] = 5
    width: Annotated[int, Ge(1)] = 5
    z_dim: Annotated[int, Ge(2)] = 5
    iterations: Annotated[int, Ge(1)] = 4
    epsilon: float = 1.0e-6

    @property
    def __model_cls__(self):
        return BiCGSTABModel


class BiCGSTABModel(nn.Module):
    def __init__(self, config: BiCGSTABModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = BiCGSTABModelConfig(**config)
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
        r0 = b - self._stencil(x)
        r = r0
        p = r0
        rho = (r0 * r).sum(dim=(1, 2, 3), keepdim=True)
        xi = (r * r).sum(dim=(1, 2, 3), keepdim=True)

        for _ in range(self.iterations):
            v = self._stencil(p)
            r0_dot_v = (r0 * v).sum(dim=(1, 2, 3), keepdim=True)
            alpha = rho / (r0_dot_v + self.epsilon)
            s = r - alpha * v
            t = self._stencil(s)
            t_dot_s = (t * s).sum(dim=(1, 2, 3), keepdim=True)
            t_dot_t = (t * t).sum(dim=(1, 2, 3), keepdim=True)
            omega = t_dot_s / (t_dot_t + self.epsilon)
            x = x + alpha * p + omega * s
            r = s - omega * t
            rho_next = (r0 * r).sum(dim=(1, 2, 3), keepdim=True)
            beta = (rho_next / (rho + self.epsilon)) * (
                alpha / (omega + self.epsilon)
            )
            p = r + beta * (p - omega * v)
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
