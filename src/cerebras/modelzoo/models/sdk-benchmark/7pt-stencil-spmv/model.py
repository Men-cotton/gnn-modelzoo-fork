from typing import Literal

import torch
import torch.nn as nn

from cerebras.modelzoo.config import ModelConfig


class SdkBenchmark7ptStencilSpmvConfig(ModelConfig):
    name: Literal["sdk_benchmark/7pt-stencil-spmv"]
    height: int = 4
    width: int = 4
    z_dim: int = 8

    @property
    def __model_cls__(self):
        return SdkBenchmark7ptStencilSpmvModel


def _shift_zero(x, dim, offset):
    out = torch.zeros_like(x)
    if dim == 1 and offset == -1:
        out[:, 1:, :, :] = x[:, :-1, :, :]
    elif dim == 1 and offset == 1:
        out[:, :-1, :, :] = x[:, 1:, :, :]
    elif dim == 2 and offset == -1:
        out[:, :, 1:, :] = x[:, :, :-1, :]
    elif dim == 2 and offset == 1:
        out[:, :, :-1, :] = x[:, :, 1:, :]
    elif dim == 3 and offset == -1:
        out[:, :, :, 1:] = x[:, :, :, :-1]
    else:
        out[:, :, :, :-1] = x[:, :, :, 1:]
    return out


def apply_7pt_stencil(x, coeff):
    west = _shift_zero(x, 2, -1)
    east = _shift_zero(x, 2, 1)
    south = _shift_zero(x, 1, -1)
    north = _shift_zero(x, 1, 1)
    bottom = _shift_zero(x, 3, -1)
    top = _shift_zero(x, 3, 1)

    return (
        coeff[..., 0].unsqueeze(-1) * west
        + coeff[..., 1].unsqueeze(-1) * east
        + coeff[..., 2].unsqueeze(-1) * south
        + coeff[..., 3].unsqueeze(-1) * north
        + coeff[..., 4].unsqueeze(-1) * bottom
        + coeff[..., 5].unsqueeze(-1) * top
        + coeff[..., 6].unsqueeze(-1) * x
    )


class SdkBenchmark7ptStencilSpmvModel(nn.Module):
    def __init__(self, config: SdkBenchmark7ptStencilSpmvConfig):
        super().__init__()
        if isinstance(config, dict):
            config = config.get("model", config)
            config = SdkBenchmark7ptStencilSpmvConfig(**config)
        self.height = config.height
        self.width = config.width
        self.z_dim = config.z_dim

    def forward(self, batch):
        x, coeff, reference = batch
        output = apply_7pt_stencil(x, coeff)
        error = output - reference
        loss = (error.float() * error.float()).mean()
        max_abs_error = torch.amax(torch.abs(error.float()))
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_abs_error": max_abs_error,
        }
