from typing import Literal

import torch
import torch.nn as nn

from cerebras.modelzoo.config import ModelConfig


class SdkBenchmark25PtStencilConfig(ModelConfig):
    name: Literal["sdk_benchmark/25-pt-stencil"]
    size: int = 10
    z_dim: int = 10
    iterations: int = 10
    dx: int = 20

    @property
    def __model_cls__(self):
        return SdkBenchmark25PtStencilModel


def _shift_zero(x, dim, offset):
    out = torch.zeros_like(x)
    if dim == 1 and offset < 0:
        out[:, -offset:, :, :] = x[:, :offset, :, :]
    elif dim == 1:
        out[:, :-offset, :, :] = x[:, offset:, :, :]
    elif dim == 2 and offset < 0:
        out[:, :, -offset:, :] = x[:, :, :offset, :]
    elif dim == 2:
        out[:, :, :-offset, :] = x[:, :, offset:, :]
    elif dim == 3 and offset < 0:
        out[:, :, :, -offset:] = x[:, :, :, :offset]
    else:
        out[:, :, :, :-offset] = x[:, :, :, offset:]
    return out


def _stencil25_laplacian(x, dx):
    dx2 = float(dx * dx)
    c0 = -205.0 / 72.0 / dx2
    coeffs = (
        8.0 / 5.0 / dx2,
        -1.0 / 5.0 / dx2,
        8.0 / 315.0 / dx2,
        -1.0 / 560.0 / dx2,
    )
    lap = (c0 * 3.0) * x
    for distance, coeff in enumerate(coeffs, start=1):
        lap = lap + coeff * (
            _shift_zero(x, 1, -distance)
            + _shift_zero(x, 1, distance)
            + _shift_zero(x, 2, -distance)
            + _shift_zero(x, 2, distance)
            + _shift_zero(x, 3, -distance)
            + _shift_zero(x, 3, distance)
        )
    return lap


def run_25pt_stencil(vp, source_terms, iterations, dx):
    previous = torch.zeros_like(vp)
    current = torch.zeros_like(vp)
    for step in range(iterations):
        lap = _stencil25_laplacian(current, dx)
        update = (2.0 * current) - previous + (vp * lap)
        update = update + source_terms[:, step, :, :, :]
        previous = current
        current = update
    return current


class SdkBenchmark25PtStencilModel(nn.Module):
    def __init__(self, config: SdkBenchmark25PtStencilConfig):
        super().__init__()
        if isinstance(config, dict):
            config = config.get("model", config)
            config = SdkBenchmark25PtStencilConfig(**config)
        self.iterations = config.iterations
        self.dx = config.dx

    def forward(self, batch):
        vp, source_terms, reference = batch
        output = run_25pt_stencil(
            vp, source_terms, self.iterations, self.dx
        )
        error = output - reference
        loss = (error.float() * error.float()).mean()
        max_abs_error = torch.amax(torch.abs(error.float()))
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_abs_error": max_abs_error,
        }
