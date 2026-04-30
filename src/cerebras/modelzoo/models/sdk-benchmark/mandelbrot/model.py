from typing import Literal

import torch
from torch import nn

from cerebras.modelzoo.config import ModelConfig


class MandelbrotBenchmarkModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/mandelbrot"]
    rows: int = 16
    cols: int = 16
    max_iters: int = 32

    @property
    def __model_cls__(self):
        return MandelbrotBenchmarkModel


class MandelbrotKernel(nn.Module):
    def __init__(self, max_iters: int):
        super().__init__()
        self.max_iters = max_iters

    def forward(self, c_real, c_imag):
        z_real = c_real
        z_imag = c_imag
        active = torch.ones_like(c_real, dtype=torch.bool)
        counts = torch.full_like(c_real, float(self.max_iters))

        for step in range(self.max_iters + 1):
            escaped = (z_real * z_real + z_imag * z_imag) >= 4.0
            escaped_now = active & escaped
            counts = torch.where(
                escaped_now,
                torch.full_like(counts, float(step)),
                counts,
            )
            active = active & (~escaped_now)

            if step < self.max_iters:
                next_real = z_real * z_real - z_imag * z_imag + c_real
                next_imag = 2.0 * z_real * z_imag + c_imag
                z_real = torch.where(active, next_real, z_real)
                z_imag = torch.where(active, next_imag, z_imag)

        return counts


class MandelbrotBenchmarkModel(nn.Module):
    def __init__(self, config: MandelbrotBenchmarkModelConfig):
        if isinstance(config, dict):
            config = config.get("model", config)
            config = MandelbrotBenchmarkModelConfig(**config)

        super().__init__()
        self.kernel = MandelbrotKernel(config.max_iters)

    def forward(self, batch):
        output = self.kernel(batch["c_real"], batch["c_imag"])
        reference = batch["reference"]
        error = (output - reference).abs()
        loss = error.float().mean()
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_error": error.max(),
        }
