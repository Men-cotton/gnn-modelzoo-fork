from typing import Literal

import torch
from torch import nn

from cerebras.modelzoo.config import ModelConfig


class FFTBenchmarkModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/FFT"]
    dim: Literal[1, 2] = 2
    size: int = 4

    @property
    def __model_cls__(self):
        return FFTBenchmarkModel


def _complex_matmul(a_real, a_imag, b_real, b_imag):
    real = torch.matmul(a_real, b_real) - torch.matmul(a_imag, b_imag)
    imag = torch.matmul(a_real, b_imag) + torch.matmul(a_imag, b_real)
    return real, imag


class DenseDFTKernel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x_real, x_imag, w_real, w_imag):
        if self.dim == 1:
            w_t_real = w_real.transpose(-1, -2)
            w_t_imag = w_imag.transpose(-1, -2)
            return _complex_matmul(x_real, x_imag, w_t_real, w_t_imag)

        row_real, row_imag = _complex_matmul(w_real, w_imag, x_real, x_imag)
        w_t_real = w_real.transpose(-1, -2)
        w_t_imag = w_imag.transpose(-1, -2)
        return _complex_matmul(row_real, row_imag, w_t_real, w_t_imag)


class FFTBenchmarkModel(nn.Module):
    def __init__(self, config: FFTBenchmarkModelConfig):
        if isinstance(config, dict):
            config = config.get("model", config)
            config = FFTBenchmarkModelConfig(**config)

        super().__init__()
        self.kernel = DenseDFTKernel(config.dim)

    def forward(self, batch):
        output_real, output_imag = self.kernel(
            batch["input_real"],
            batch["input_imag"],
            batch["twiddle_real"],
            batch["twiddle_imag"],
        )
        real_error = output_real - batch["reference_real"]
        imag_error = output_imag - batch["reference_imag"]
        squared_error = real_error * real_error + imag_error * imag_error
        loss = squared_error.mean()
        return {
            "loss": loss,
            "output_real": output_real,
            "output_imag": output_imag,
            "reference_real": batch["reference_real"],
            "reference_imag": batch["reference_imag"],
            "max_error": squared_error.sqrt().max(),
        }
