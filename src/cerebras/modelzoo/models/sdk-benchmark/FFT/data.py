import math
from typing import Literal

import torch

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class SdkBenchmarkFFTDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmarkFFTDataProcessor"]
    batch_size: int = 1
    sample_count: int = 4
    dim: Literal[1, 2] = 2
    size: int = 4
    inverse: bool = False


class SdkBenchmarkFFTDataProcessor:
    def __init__(self, config: SdkBenchmarkFFTDataProcessorConfig):
        if isinstance(config, dict):
            config = SdkBenchmarkFFTDataProcessorConfig(**config)
        self.config = config
        self.batch_size = get_streaming_batch_size(config.batch_size)

    def create_dataloader(self):
        size = self.config.size
        values = torch.arange(size**self.config.dim, dtype=torch.float32)
        values = values / torch.tensor(float(size), dtype=torch.float32)
        if self.config.dim == 1:
            input_real = values.reshape(1, size).repeat(self.batch_size, 1)
            input_imag = torch.zeros_like(input_real)
            reference = (
                torch.fft.ifft(torch.complex(input_real, input_imag))
                if self.config.inverse
                else torch.fft.fft(torch.complex(input_real, input_imag))
            )
        else:
            input_real = values.reshape(1, size, size).repeat(
                self.batch_size, 1, 1
            )
            input_imag = torch.zeros_like(input_real)
            reference = (
                torch.fft.ifft2(torch.complex(input_real, input_imag))
                if self.config.inverse
                else torch.fft.fft2(torch.complex(input_real, input_imag))
            )

        indices = torch.arange(size, dtype=torch.float32)
        phase = torch.outer(indices, indices)
        sign = 1.0 if self.config.inverse else -1.0
        angle = sign * 2.0 * math.pi * phase / float(size)
        scale = 1.0 / float(size) if self.config.inverse else 1.0
        twiddle_real = (torch.cos(angle) * scale).unsqueeze(0).repeat(
            self.batch_size, 1, 1
        )
        twiddle_imag = (torch.sin(angle) * scale).unsqueeze(0).repeat(
            self.batch_size, 1, 1
        )

        data = {
            "input_real": input_real,
            "input_imag": input_imag,
            "twiddle_real": twiddle_real,
            "twiddle_imag": twiddle_imag,
            "reference_real": reference.real.float(),
            "reference_imag": reference.imag.float(),
        }
        return SampleGenerator(data, self.config.sample_count)
