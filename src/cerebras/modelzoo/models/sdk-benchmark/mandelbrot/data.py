from typing import Literal

import torch

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig

from .model import MandelbrotKernel


class SdkBenchmarkMandelbrotDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmarkMandelbrotDataProcessor"]
    batch_size: int = 1
    sample_count: int = 4
    rows: int = 16
    cols: int = 16
    max_iters: int = 32


class SdkBenchmarkMandelbrotDataProcessor:
    def __init__(self, config: SdkBenchmarkMandelbrotDataProcessorConfig):
        if isinstance(config, dict):
            config = SdkBenchmarkMandelbrotDataProcessorConfig(**config)
        self.config = config
        self.batch_size = get_streaming_batch_size(config.batch_size)

    def create_dataloader(self):
        rows = self.config.rows
        cols = self.config.cols
        y = torch.linspace(-1.5, 1.5, rows, dtype=torch.float32)
        x = torch.linspace(-2.0, 1.0, cols, dtype=torch.float32)
        c_imag, c_real = torch.meshgrid(y, x, indexing="ij")
        c_real = c_real.unsqueeze(0).repeat(self.batch_size, 1, 1)
        c_imag = c_imag.unsqueeze(0).repeat(self.batch_size, 1, 1)

        kernel = MandelbrotKernel(self.config.max_iters)
        reference = kernel(c_real, c_imag)
        data = {
            "c_real": c_real,
            "c_imag": c_imag,
            "reference": reference,
        }
        return SampleGenerator(data, self.config.sample_count)
