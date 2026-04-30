from typing import Literal

import torch

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class SdkBenchmarkCholeskyDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmarkCholeskyDataProcessor"]
    batch_size: int = 1
    sample_count: int = 4
    matrix_size: int = 40


class SdkBenchmarkCholeskyDataProcessor:
    def __init__(self, config: SdkBenchmarkCholeskyDataProcessorConfig):
        if isinstance(config, dict):
            config = SdkBenchmarkCholeskyDataProcessorConfig(**config)
        self.config = config
        self.batch_size = get_streaming_batch_size(config.batch_size)

    def create_dataloader(self):
        n = self.config.matrix_size
        values = torch.arange(1, n * n + 1, dtype=torch.float32).reshape(n, n)
        lower = torch.tril(values) / torch.tensor(float(n * n), dtype=torch.float32)
        lower = lower + torch.eye(n, dtype=torch.float32)
        matrix = torch.matmul(lower, lower.transpose(0, 1))
        data = {
            "matrix": matrix.unsqueeze(0).repeat(self.batch_size, 1, 1),
            "reference": lower.unsqueeze(0).repeat(self.batch_size, 1, 1),
        }
        return SampleGenerator(data, self.config.sample_count)
