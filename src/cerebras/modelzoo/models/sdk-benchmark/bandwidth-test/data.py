from typing import Literal

import torch
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class SdkBenchmarkBandwidthTestDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmarkBandwidthTestDataProcessor"]
    """Name of the data processor."""

    batch_size: Annotated[int, Ge(1)] = 1
    tensor_length: Annotated[int, Ge(1)] = 4096
    loop_count: Annotated[int, Ge(1)] = 8
    channel_count: Annotated[int, Ge(1)] = 4
    sample_count: Annotated[int, Ge(1)] = 16


class SdkBenchmarkBandwidthTestDataProcessor:
    def __init__(self, config: SdkBenchmarkBandwidthTestDataProcessorConfig):
        if isinstance(config, dict):
            config = SdkBenchmarkBandwidthTestDataProcessorConfig(**config)
        self.config = config
        self.batch_size = get_streaming_batch_size(config.batch_size)

    def create_dataloader(self):
        cfg = self.config
        base = torch.arange(cfg.tensor_length, dtype=torch.float32)
        payload = (base.remainder(257.0) / 257.0).repeat(self.batch_size, 1)
        scale = torch.tensor(
            1.0 + (0.001 * float(cfg.channel_count)), dtype=torch.float32
        )
        bias = torch.tensor(
            0.0001 * float(cfg.channel_count), dtype=torch.float32
        )
        reference = payload
        for _ in range(cfg.loop_count):
            reference = reference * scale + bias
        return SampleGenerator(
            data=(payload, reference), sample_count=cfg.sample_count
        )
