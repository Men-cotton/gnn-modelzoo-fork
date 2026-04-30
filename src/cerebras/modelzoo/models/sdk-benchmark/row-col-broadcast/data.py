from typing import Literal

import torch
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class SdkBenchmarkRowColBroadcastDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmarkRowColBroadcastDataProcessor"]
    """Name of the data processor."""

    batch_size: Annotated[int, Ge(1)] = 1
    height: Annotated[int, Ge(1)] = 8
    width: Annotated[int, Ge(1)] = 8
    pe_length: Annotated[int, Ge(1)] = 16
    broadcast_axis: Literal["row", "column"] = "row"
    sample_count: Annotated[int, Ge(1)] = 16


class SdkBenchmarkRowColBroadcastDataProcessor:
    def __init__(self, config: SdkBenchmarkRowColBroadcastDataProcessorConfig):
        if isinstance(config, dict):
            config = SdkBenchmarkRowColBroadcastDataProcessorConfig(**config)
        self.config = config
        self.batch_size = get_streaming_batch_size(config.batch_size)

    def create_dataloader(self):
        cfg = self.config
        if cfg.broadcast_axis == "row":
            source_shape = (self.batch_size, cfg.height, 1, cfg.pe_length)
        else:
            source_shape = (self.batch_size, 1, cfg.width, cfg.pe_length)

        source = torch.arange(
            self.batch_size
            * source_shape[1]
            * source_shape[2]
            * cfg.pe_length,
            dtype=torch.float32,
        ).reshape(source_shape)
        source = source.remainder(97.0) / 97.0
        reference = source.expand(
            self.batch_size, cfg.height, cfg.width, cfg.pe_length
        ).clone()
        return SampleGenerator(
            data=(source, reference), sample_count=cfg.sample_count
        )
