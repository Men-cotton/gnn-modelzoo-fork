from typing import Literal

import torch
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class SdkBenchmarkHistogramTorusDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmarkHistogramTorusDataProcessor"]
    """Name of the data processor."""

    batch_size: Annotated[int, Ge(1)] = 1
    hist_height: Annotated[int, Ge(1)] = 4
    hist_width: Annotated[int, Ge(1)] = 4
    input_size: Annotated[int, Ge(1)] = 8
    num_buckets: Annotated[int, Ge(1)] = 5
    bucket_size: Annotated[int, Ge(1)] = 10
    sample_count: Annotated[int, Ge(1)] = 16


class SdkBenchmarkHistogramTorusDataProcessor:
    def __init__(self, config: SdkBenchmarkHistogramTorusDataProcessorConfig):
        if isinstance(config, dict):
            config = SdkBenchmarkHistogramTorusDataProcessorConfig(**config)
        self.config = config
        self.batch_size = get_streaming_batch_size(config.batch_size)

    def create_dataloader(self):
        cfg = self.config
        input_count = cfg.hist_height * cfg.hist_width * cfg.input_size
        value_range = (
            cfg.hist_height
            * cfg.hist_width
            * cfg.num_buckets
            * cfg.bucket_size
        )
        base = torch.arange(
            self.batch_size * input_count, dtype=torch.int64
        ).reshape(
            self.batch_size, cfg.hist_height, cfg.hist_width, cfg.input_size
        )
        inputs = ((base * 17) + 7).remainder(value_range)
        global_bucket_ids = torch.arange(
            cfg.hist_height * cfg.hist_width * cfg.num_buckets,
            dtype=torch.int64,
        ).reshape(
            1, 1, 1, 1, cfg.hist_height, cfg.hist_width, cfg.num_buckets
        )
        global_bucket = inputs.reshape(
            self.batch_size, cfg.hist_height, cfg.hist_width, cfg.input_size, 1, 1, 1
        ) // cfg.bucket_size
        reference = (global_bucket == global_bucket_ids).to(torch.float32).sum(
            dim=(1, 2, 3)
        )
        return SampleGenerator(
            data=(inputs, reference), sample_count=cfg.sample_count
        )
