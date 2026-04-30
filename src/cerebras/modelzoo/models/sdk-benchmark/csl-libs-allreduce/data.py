from typing import Literal

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class CslLibsAllreduceDataProcessorConfig(DataConfig):
    data_processor: Literal["CslLibsAllreduceDataProcessor"]
    batch_size: int = 1
    sample_count: int = 10
    grid_height: int = 4
    grid_width: int = 4
    vector_length: int = 16
    op: Literal["add", "max"] = "add"


class CslLibsAllreduceDataProcessor:
    def __init__(self, config: CslLibsAllreduceDataProcessorConfig):
        if isinstance(config, dict):
            config = CslLibsAllreduceDataProcessorConfig(**config)

        self.config = config
        self.batch_size = get_streaming_batch_size(config.batch_size)
        self.dtype = cstorch.amp.get_floating_point_dtype()

    def create_dataloader(self):
        cfg = self.config
        values = torch.arange(
            self.batch_size
            * cfg.grid_height
            * cfg.grid_width
            * cfg.vector_length,
            dtype=torch.float32,
        ).reshape(
            self.batch_size,
            cfg.grid_height,
            cfg.grid_width,
            cfg.vector_length,
        )
        values = ((values % 23.0) - 7.0) / 11.0
        values = values.to(self.dtype)

        if cfg.op == "add":
            reduced = values.sum(dim=(1, 2), keepdim=True)
        else:
            reduced = values.amax(dim=(1, 2), keepdim=True)
        reference = reduced.expand_as(values)

        return SampleGenerator(
            {"input": values, "reference": reference},
            cfg.sample_count,
        )
