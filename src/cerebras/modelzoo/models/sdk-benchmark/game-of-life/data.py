from typing import Literal

import torch
import torch.nn.functional as F
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class SdkBenchmarkGameOfLifeDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmarkGameOfLifeDataProcessor"]
    """Name of the data processor."""

    batch_size: Annotated[int, Ge(1)] = 1
    height: Annotated[int, Ge(4)] = 16
    width: Annotated[int, Ge(4)] = 16
    generations: Annotated[int, Ge(1)] = 8
    sample_count: Annotated[int, Ge(1)] = 16


def _life_step(state):
    padded = F.pad(state, (1, 1, 1, 1))
    neighbors = (
        padded[:, :-2, :-2]
        + padded[:, :-2, 1:-1]
        + padded[:, :-2, 2:]
        + padded[:, 1:-1, :-2]
        + padded[:, 1:-1, 2:]
        + padded[:, 2:, :-2]
        + padded[:, 2:, 1:-1]
        + padded[:, 2:, 2:]
    )
    alive = state > 0.5
    survives = alive & ((neighbors == 2.0) | (neighbors == 3.0))
    born = (~alive) & (neighbors == 3.0)
    return torch.where(
        survives | born, torch.ones_like(state), torch.zeros_like(state)
    )


class SdkBenchmarkGameOfLifeDataProcessor:
    def __init__(self, config: SdkBenchmarkGameOfLifeDataProcessorConfig):
        if isinstance(config, dict):
            config = SdkBenchmarkGameOfLifeDataProcessorConfig(**config)
        self.config = config
        self.batch_size = get_streaming_batch_size(config.batch_size)

    def create_dataloader(self):
        cfg = self.config
        rows = torch.arange(cfg.height, dtype=torch.float32).reshape(1, cfg.height, 1)
        cols = torch.arange(cfg.width, dtype=torch.float32).reshape(1, 1, cfg.width)
        batch_offsets = torch.arange(self.batch_size, dtype=torch.float32).reshape(
            self.batch_size, 1, 1
        )
        initial_state = (
            ((rows * 3.0 + cols * 5.0 + batch_offsets) % 7.0) < 3.0
        ).to(torch.float32)
        reference = initial_state
        for _ in range(cfg.generations):
            reference = _life_step(reference)
        return SampleGenerator(
            data=(initial_state, reference), sample_count=cfg.sample_count
        )
