from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.config import ModelConfig


class SdkBenchmarkGameOfLifeModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/game-of-life"]
    """Model registry name."""

    height: Annotated[int, Ge(4)] = 16
    width: Annotated[int, Ge(4)] = 16
    generations: Annotated[int, Ge(1)] = 8

    @property
    def __model_cls__(self):
        return SdkBenchmarkGameOfLifeModel


class SdkBenchmarkGameOfLifeModel(nn.Module):
    """Fixed-shape Conway's Game of Life stencil.

    The CSL sample assigns one cell to each PE and exchanges neighbor state
    through fabric routes. This PyTorch analogue captures the cellular-automaton
    update rule, but not PE-local sends, forwards, or route timing.
    """

    def __init__(self, config: SdkBenchmarkGameOfLifeModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = SdkBenchmarkGameOfLifeModelConfig(**config)
        super().__init__()
        self.generations = config.generations

    def _step(self, state):
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

    def forward(self, batch):
        initial_state, reference = batch
        state = initial_state
        for _ in range(self.generations):
            state = self._step(state)

        error = state - reference
        loss = (error * error).mean()
        return {
            "loss": loss,
            "output": state,
            "reference": reference,
            "live_cells": state.sum(),
            "max_abs_error": error.abs().amax(),
        }
