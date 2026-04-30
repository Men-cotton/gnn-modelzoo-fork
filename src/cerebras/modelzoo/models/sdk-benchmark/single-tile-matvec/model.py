from typing import Literal

import torch
import torch.nn as nn

from cerebras.modelzoo.config import ModelConfig


class SingleTileMatvecModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/single-tile-matvec"] = (
        "sdk_benchmark/single-tile-matvec"
    )
    width: int = 2
    height: int = 2
    tile_size: int = 25
    iters: int = 1

    @property
    def __model_cls__(self):
        return SingleTileMatvecModel


class SingleTileMatvecModel(nn.Module):
    def __init__(self, config: SingleTileMatvecModelConfig):
        if isinstance(config, dict):
            config = SingleTileMatvecModelConfig(**config)
        super().__init__()
        self.iters = config.iters

    def forward(self, batch):
        a = batch["A"]
        x = batch["x"]
        reference = batch["reference"]

        single_iter = torch.matmul(a, x.unsqueeze(-1)).squeeze(-1)
        output = torch.zeros_like(single_iter)
        for _ in range(self.iters):
            output = output + single_iter

        error = output - reference
        squared_error = error * error
        loss = squared_error.mean()
        max_abs_error = torch.amax(torch.abs(error))
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_abs_error": max_abs_error,
        }
