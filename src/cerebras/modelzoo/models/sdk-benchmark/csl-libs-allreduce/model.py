from typing import Literal

import torch
import torch.nn as nn

from cerebras.modelzoo.config import ModelConfig


class CslLibsAllreduceConfig(ModelConfig):
    name: Literal["sdk_benchmark/csl-libs-allreduce"]
    grid_height: int = 4
    grid_width: int = 4
    vector_length: int = 16
    op: Literal["add", "max"] = "add"

    @property
    def __model_cls__(self):
        return CslLibsAllreduceModel


class CslLibsAllreduceModel(nn.Module):
    def __init__(self, config: CslLibsAllreduceConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = CslLibsAllreduceConfig(**config)

        super().__init__()
        self.grid_height = config.grid_height
        self.grid_width = config.grid_width
        self.vector_length = config.vector_length
        self.op = config.op

    def forward(self, batch):
        values = batch["input"]
        reference = batch["reference"]

        if self.op == "add":
            reduced = values.sum(dim=(1, 2), keepdim=True)
        else:
            reduced = values.amax(dim=(1, 2), keepdim=True)
        output = reduced.expand_as(values)

        error = (output - reference).abs()
        loss = (error * error).mean()
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_error": error.max(),
        }
