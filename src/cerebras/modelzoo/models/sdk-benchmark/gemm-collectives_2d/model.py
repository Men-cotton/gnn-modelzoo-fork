from typing import Literal

import torch
import torch.nn as nn

from cerebras.modelzoo.config import ModelConfig


class GemmCollectives2dModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/gemm-collectives_2d"] = (
        "sdk_benchmark/gemm-collectives_2d"
    )
    p: int = 4
    mt: int = 14
    kt: int = 14
    nt: int = 14

    @property
    def __model_cls__(self):
        return GemmCollectives2dModel


class GemmCollectives2dModel(nn.Module):
    def __init__(self, config: GemmCollectives2dModelConfig):
        if isinstance(config, dict):
            config = GemmCollectives2dModelConfig(**config)
        super().__init__()
        self.p = config.p
        self.mt = config.mt
        self.kt = config.kt
        self.nt = config.nt
        self.m = config.p * config.mt
        self.n = config.p * config.nt

    def forward(self, batch):
        a = batch["A"]
        b = batch["B"]
        reference = batch["reference"]

        output = torch.zeros_like(reference)
        for step in range(self.p):
            k_start = step * self.kt
            k_end = k_start + self.kt
            output = output + torch.matmul(
                a[:, :, k_start:k_end], b[:, k_start:k_end, :]
            )

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
