from typing import Literal

import torch
import torch.nn as nn
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.config import ModelConfig


class SdkBenchmarkBandwidthTestModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/bandwidth-test"]
    """Model registry name."""

    tensor_length: Annotated[int, Ge(1)] = 4096
    """Number of values in the static payload tensor."""

    loop_count: Annotated[int, Ge(1)] = 8
    """Fixed number of repeated tensor passes."""

    channel_count: Annotated[int, Ge(1)] = 4
    """Static channel count used to vary the deterministic transform."""

    @property
    def __model_cls__(self):
        return SdkBenchmarkBandwidthTestModel


class SdkBenchmarkBandwidthTestModel(nn.Module):
    """Static tensor analogue of the CSL host/device bandwidth test.

    The CSL benchmark times explicit H2D/D2H transfers with PE timestamp
    counters. PyTorch exposes neither those APIs nor routing timestamps, so this
    model measures a fixed-shape memory-traffic-like tensor transform instead.
    """

    def __init__(self, config: SdkBenchmarkBandwidthTestModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = SdkBenchmarkBandwidthTestModelConfig(**config)
        super().__init__()
        self.loop_count = config.loop_count
        scale = 1.0 + (0.001 * float(config.channel_count))
        bias = 0.0001 * float(config.channel_count)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

    def forward(self, batch):
        payload, reference = batch
        output = payload
        for _ in range(self.loop_count):
            output = output * self.scale + self.bias

        error = output - reference
        loss = (error * error).mean()
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_abs_error": error.abs().amax(),
        }
