from typing import Literal

import torch
import torch.nn as nn
from annotated_types import Ge
from typing_extensions import Annotated

from cerebras.modelzoo.config import ModelConfig


class SdkBenchmarkHistogramTorusModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/histogram-torus"]
    """Model registry name."""

    hist_height: Annotated[int, Ge(1)] = 4
    hist_width: Annotated[int, Ge(1)] = 4
    input_size: Annotated[int, Ge(1)] = 8
    num_buckets: Annotated[int, Ge(1)] = 5
    bucket_size: Annotated[int, Ge(1)] = 10

    @property
    def __model_cls__(self):
        return SdkBenchmarkHistogramTorusModel


class SdkBenchmarkHistogramTorusModel(nn.Module):
    """Static tensor analogue of the CSL torus-routed histogram.

    The CSL benchmark routes encoded wavelets to the PE that owns each bucket.
    PyTorch cannot express that per-PE packet routing, so this model implements
    the same bucket ownership calculation as fixed-shape tensor comparisons.
    """

    def __init__(self, config: SdkBenchmarkHistogramTorusModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = SdkBenchmarkHistogramTorusModelConfig(**config)
        super().__init__()
        self.bucket_size = config.bucket_size
        global_bucket_ids = torch.arange(
            config.hist_height * config.hist_width * config.num_buckets,
            dtype=torch.int64,
        ).reshape(1, 1, 1, 1, config.hist_height, config.hist_width, config.num_buckets)
        self.register_buffer("global_bucket_ids", global_bucket_ids)

    def forward(self, batch):
        inputs, reference = batch
        global_bucket = inputs.to(torch.int64).reshape(
            inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3], 1, 1, 1
        ) // self.bucket_size
        matches = global_bucket == self.global_bucket_ids
        output = matches.to(torch.float32).sum(dim=(1, 2, 3))
        error = output - reference
        loss = (error * error).mean()
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "total_count": output.sum(),
            "max_abs_error": error.abs().amax(),
        }
