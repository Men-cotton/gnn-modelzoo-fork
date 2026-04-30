from typing import Literal

import torch

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class SdkBenchmarkWideMultiplicationDataProcessorConfig(DataConfig):
    data_processor: Literal["SdkBenchmarkWideMultiplicationDataProcessor"]
    batch_size: int = 1
    sample_count: int = 4
    operand_bits: int = 128
    result_bits: int = 256


def _int_to_bits(value: int, bit_count: int):
    return torch.tensor(
        [(value >> bit_idx) & 1 for bit_idx in range(bit_count)],
        dtype=torch.float32,
    )


def _int_to_words(value: int, word_count: int):
    return torch.tensor(
        [
            (value >> (16 * word_idx)) & 0xFFFF
            for word_idx in range(word_count)
        ],
        dtype=torch.float32,
    )


class SdkBenchmarkWideMultiplicationDataProcessor:
    def __init__(
        self, config: SdkBenchmarkWideMultiplicationDataProcessorConfig
    ):
        if isinstance(config, dict):
            config = SdkBenchmarkWideMultiplicationDataProcessorConfig(**config)
        self.config = config
        self.batch_size = get_streaming_batch_size(config.batch_size)

    def create_dataloader(self):
        left = (0x5EED123456789ABC << 64) | 0x13579BDF2468ACE0
        right = (0x1234FEDCBA987654 << 64) | 0x0FEDCBA987654321
        x_bits = _int_to_bits(left, self.config.operand_bits)
        y_bits = _int_to_bits(right, self.config.operand_bits)
        reference_words = _int_to_words(
            left * right, self.config.result_bits // 16
        )

        data = {
            "x_bits": x_bits.unsqueeze(0).repeat(self.batch_size, 1),
            "y_bits": y_bits.unsqueeze(0).repeat(self.batch_size, 1),
            "reference_words": reference_words.unsqueeze(0).repeat(
                self.batch_size, 1
            ),
        }
        return SampleGenerator(data, self.config.sample_count)
