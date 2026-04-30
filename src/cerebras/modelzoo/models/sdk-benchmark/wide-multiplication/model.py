from typing import Literal

import torch
from torch import nn

from cerebras.modelzoo.config import ModelConfig


class WideMultiplicationBenchmarkModelConfig(ModelConfig):
    name: Literal["sdk_benchmark/wide-multiplication"]
    operand_bits: int = 128
    result_bits: int = 256

    @property
    def __model_cls__(self):
        return WideMultiplicationBenchmarkModel


class WideMultiplicationKernel(nn.Module):
    def __init__(self, operand_bits: int, result_bits: int):
        super().__init__()
        self.operand_bits = operand_bits
        self.result_bits = result_bits
        self.result_words = result_bits // 16

        x_positions = torch.arange(operand_bits).reshape(1, operand_bits, 1)
        y_positions = torch.arange(operand_bits).reshape(1, 1, operand_bits)
        result_positions = torch.arange(result_bits).reshape(result_bits, 1, 1)
        mask = (x_positions + y_positions) == result_positions
        self.register_buffer("partial_mask", mask.to(torch.float32))
        self.register_buffer(
            "word_weights",
            (2.0 ** torch.arange(16, dtype=torch.float32)).reshape(1, 1, 16),
        )

    def forward(self, x_bits, y_bits):
        partials = x_bits.unsqueeze(2) * y_bits.unsqueeze(1)
        coeffs = (partials.unsqueeze(1) * self.partial_mask.unsqueeze(0)).sum(
            dim=(-1, -2)
        )

        carry = torch.zeros_like(coeffs[:, 0])
        result_bits = []
        for bit_idx in range(self.result_bits):
            total = coeffs[:, bit_idx] + carry
            next_carry = torch.floor(total * 0.5)
            result_bit = total - 2.0 * next_carry
            result_bits.append(result_bit)
            carry = next_carry

        bits = torch.stack(result_bits, dim=1)
        words = bits.reshape(bits.shape[0], self.result_words, 16)
        return (words * self.word_weights).sum(dim=2)


class WideMultiplicationBenchmarkModel(nn.Module):
    def __init__(self, config: WideMultiplicationBenchmarkModelConfig):
        if isinstance(config, dict):
            config = config.get("model", config)
            config = WideMultiplicationBenchmarkModelConfig(**config)

        super().__init__()
        self.kernel = WideMultiplicationKernel(
            config.operand_bits, config.result_bits
        )

    def forward(self, batch):
        output = self.kernel(batch["x_bits"], batch["y_bits"])
        reference = batch["reference_words"]
        error = (output - reference).abs()
        loss = error.float().mean()
        return {
            "loss": loss,
            "output": output,
            "reference": reference,
            "max_error": error.max(),
        }
