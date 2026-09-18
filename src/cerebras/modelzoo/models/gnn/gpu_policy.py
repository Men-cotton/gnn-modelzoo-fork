"""Configuration shared by native PyTorch GNN runners."""

import torch


def precision_dtype(init, override=None, *, default=torch.float16):
    if override is not None:
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
            override
        ]
    precision = init.get("precision")
    if precision is None:
        return default
    if not precision.get("enabled", True):
        return torch.float32
    kind = precision.get("fp16_type", "float16")
    if kind not in ("float16", "bfloat16"):
        raise ValueError(f"GPU precision {kind!r} is unsupported; select a GPU dtype")
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[kind]


def adamw_kwargs(init):
    config = init["optimizer"]
    if set(config) != {"AdamW"}:
        raise ValueError("Native GPU training currently supports AdamW only")
    kwargs = dict(config["AdamW"])
    if kwargs.pop("params", None):
        raise ValueError("custom optimizer parameter groups are unsupported")
    if "learning_rate" in kwargs:
        if "lr" in kwargs:
            raise ValueError("Specify only one of learning_rate and lr")
        kwargs["lr"] = kwargs.pop("learning_rate")
    return kwargs
