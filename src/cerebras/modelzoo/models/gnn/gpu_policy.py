"""Configuration shared by native PyTorch GNN runners."""

import math

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
    # Match the Model Zoo comparison profiles when the YAML omits epsilon.
    kwargs.setdefault("eps", 1e-6)
    kwargs.setdefault("betas", (0.9, 0.999))
    kwargs.setdefault("weight_decay", 0.0)
    if kwargs.pop("params", None):
        raise ValueError("custom optimizer parameter groups are unsupported")
    if "learning_rate" in kwargs:
        if "lr" in kwargs:
            raise ValueError("Specify only one of learning_rate and lr")
        kwargs["lr"] = kwargs.pop("learning_rate")
    return kwargs


def adamw_param_groups(model):
    """Use native parameter groups to exclude biases/norms from weight decay.

    GraphSAGE has no normalization layers, but handle native normalization
    modules too so the policy is explicit for the other GPU architectures.
    Never import the Model Zoo SDK optimizer or reproduce its update equation.
    """
    norms = (
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
    )
    no_decay_ids = {
        id(p)
        for module in model.modules()
        if isinstance(module, norms)
        for p in module.parameters(recurse=False)
    }
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            target = (
                no_decay
                if name.endswith(".bias")
                or name == "bias"
                or id(parameter) in no_decay_ids
                else decay
            )
            target.append(parameter)
    return ([{"params": decay}] if decay else []) + (
        [{"params": no_decay, "weight_decay": 0.0}] if no_decay else []
    )


def grad_scaler_kwargs(init, dtype):
    """Resolve supported native AMP knobs; do not emulate SDK scale clamps."""
    precision = init.get("precision") or {}
    for key in ("max_gradient_norm", "max_gradient_value"):
        if precision.get(key) is not None:
            raise ValueError(f"Native GPU policy does not implement precision.{key}")
    if dtype != torch.float16:
        return {"enabled": False}
    for key in ("min_loss_scale", "max_loss_scale"):
        if precision.get(key) is not None:
            raise ValueError(f"Native GPU policy does not implement precision.{key}")
    scale = precision.get("loss_scaling_factor", "dynamic")
    if scale != "dynamic":
        raise ValueError("Native FP16 training requires dynamic loss_scaling_factor")
    initial = precision.get("initial_loss_scale")
    initial = 32768.0 if initial is None else initial
    interval = precision.get("steps_per_increase", 2000)
    if (
        isinstance(initial, bool)
        or not isinstance(initial, (int, float))
        or not math.isfinite(initial)
        or initial <= 0
    ):
        raise ValueError("initial_loss_scale must be finite and positive")
    if type(interval) is not int or interval <= 0:
        raise ValueError("steps_per_increase must be a positive integer")
    return dict(
        enabled=True,
        init_scale=float(initial),
        growth_interval=interval,
        growth_factor=2.0,
        backoff_factor=0.5,
    )


def optimizer_policy(model, optimizer, scaler_kwargs):
    """Record effective parameter groups, not only optimizer-wide defaults."""
    names = {id(p): name for name, p in model.named_parameters()}
    return dict(
        version=1,
        optimizer="torch.optim.AdamW",
        epsilon_convention="sqrt_bias_corrected_second_moment_plus_eps",
        optimizer_defaults=optimizer.defaults,
        optimizer_groups=[
            {
                "parameters": [names[id(p)] for p in group["params"]],
                "weight_decay": group["weight_decay"],
            }
            for group in optimizer.param_groups
        ],
        parameter_count=sum(p.numel() for p in model.parameters()),
        amp={
            "implementation": "torch.amp.GradScaler",
            **scaler_kwargs,
            "sdk_scale_clamps": False,
        },
    )


class OptimizerStepCounter:
    """Count real updates for native foreach and fused AMP optimizers.

    Fused AdamW is called even on overflow, and receives found_inf on device.
    Accumulate that flag without forcing an extra host sync at every step.
    Read `steps` only at an existing synchronized logging boundary.
    """

    def __init__(self, optimizer):
        self.calls = 0
        self.fused_skips = None
        self.handle = optimizer.register_step_post_hook(self._completed)

    def _completed(self, optimizer, *_):
        self.calls += 1
        found_inf = getattr(optimizer, "found_inf", None)
        if found_inf is not None:
            if self.fused_skips is None:
                self.fused_skips = torch.zeros_like(found_inf)
            self.fused_skips.add_(found_inf.ne(0))

    @property
    def steps(self):
        return self.calls - (
            int(self.fused_skips.item()) if self.fused_skips is not None else 0
        )
