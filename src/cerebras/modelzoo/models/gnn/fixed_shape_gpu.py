"""Train ModelZoo's fixed-shape GraphSAGE batches with native PyTorch on one GPU."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from cerebras.modelzoo.common.utils.run.config_loader import load_params_file
from cerebras.modelzoo.models.gnn.data_processing.batches import GraphSAGEBatch
from cerebras.modelzoo.models.gnn.data_processing.padding import NeighborPaddingStats
from cerebras.modelzoo.models.gnn.data_processing.processor import (
    GNNDataProcessorConfig,
)
from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
    NeighborSamplingDataProcessor,
)
from cerebras.modelzoo.models.gnn.model import GNNModel


def make_loader(config, *, num_layers, float_dtype):
    """Use the ModelZoo source, sampler, cache and collation without an SDK executor."""
    config = GNNDataProcessorConfig(**config)
    if config.sampling_mode != "neighbor" or config.use_fake_data:
        raise ValueError(
            "fixed_shape_gpu requires real neighbor-sampled GraphSAGE data"
        )
    if (
        not config.fanouts
        or len(config.fanouts) != num_layers
        or any(fanout <= 0 for fanout in config.fanouts)
    ):
        raise ValueError("fanouts must contain one positive value per GraphSAGE layer")
    processor = NeighborSamplingDataProcessor(
        dataset_name=config.dataset_name,
        data_dir=config.data_dir,
        current_split=config.split or "train",
        float_dtype=float_dtype,
        label_dtype=torch.long,
        adj_normalization_fn=None,
        fanouts=config.fanouts,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        sampler_seed=config.sampler_seed,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        pad_id=config.pad_node_id,
        cache_fraction=config.cache_fraction,
        static_batch_cache_size=config.static_batch_cache_size,
    )
    return processor.create_torch_dataloader()


def precision_dtype(init, override=None):
    if override is not None:
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
            override
        ]
    precision = init.get("precision", {})
    if not precision.get("enabled", True):
        return torch.float32
    kind = precision.get("fp16_type", "float16")
    if kind not in ("float16", "bfloat16"):
        raise ValueError(f"GPU precision {kind!r} is unsupported; select --precision")
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[kind]


@torch.no_grad()
def evaluate(model, loader, device, dtype):
    """Count only real targets, including the padded final validation batch."""
    model.eval()
    correct = torch.zeros((), device=device, dtype=torch.long)
    targets = 0
    for payload in loader:
        targets += int(payload["target_mask"].sum())
        batch = GraphSAGEBatch.from_payload(payload).to(device, non_blocking=True)
        param = next(model.parameters())
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            adapted = model.architecture_spec.adapt_batch(
                batch, device, param.dtype, model.architecture_spec.name
            )
            logits = model.model(*adapted.model_args)
            correct += (
                (logits.argmax(-1) == adapted.labels) & adapted.target_mask
            ).sum()
    model.train()
    return {"accuracy": int(correct) / targets, "targets": targets}


def train(
    cfg,
    output_dir,
    *,
    device,
    dtype,
    compile_model=False,
    warmup_steps=40,
    measure_neighbor_padding=False,
):
    """Run synchronized training windows; exclude setup, warmup and evaluation."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("fixed_shape_gpu supports one process and one GPU")
    cfg = copy.deepcopy(cfg)
    init = cfg["trainer"]["init"]
    fit = cfg["trainer"]["fit"]
    model_cfg = init["model"]
    if model_cfg["architecture"]["type"].lower() != "graphsage":
        raise ValueError("fixed_shape_gpu supports the GraphSAGE architecture")
    loop = init["loop"]
    max_steps = int(loop["max_steps"])
    log_steps = int(init.get("logging", {}).get("log_steps", 20))
    if max_steps <= warmup_steps or warmup_steps < 0 or log_steps <= 0:
        raise ValueError("require max_steps > warmup_steps >= 0 and log_steps > 0")
    if loop.get("grad_accum_steps", 1) != 1 or init.get("schedulers"):
        raise ValueError("gradient accumulation and schedulers are not supported")
    eval_frequency = loop.get("eval_frequency")
    if eval_frequency is not None and (
        isinstance(eval_frequency, bool)
        or not isinstance(eval_frequency, int)
        or eval_frequency <= 0
    ):
        raise ValueError("eval_frequency must be a positive step count or null")
    compute_metrics = model_cfg.get("task", {}).get("compute_eval_metrics", True)
    if (
        compute_metrics
        and eval_frequency
        and fit["val_dataloader"].get("static_batch_cache_size", 0)
    ):
        raise ValueError("validation cannot repeat cached static batches")
    optimizer_cfg = copy.deepcopy(init["optimizer"])
    if set(optimizer_cfg) != {"AdamW"}:
        raise ValueError("fixed_shape_gpu currently supports AdamW only")
    adamw = optimizer_cfg["AdamW"]
    if adamw.pop("params", []):
        raise ValueError("custom optimizer parameter groups are unsupported")
    adamw["lr"] = adamw.pop("learning_rate")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    with (output_dir / "resolved_config.yaml").open("x") as stream:
        yaml.safe_dump(cfg, stream)

    seed = int(init.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader_args = dict(
        num_layers=model_cfg["architecture"]["num_layers"], float_dtype=dtype
    )
    train_loader = make_loader(fit["train_dataloader"], **loader_args)
    val_loader = (
        make_loader(fit["val_dataloader"], **loader_args)
        if compute_metrics and eval_frequency
        else None
    )
    # Native accuracy counting does not need the SDK's metric/backend state.
    native_model_cfg = copy.deepcopy(model_cfg)
    native_model_cfg.setdefault("task", {})["compute_eval_metrics"] = False
    model = GNNModel(native_model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **adamw)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    training_model = torch.compile(model) if compile_model else model
    model.train()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def emit(record):
        line = json.dumps(record, allow_nan=False)
        print(line, flush=True)
        metrics.write(line + "\n")
        metrics.flush()

    iterator = iter(train_loader)
    measured_seconds = 0.0
    measured_targets = measured_slots = measured_steps = 0
    window_targets = window_slots = window_steps = 0
    window_padding = NeighborPaddingStats() if measure_neighbor_padding else None
    measured_padding = NeighborPaddingStats() if measure_neighbor_padding else None
    loss_sum = torch.zeros((), device=device)
    all_finite = torch.ones((), device=device, dtype=torch.bool)
    with (output_dir / "metrics.jsonl").open("x") as metrics:
        emit(
            {
                "event": "run",
                "backend": "fixed_shape_gpu",
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "cpu"
                ),
                "torch_version": torch.__version__,
                "precision": str(dtype),
                "compile": compile_model,
                "warmup_steps": warmup_steps,
                "measure_neighbor_padding": measure_neighbor_padding,
                "cache_device": "cpu",
                "static_batch_cache_size": fit["train_dataloader"].get(
                    "static_batch_cache_size", 0
                ),
            }
        )
        synchronize()
        window_start = time.perf_counter()
        for step in range(1, max_steps + 1):
            try:
                payload = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                payload = next(iterator)
            window_targets += int(payload["target_mask"].sum())
            window_slots += payload["target_mask"].numel()
            window_steps += 1
            if measure_neighbor_padding:
                window_padding.update(payload)
            batch = GraphSAGEBatch.from_payload(payload).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type, dtype=dtype, enabled=dtype != torch.float32
            ):
                loss = training_model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach()
            all_finite &= torch.isfinite(loss.detach())
            do_eval = val_loader is not None and (
                step % eval_frequency == 0 or step == max_steps
            )
            if step % log_steps == 0 or step in (warmup_steps, max_steps) or do_eval:
                synchronize()
                elapsed = time.perf_counter() - window_start
                if not bool(all_finite):
                    raise FloatingPointError(f"non-finite training loss by step {step}")
                measured = step > warmup_steps
                if measured:
                    measured_seconds += elapsed
                    measured_targets += window_targets
                    measured_slots += window_slots
                    measured_steps += window_steps
                    if measure_neighbor_padding:
                        measured_padding.merge(window_padding)
                emit(
                    {
                        "event": "train",
                        "step": step,
                        "measured": measured,
                        "loss": float(loss_sum) / window_steps,
                        "seconds": elapsed,
                        "steps": window_steps,
                        "seed_nodes": window_targets,
                        "nominal_slots": window_slots,
                        "seed_nodes_per_second": window_targets / elapsed,
                        "nominal_slots_per_second": window_slots / elapsed,
                        **(
                            {"neighbor_padding": window_padding.summary()}
                            if measure_neighbor_padding
                            else {}
                        ),
                    }
                )
                if do_eval:
                    emit(
                        {
                            "event": "eval",
                            "step": step,
                            **evaluate(model, val_loader, device, dtype),
                        }
                    )
                window_targets = window_slots = window_steps = 0
                if measure_neighbor_padding:
                    window_padding = NeighborPaddingStats()
                loss_sum.zero_()
                synchronize()
                window_start = time.perf_counter()
        summary = {
            "event": "summary",
            "steps": measured_steps,
            "seed_nodes": measured_targets,
            "nominal_slots": measured_slots,
            "seconds": measured_seconds,
            "seed_nodes_per_second": measured_targets / measured_seconds,
            "nominal_slots_per_second": measured_slots / measured_seconds,
        }
        if measure_neighbor_padding:
            summary["neighbor_padding"] = measured_padding.summary()
        emit(summary)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": max_steps,
        },
        output_dir / "checkpoint.pt",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="cpu is for correctness smoke tests only",
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--measure-neighbor-padding",
        action="store_true",
        help="count neighbor padding on the CPU and include it in metrics (default: off)",
    )
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; no automatic CPU fallback")
    cfg = load_params_file(args.config)
    if args.max_steps is not None:
        cfg["trainer"]["init"]["loop"]["max_steps"] = args.max_steps
    if args.num_workers is not None:
        for name in ("train_dataloader", "val_dataloader"):
            if name in cfg["trainer"]["fit"]:
                cfg["trainer"]["fit"][name]["num_workers"] = args.num_workers
    dtype = precision_dtype(cfg["trainer"]["init"], args.precision)
    if device.type == "cpu" and dtype == torch.float16:
        parser.error("CPU smoke tests require --precision fp32 or bf16")
    train(
        cfg,
        args.output_dir,
        device=device,
        dtype=dtype,
        compile_model=args.compile,
        warmup_steps=args.warmup_steps,
        measure_neighbor_padding=args.measure_neighbor_padding,
    )


if __name__ == "__main__":
    main()
