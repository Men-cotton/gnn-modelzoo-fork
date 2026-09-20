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
from cerebras.modelzoo.models.gnn.gpu_policy import (
    adamw_kwargs,
    adamw_param_groups,
    grad_scaler_kwargs,
    optimizer_policy,
    precision_dtype,
    OptimizerStepCounter,
)
from cerebras.modelzoo.models.gnn.gpu_measurements import (
    logical_tensor_bytes,
    save_provenance,
)


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
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        pad_id=config.pad_node_id,
        cache_fraction=config.cache_fraction,
        static_batch_cache_size=config.static_batch_cache_size,
        measure_batch_accounting=True,
        worker_diagnostics=config.worker_diagnostics,
    )
    return processor.create_torch_dataloader()


@torch.no_grad()
def evaluate(model, loader, device, dtype):
    """Count only real targets, including the padded final validation batch."""
    model.eval()
    correct = torch.zeros((), device=device, dtype=torch.long)
    targets = 0
    for payload in loader:
        batch = GraphSAGEBatch.from_payload(payload).to(device, non_blocking=True)
        param = next(model.parameters())
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            adapted = model.architecture_spec.adapt_batch(
                batch, device, param.dtype, model.architecture_spec.name
            )
            logits = model.model(*adapted.model_args)
            mask = adapted.target_mask & (adapted.labels != -100)
            targets += int(mask.sum())
            correct += ((logits.argmax(-1) == adapted.labels) & mask).sum()
    model.train()
    return {"accuracy": int(correct) / max(targets, 1), "targets": targets}


def train(
    cfg,
    output_dir,
    *,
    device,
    dtype,
    compile_model=False,
    warmup_steps=40,
    measure_neighbor_padding=False,
    measure_input=False,
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
    adamw = adamw_kwargs(init)

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
    optimizer = torch.optim.AdamW(adamw_param_groups(model), **adamw)
    save_provenance(output_dir, optimizer=optimizer, device=device)
    update_counter = OptimizerStepCounter(optimizer)
    scaler_options = grad_scaler_kwargs(init, dtype)
    scaler = torch.amp.GradScaler(device.type, **scaler_options)
    with (output_dir / "gpu_policy.json").open("x") as stream:
        json.dump(optimizer_policy(model, optimizer, scaler_options), stream, indent=2)
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
    measured_supervised = measured_optimizer_steps = 0
    window_targets = window_slots = window_steps = 0
    window_supervised = window_optimizer_start = 0
    window_loader_seconds = 0.0
    window_payload_bytes = 0
    measured_loader_seconds = 0.0
    measured_payload_bytes = 0
    measured_copy_seconds = measured_device_step_seconds = 0.0
    device_events = []
    window_padding = NeighborPaddingStats() if measure_neighbor_padding else None
    measured_padding = NeighborPaddingStats() if measure_neighbor_padding else None
    loss_sum = torch.zeros((), device=device)
    all_finite = torch.ones((), device=device, dtype=torch.bool)
    with (output_dir / "metrics.jsonl").open("x") as metrics:
        emit(
            {
                "event": "run",
                "backend": "fixed_shape_gpu",
                "input_contract": getattr(train_loader, "gnn_input_contract", None),
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
                "measure_input": measure_input,
                "window_definition": "Synchronized training windows include loader, transfer, forward/backward and optimizer work; exclude logging, evaluation and checkpoint writing. Exact endpoint timing is available for comparisons including logging.",
                "cache_device": "cpu",
                "static_batch_cache_size": fit["train_dataloader"].get(
                    "static_batch_cache_size", 0
                ),
            }
        )
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        window_start = time.perf_counter()
        for step in range(1, max_steps + 1):
            if measure_input:
                loader_start = time.perf_counter()
            try:
                payload = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                payload = next(iterator)
            if measure_input:
                window_loader_seconds += time.perf_counter() - loader_start
                window_payload_bytes += logical_tensor_bytes(payload)
            window_targets += int(payload["target_mask"].sum())
            window_supervised += int(
                (payload["target_mask"] & (payload["labels"] != -100)).sum()
            )
            window_slots += payload["target_mask"].numel()
            window_steps += 1
            if measure_neighbor_padding:
                window_padding.update(payload)
            if measure_input and device.type == "cuda":
                events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
                events[0].record()
            batch = GraphSAGEBatch.from_payload(payload).to(device, non_blocking=True)
            if measure_input and device.type == "cuda":
                events[1].record()
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
            if measure_input and device.type == "cuda":
                events[2].record()
                device_events.append(events)
            do_eval = val_loader is not None and (
                step % eval_frequency == 0 or step == max_steps
            )
            if step % log_steps == 0 or step in (warmup_steps, max_steps) or do_eval:
                synchronize()
                boundary = time.perf_counter()
                elapsed = boundary - window_start
                optimizer_steps = update_counter.steps
                window_optimizer_steps = optimizer_steps - window_optimizer_start
                input_metrics = {}
                if measure_input:
                    copy_seconds = (
                        sum(a.elapsed_time(b) for a, b, _ in device_events) / 1000
                    )
                    device_step_seconds = (
                        sum(b.elapsed_time(c) for _, b, c in device_events) / 1000
                    )
                    input_metrics = {
                        "loader_wait_seconds": window_loader_seconds,
                        "logical_payload_bytes": window_payload_bytes,
                        "logical_payload_bytes_per_second": window_payload_bytes
                        / elapsed,
                    }
                    if device.type == "cuda":
                        input_metrics.update(
                            cuda_copy_stream_seconds=copy_seconds,
                            cuda_step_stream_seconds=device_step_seconds,
                        )
                if not bool(all_finite):
                    raise FloatingPointError(f"non-finite training loss by step {step}")
                measured = step > warmup_steps
                if measured:
                    measured_seconds += elapsed
                    measured_targets += window_targets
                    measured_slots += window_slots
                    measured_steps += window_steps
                    measured_supervised += window_supervised
                    measured_optimizer_steps += window_optimizer_steps
                    if measure_input:
                        measured_loader_seconds += window_loader_seconds
                        measured_payload_bytes += window_payload_bytes
                        measured_copy_seconds += copy_seconds
                        measured_device_step_seconds += device_step_seconds
                    if measure_neighbor_padding:
                        measured_padding.merge(window_padding)
                emit(
                    {
                        "event": "train",
                        "step": step,
                        "start_step": step - window_steps,
                        "end_step": step,
                        "boundary_monotonic_seconds": boundary,
                        "measured": measured,
                        "loss": float(loss_sum) / window_steps,
                        "seconds": elapsed,
                        "steps": window_steps,
                        "seed_nodes": window_targets,
                        "supervised_targets": window_supervised,
                        "optimizer_steps": window_optimizer_steps,
                        "skipped_optimizer_steps": window_steps
                        - window_optimizer_steps,
                        "nominal_slots": window_slots,
                        "seed_nodes_per_second": window_targets / elapsed,
                        "nominal_slots_per_second": window_slots / elapsed,
                        "supervised_targets_per_second": window_supervised / elapsed,
                        **input_metrics,
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
                window_supervised = 0
                window_optimizer_start = optimizer_steps
                window_loader_seconds = 0.0
                window_payload_bytes = 0
                device_events = []
                if measure_neighbor_padding:
                    window_padding = NeighborPaddingStats()
                loss_sum.zero_()
                synchronize()
                window_start = time.perf_counter()
        summary = {
            "event": "summary",
            "steps": measured_steps,
            "seed_nodes": measured_targets,
            "supervised_targets": measured_supervised,
            "optimizer_steps": measured_optimizer_steps,
            "skipped_optimizer_steps": measured_steps - measured_optimizer_steps,
            "nominal_slots": measured_slots,
            "seconds": measured_seconds,
            "seed_nodes_per_second": measured_targets / measured_seconds,
            "nominal_slots_per_second": measured_slots / measured_seconds,
            "supervised_targets_per_second": measured_supervised / measured_seconds,
            "start_step": warmup_steps,
            "end_step": max_steps,
        }
        if measure_input:
            summary.update(
                loader_wait_seconds=measured_loader_seconds,
                logical_payload_bytes=measured_payload_bytes,
                logical_payload_bytes_per_second=measured_payload_bytes
                / measured_seconds,
            )
            if device.type == "cuda":
                summary.update(
                    cuda_copy_stream_seconds=measured_copy_seconds,
                    cuda_step_stream_seconds=measured_device_step_seconds,
                )
        if device.type == "cuda":
            summary["cuda_allocator"] = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "scope": "training loop including warmup and evaluation; PyTorch allocator only, not whole-device memory",
            }
        if measure_neighbor_padding:
            summary["neighbor_padding"] = measured_padding.summary()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": max_steps,
            },
            output_dir / "checkpoint.pt",
        )
        summary["completed"] = True
        emit(summary)
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
    parser.add_argument(
        "--measure-input",
        action="store_true",
        help="record loader wait, logical payload bytes and CUDA stream timings (adds instrumentation overhead)",
    )
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
        measure_input=args.measure_input,
    )


if __name__ == "__main__":
    main()
