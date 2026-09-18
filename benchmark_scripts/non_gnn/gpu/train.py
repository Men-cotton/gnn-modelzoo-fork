#!/usr/bin/env python3
"""Single-GPU BF16 training with native SDPA, AdamW, and explicit loop timing."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import yaml


def create_loader(config):
    # Reuse the exact input contract and preprocessing, with native DataLoader.
    if config["data_processor"] == "BertCSVDynamicMaskDataProcessor":
        from cerebras.modelzoo.data.nlp.bert.BertCSVDynamicMaskDataProcessor import (
            BertCSVDynamicMaskDataProcessor,
            BertCSVDynamicMaskDataProcessorConfig,
        )

        return BertCSVDynamicMaskDataProcessor(
            BertCSVDynamicMaskDataProcessorConfig(**config)
        ).create_dataloader()
    from cerebras.modelzoo.data.nlp.gpt.GptHDF5MapDataProcessor import (
        GptHDF5MapDataProcessor,
    )

    return GptHDF5MapDataProcessor(config).create_dataloader()


def batches_forever(loader):
    while True:
        found = False
        for batch in loader:
            found = True
            yield batch
        if not found:
            raise ValueError(
                "DataLoader produced no batches; check dataset size and workers"
            )


def token_counts(window, model_name):
    """Count host masks without changing either model's loss normalization."""
    valid = sum(int(batch["attention_mask"].sum()) for batch in window)
    # BERT attention masks count input positions; its loss mask selects only MLM
    # targets. NSP contributes a separate sample-level loss, not target tokens.
    targets = (
        sum(int(batch["masked_lm_mask"].sum()) for batch in window)
        if model_name == "bert"
        else valid
    )
    return valid, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    out = config_path.parent
    launch = json.loads((out / "launch.json").read_text())
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != launch["params_sha256"]:
        parser.error("params.yaml changed after preparation; prepare a new run")
    if launch["gpu_implementation"] == "modelzoo":
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "cerebras.modelzoo.cli.main",
                "fit",
                str(config_path),
                "--target_device",
                "GPU",
                "--model_dir",
                str(out),
            ],
        )

    import numpy as np
    import torch
    import transformers
    from models import make_model

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("Native GPU training requires a CUDA device with BF16 support")
    config = yaml.safe_load(config_path.read_text())["trainer"]
    init = config["init"]
    seed = init["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    model = make_model(init["model"]).to(device).train()
    if launch["gradient_checkpointing"]:
        model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 or "bias" in name else decay).append(param)
    opt = init["optimizer"]["AdamW"]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": opt["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=opt["lr"],
        betas=tuple(opt.get("betas", (0.9, 0.999))),
        eps=opt.get("eps", 1e-6),
        fused=True,
    )
    if launch["compile"]:
        model = torch.compile(model)
    loader = create_loader(config["fit"]["train_dataloader"])
    iterator = batches_forever(loader)
    accum = init["loop"]["grad_accum_steps"]
    effective = launch["effective_batch_size"]
    length = launch["sequence_length"]
    warmup = launch["warmup_steps"]
    max_steps = init["loop"]["max_steps"]
    is_bert = init["model"]["name"] == "bert"
    token_definitions = {
        "valid_tokens_definition": (
            "nonpadding_input_positions" if is_bert else "loss_target_positions"
        ),
        "loss_target_tokens_definition": (
            "masked_language_model_positions"
            if is_bert
            else "causal_language_model_positions"
        ),
    }
    if is_bert:
        token_definitions["nsp_examples_definition"] = (
            "next_sentence_prediction_examples"
        )
    device_properties = torch.cuda.get_device_properties(device)
    run_info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_total_memory_bytes": device_properties.total_memory,
        "device_compute_capability": [
            device_properties.major,
            device_properties.minor,
        ],
        "compute_dtype": "bfloat16",
        "compute_dtype_scope": "CUDA autocast; some operators use float32",
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "attention": "sdpa",
        "optimizer": "torch.optim.AdamW(fused=True)",
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        **launch,
        **token_definitions,
    }
    (out / "gpu_environment.json").write_text(json.dumps(run_info, indent=2) + "\n")
    metrics = out / "metrics.jsonl"
    # Refuse to mix metrics from two attempts in one run directory.
    with metrics.open("x") as log:

        def record(value):
            line = json.dumps(value)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        torch.cuda.synchronize()
        start = measured_start = time.perf_counter()
        valid_tokens = measured_valid_tokens = 0
        loss_target_tokens = measured_loss_target_tokens = 0
        loss_nonfinite_steps = measurement_loss_nonfinite_steps = 0
        for step in range(1, max_steps + 1):
            if step == warmup + 1:
                torch.cuda.synchronize()
                measured_start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
            # Fetch the whole update on the host to normalize Llama's loss by
            # all valid tokens, including when microbatches have unequal masks.
            window = [next(iterator) for _ in range(accum)]
            if sum(batch["input_ids"].shape[0] for batch in window) != effective:
                raise ValueError("Incomplete effective batch")
            if any(batch["input_ids"].shape[1] != length for batch in window):
                raise ValueError("DataLoader sequence length differs from the profile")
            count, target_count = token_counts(window, init["model"]["name"])
            denominator = effective if is_bert else count
            if denominator <= 0:
                raise ValueError("Effective batch has no valid loss tokens")
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device)
            for host_batch in window:
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in host_batch.items()
                }
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(batch) / denominator
                loss.backward()
                total_loss += loss.detach()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            torch.cuda.synchronize()
            now = time.perf_counter()
            loss_value = total_loss.item()
            loss_is_finite = math.isfinite(loss_value)
            valid_tokens += count
            loss_target_tokens += target_count
            loss_nonfinite_steps += not loss_is_finite
            if step > warmup:
                measured_valid_tokens += count
                measured_loss_target_tokens += target_count
                measurement_loss_nonfinite_steps += not loss_is_finite
            record(
                {
                    "event": "train",
                    "step": step,
                    "loss": loss_value,
                    "loss_is_finite": loss_is_finite,
                    "elapsed_seconds": now - start,
                    "samples": step * effective,
                    "nominal_tokens": step * effective * length,
                    "valid_tokens": valid_tokens,
                    "loss_target_tokens": loss_target_tokens,
                    "update_samples": effective,
                    "update_nominal_tokens": effective * length,
                    "update_valid_tokens": count,
                    "update_loss_target_tokens": target_count,
                    **(
                        {
                            "nsp_examples": step * effective,
                            "update_nsp_examples": effective,
                        }
                        if is_bert
                        else {}
                    ),
                    "warmup": step <= warmup,
                }
            )
        elapsed = now - measured_start
        samples = (max_steps - warmup) * effective
        record(
            {
                "event": "summary",
                "warmup_steps": warmup,
                "window_start_step": warmup,
                "window_end_step": max_steps,
                "measured_optimizer_steps": max_steps - warmup,
                "measurement_start_elapsed_seconds": measured_start - start,
                "measurement_end_elapsed_seconds": now - start,
                "measurement_window_seconds": elapsed,
                "elapsed_seconds": elapsed,
                "samples": samples,
                "nominal_tokens": samples * length,
                "measured_valid_tokens": measured_valid_tokens,
                "measured_loss_target_tokens": measured_loss_target_tokens,
                **({"measured_nsp_examples": samples} if is_bert else {}),
                "samples_per_second": samples / elapsed,
                "nominal_tokens_per_second": samples * length / elapsed,
                "valid_tokens_per_second": measured_valid_tokens / elapsed,
                "loss_target_tokens_per_second": measured_loss_target_tokens / elapsed,
                "loss_nonfinite_steps": loss_nonfinite_steps,
                "measurement_loss_nonfinite_steps": measurement_loss_nonfinite_steps,
                "loss_is_finite": loss_nonfinite_steps == 0,
                "measurement_loss_is_finite": measurement_loss_nonfinite_steps == 0,
                **token_definitions,
                "peak_memory_scope": "measurement_window",
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )


if __name__ == "__main__":
    main()
