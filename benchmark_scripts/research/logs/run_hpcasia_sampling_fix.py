"""Local GPU probe of refreshed GraphSAGE sampling and balanced target batches.

Reuses the earlier probe's model, loss, configuration and native CUDA policy.
This is a local input provider, not a deployed CS-3 DataLoader fix.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from benchmark_scripts.research.logs import reproduce_hpcasia_tail_loss as reference

ROOT = reference.ROOT


def cuda_features(dataset, providers, seed):
    original = dataset._ordered_targets.copy()
    cpu = dataset.features
    gpu = cpu.cuda()
    checks = []
    try:
        for name, provider in providers.items():
            for epoch in (0, 1):
                dataset._ordered_targets = (original.copy() if epoch == 0 else
                    np.random.default_rng(seed + 10000 + epoch).permutation(original))
                provider.set_epoch(epoch)
                for index in (0, len(dataset) - 1):
                    tree = provider.tree(index)
                    dataset.features = cpu
                    expected = provider.payload(index, tree)
                    dataset.features = gpu
                    actual = provider.payload(index, tree)
                    reference.assert_payload_equal(actual, expected)
                    del actual, expected, tree
                    checks.append(dict(condition=name, epoch=epoch,
                                       batch_index=index, equal=True))
    finally:
        dataset._ordered_targets = original
        dataset.features = gpu
    torch.cuda.empty_cache()
    return dict(bytes=gpu.numel() * gpu.element_size(), checks=checks)


def run_arm(args, dataset, provider, config, condition, anchors, old_summary, old_rows):
    from cerebras.modelzoo.models.gnn.data_processing.batches import GraphSAGEBatch
    from cerebras.modelzoo.models.gnn.model import GNNModel
    from cerebras.modelzoo.models.gnn.gpu_policy import (
        adamw_kwargs, adamw_param_groups, grad_scaler_kwargs,
        optimizer_policy, OptimizerStepCounter,
    )
    output = args.output / condition
    output.mkdir()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset._ordered_targets = provider.original_order.copy()
    init = config["trainer"]["init"]
    model = GNNModel(copy.deepcopy(init["model"])).cuda().train()
    initial_hash = reference.state_sha256(model.state_dict())
    initial_rng = reference.rng_sha256()
    assert initial_hash == old_summary["initial_model_sha256"]
    assert initial_rng == old_summary["initial_rng_sha256"]
    optimizer = torch.optim.AdamW(adamw_param_groups(model), **adamw_kwargs(init))
    scaler_options = grad_scaler_kwargs(init, torch.float16)
    scaler = torch.amp.GradScaler("cuda", **scaler_options)
    counter = OptimizerStepCounter(optimizer)
    policy = optimizer_policy(model, optimizer, scaler_options)
    old_policy = json.loads((args.baseline / "reshuffle/optimizer_policy.json").read_text())
    assert json.loads(json.dumps(policy)) == old_policy, "Optimizer policy differs from the completed baseline"
    reference.write_json(output / "optimizer_policy.json", policy)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with (output / "steps.jsonl").open("x") as stream, \
            (output / "epoch_orders.jsonl").open("x") as orders, \
            (output / "sampling_audit.jsonl").open("x") as audit:
        for step in range(1, args.max_steps + 1):
            epoch, index = divmod(step - 1, len(dataset))
            if index == 0:
                dataset._ordered_targets = (
                    np.random.default_rng(args.seed + 10000 + epoch).permutation(provider.original_order)
                    if epoch else provider.original_order.copy())
                provider.set_epoch(epoch)
                epoch_hash = reference.arrays_sha256([dataset._ordered_targets])
                orders.write(json.dumps(dict(epoch=epoch, targets=dataset._ordered_targets.tolist(),
                                             sha256=epoch_hash)) + "\n")
                orders.flush()
            tree = provider.tree(index)
            nodes, masks, neighbors = tree
            targets = nodes[0][masks[0]]
            target_hash = reference.arrays_sha256([targets])
            neighbor_hash = reference.arrays_sha256([*nodes[1:], *masks[1:]])
            for row in np.flatnonzero(masks[0] & np.isin(nodes[0], anchors)):
                selected = nodes[1].reshape(dataset.batch_size, -1)[row][neighbors[0][row]]
                audit.write(json.dumps(dict(epoch=epoch, step=step, node=int(nodes[0][row]),
                    neighbors=selected.tolist(),
                    subset_sha256=reference.arrays_sha256([np.sort(selected)]))) + "\n")
            audit.flush()
            payload = provider.payload(index, tree)
            valid = int(payload["target_mask"].sum())
            supervised = int((payload["target_mask"] & (payload["labels"] != -100)).sum())
            assert valid == supervised == provider.target_counts[index]
            batch = GraphSAGEBatch.from_payload(payload).to(torch.device("cuda"))
            del payload
            optimizer.zero_grad(set_to_none=True)
            before = counter.steps
            scale_before = scaler.get_scale()
            rng_hash = reference.rng_sha256()
            previous = old_rows[step - 1]
            assert rng_hash == previous["torch_rng_sha256"], f"Torch RNG differs at step {step}"
            assert epoch_hash == previous["epoch_order_sha256"]
            if condition == "resampled":
                assert target_hash == previous["target_sha256"]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(batch)
            scalar_loss = float(loss.detach())
            if not np.isfinite(scalar_loss):
                raise FloatingPointError(f"Nonfinite loss at {condition} step {step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            record = dict(condition=condition, step=step, epoch=epoch, batch_index=index,
                loss=scalar_loss, valid_targets=valid, supervised_targets=supervised,
                tail=index == len(dataset) - 1, epoch_end=index == len(dataset) - 1,
                target_sha256=target_hash, neighbor_sha256=neighbor_hash,
                epoch_order_sha256=epoch_hash, torch_rng_sha256=rng_hash,
                optimizer_updates=counter.steps, optimizer_updated=counter.steps > before,
                loss_scale_before=scale_before, loss_scale_after=scaler.get_scale(),
                elapsed_seconds=time.perf_counter() - start)
            if record["tail"]:
                record["target_ids"] = targets.tolist()
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if step % 10 == 0 or record["tail"] or step == 1:
                print(json.dumps({key: record[key] for key in
                    ("condition", "step", "loss", "valid_targets", "epoch_end",
                     "optimizer_updated", "elapsed_seconds")}), flush=True)
            del batch, loss, tree, nodes, masks, neighbors
    result = dict(status="complete", condition=condition, steps=args.max_steps,
        initial_model_sha256=initial_hash, initial_rng_sha256=initial_rng,
        final_model_sha256=reference.state_sha256(model.state_dict()),
        optimizer_updates=counter.steps, skipped_updates=args.max_steps - counter.steps,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        seconds=time.perf_counter() - start)
    reference.write_json(output / "summary.json", result)
    del model, optimizer, scaler, counter
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelzoo-root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dataset", choices=["arxiv", "products"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.modelzoo_root = args.modelzoo_root.resolve()
    args.data_dir = (args.data_dir or args.modelzoo_root /
                    "src/cerebras/modelzoo/models/gnn/data/datasets").resolve()
    args.baseline = args.baseline.resolve()
    args.max_steps = args.max_steps or (500 if args.dataset == "arxiv" else 1000)
    args.graph_loader = "bounded"
    args.precision = "fp16"
    if min(args.max_steps, args.hidden, args.threads) <= 0:
        parser.error("steps, hidden width, and thread count must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output}")
    sys.path.insert(0, str(args.modelzoo_root / "src"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    global np, torch
    import numpy as np
    import torch
    reference.np, reference.torch = np, torch
    from benchmark_scripts.research.logs.local_resampled_sampler import ResampledTreeProvider
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cgroup_line = next(x for x in Path("/proc/self/cgroup").read_text().splitlines()
                       if x.startswith("0::"))
    cgroup_root = Path("/sys/fs/cgroup") / cgroup_line[3:].lstrip("/")
    limits = {name: (cgroup_root / name).read_text().strip()
              for name in ("memory.max", "memory.swap.max")}
    if limits["memory.max"] == "max" or int(limits["memory.max"]) > 10 * 1024**3:
        raise RuntimeError("A cgroup MemoryMax of at most 10 GiB is required")
    if limits["memory.swap.max"] != "0":
        raise RuntimeError("A cgroup MemorySwapMax of zero is required")
    args.output.mkdir(parents=True, exist_ok=True)
    old = json.loads((args.baseline / "manifest.json").read_text())
    assert old["status"] == "complete"
    manifest = dict(status="running", scope="local GPU sampling and batch-partition probe",
        arguments={key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()}, baseline=str(args.baseline),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), memory_limits=limits,
        loss_definition="instantaneous pre-update supervised-target mean at every step",
        input_policy={"resampled": "epoch target reshuffle, fresh uniform neighbors, original batch sizes",
                      "balanced": "epoch target reshuffle, fresh uniform neighbors, balanced batch sizes"},
        differences_from_cs3=["native CUDA and torch.optim.AdamW/GradScaler",
            "hidden width 128 instead of 1024", "no validation or CS-3 execution segments",
            "GPU feature store with original feature gathering", "modified local input provider"],
        local_source_sha256={str(Path(__file__).with_name(name)): reference.file_sha256(
            Path(__file__).with_name(name)) for name in
            (Path(__file__).name, "local_resampled_sampler.py", "local_ogb_graph.py",
             "reproduce_hpcasia_tail_loss.py")}, runs=[])
    try:
        config, config_path = reference.configure(args)
        assert config == json.loads((args.baseline / "resolved_config.json").read_text())
        reference.write_json(args.output / "resolved_config.json", config)
        manifest["reference_config"] = dict(path=str(config_path), sha256=reference.file_sha256(config_path))
        assert manifest["reference_config"] == old["reference_config"]
        for key in ("source_sha256", "dataset_sha256"):
            manifest[key] = {path: reference.file_sha256(path) for path in old[key]}
            assert manifest[key] == old[key], f"Baseline source/data changed: {key}"
        print(json.dumps(dict(event="graph_loading", dataset=args.dataset)), flush=True)
        dataset, metadata = reference.make_dataset(args, config, torch.float16)
        manifest["graph_loader"] = metadata
        count, batches = len(dataset._ordered_targets), len(dataset)
        manifest.update(train_targets=count, batches_per_epoch=batches)
        size, extra = divmod(count, batches)
        providers = {
            "resampled": ResampledTreeProvider(dataset, seed=args.seed),
            "balanced": ResampledTreeProvider(dataset, seed=args.seed,
                batch_sizes=[size + (index < extra) for index in range(batches)]),
        }
        manifest["batch_sizes"] = {name: obj.target_counts for name, obj in providers.items()}
        starts = dataset._forward_index.row_offsets
        ids = dataset._ordered_targets
        anchors = ids[(starts[ids + 1] - starts[ids]) >= 30][:8]
        if len(anchors) != 8:
            raise ValueError("Expected eight high-degree training targets for the sampling audit")
        manifest["sampling_audit_nodes"] = anchors.tolist()
        manifest["sampling_audit_degrees"] = (starts[anchors + 1] - starts[anchors]).tolist()
        manifest["cuda_feature_store"] = cuda_features(dataset, providers, args.seed)
        old_rows = [json.loads(line) for line in (args.baseline / "reshuffle/steps.jsonl").read_text().splitlines()]
        old_summary = next(run for run in old["runs"] if run["condition"] == "reshuffle")
        assert len(old_rows) == args.max_steps
        reference.write_json(args.output / "manifest.json", manifest)
        for name, provider in providers.items():
            manifest["runs"].append(run_arm(args, dataset, provider, config, name,
                                           anchors, old_summary, old_rows))
            reference.write_json(args.output / "manifest.json", manifest)
        manifest["baseline_checks"] = dict(config_equal=True, source_equal=True, dataset_equal=True,
            initial_model_equal=True, initial_rng_equal=True, optimizer_policy_equal=True,
            all_step_torch_rng_equal=True, all_epoch_orders_equal=True,
            resampled_all_step_targets_equal=True)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
        raise
    finally:
        reference.write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
