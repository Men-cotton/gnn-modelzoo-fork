"""Complete local input-factor ablations with paired weights and RNG streams.

All hardware execution is a bounded local GPU mechanism probe. The original
four seed42 conditions remain immutable; the manifest links to those records.
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
from benchmark_scripts.research.logs import run_hpcasia_sampling_fix as sampling

ROOT = reference.ROOT
POLICIES = {
    f"o{o}n{n}b{b}": dict(shuffle=bool(o), neighbor="fresh_uniform" if n else "fixed_cyclic",
                           balanced=bool(b))
    for o in (0, 1) for n in (0, 1) for b in (0, 1)
}
POLICIES["uniform_frozen"] = dict(shuffle=False, neighbor="frozen_uniform", balanced=False)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def check_fixed_provider(dataset, provider, baseline_rows):
    """Check every original batch against recorded native-sampler tree hashes."""
    checks = []
    for index in range(len(dataset)):
        tree = provider.tree(index)
        nodes, masks, neighbors = tree
        if baseline_rows is not None:
            row = baseline_rows[index]
            assert reference.arrays_sha256([nodes[0][masks[0]]]) == row["target_sha256"]
            assert reference.arrays_sha256([*nodes[1:], *masks[1:]]) == row["neighbor_sha256"]
        # Native-code check of a bounded root prefix, independent of old hashes.
        if index in (0, len(dataset) - 1):
            width = min(8, int(masks[0].sum()))
            expected = dataset._sample_layers(nodes[0][:width], masks[0][:width])
            for group_index, (actual_group, expected_group) in enumerate(zip(tree, expected)):
                for actual, wanted in zip(actual_group, expected_group):
                    actual = actual.reshape(dataset.batch_size, -1)[:width].reshape(wanted.shape)
                    np.testing.assert_array_equal(actual, wanted)
        checks.append(dict(batch_index=index, recorded_hash_equal=baseline_rows is not None))
    return checks


def run_arm(args, dataset, provider, config, condition, anchors, old_summary, old_rows, first_epoch=None):
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
    if old_summary is not None:
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
    recorded = []
    policy_factors = POLICIES[condition]
    with (output / "steps.jsonl").open("x") as stream, \
            (output / "epoch_orders.jsonl").open("x") as orders, \
            (output / "sampling_audit.jsonl").open("x") as audit:
        for step in range(1, args.max_steps + 1):
            epoch, index = divmod(step - 1, len(dataset))
            if index == 0:
                dataset._ordered_targets = (
                    np.random.default_rng(args.seed + 10000 + epoch).permutation(provider.original_order)
                    if epoch and policy_factors["shuffle"] else provider.original_order.copy())
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
            if old_rows is not None:
                previous = old_rows[step - 1]
                assert rng_hash == previous["torch_rng_sha256"], f"Torch RNG differs at step {step}"
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
            if first_epoch is not None and step <= len(dataset):
                for key in ("loss", "valid_targets", "target_sha256", "neighbor_sha256",
                            "torch_rng_sha256", "optimizer_updated", "loss_scale_before"):
                    assert record[key] == first_epoch[step - 1][key], (condition, step, key)
            recorded.append(record)
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
    return result, recorded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelzoo-root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dataset", choices=["arxiv", "products"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--conditions", nargs="+", choices=list(POLICIES))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sampling-baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.modelzoo_root = args.modelzoo_root.resolve()
    args.data_dir = (args.data_dir or args.modelzoo_root /
                    "src/cerebras/modelzoo/models/gnn/data/datasets").resolve()
    args.baseline = args.baseline.resolve()
    args.max_steps = args.max_steps or (500 if args.dataset == "arxiv" else 1000)
    args.graph_loader, args.precision = "bounded", "fp16"
    args.output = args.output.resolve()
    args.conditions = args.conditions or (
        ["o0n0b1", "o1n0b1", "o0n1b0", "o0n1b1", "uniform_frozen"]
        if args.seed == 42 else ["o0n0b0", "o0n0b1"])
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("Conditions must be distinct")
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
    sampling.np, sampling.torch = np, torch
    from benchmark_scripts.research.logs.local_resampled_sampler import ResampledTreeProvider
    from benchmark_scripts.research.logs.local_factorial_sampler import FixedCyclicTreeProvider, FrozenUniformTreeProvider
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cg = next(x[3:] for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
    cgroot = Path("/sys/fs/cgroup") / cg.lstrip("/")
    limits = {name: (cgroot / name).read_text().strip() for name in ("memory.max", "memory.swap.max")}
    if limits["memory.max"] == "max" or int(limits["memory.max"]) > 10 * 1024**3:
        raise RuntimeError("A cgroup MemoryMax of at most 10 GiB is required")
    if limits["memory.swap.max"] != "0":
        raise RuntimeError("A cgroup MemorySwapMax of zero is required")
    args.output.mkdir(parents=True, exist_ok=True)
    old = json.loads((args.baseline / "manifest.json").read_text())
    assert old["status"] == "complete"
    manifest = dict(status="running", scope="local GPU factorial input-mechanism probe",
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        baseline=str(args.baseline), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), memory_limits=limits, runs=[], cells={},
        condition_policy={name: POLICIES[name] for name in args.conditions},
        loss_definition="instantaneous pre-update supervised-target mean at every step",
        local_source_sha256={str(Path(__file__).with_name(name)): reference.file_sha256(
            Path(__file__).with_name(name)) for name in
            (Path(__file__).name, "local_factorial_sampler.py", "local_resampled_sampler.py",
             "local_ogb_graph.py", "reproduce_hpcasia_tail_loss.py", "run_hpcasia_sampling_fix.py")})
    try:
        config, config_path = reference.configure(args)
        expected_config = json.loads((args.baseline / "resolved_config.json").read_text())
        expected_config["trainer"]["init"]["seed"] = args.seed
        expected_config["trainer"]["fit"]["train_dataloader"]["sampler_seed"] = args.seed
        assert config == expected_config
        for key, actual in (("torch_version", torch.__version__), ("cuda_version", torch.version.cuda),
                            ("gpu", torch.cuda.get_device_name())):
            assert actual == old[key], key
        reference.write_json(args.output / "resolved_config.json", config)
        manifest["reference_config"] = dict(path=str(config_path), sha256=reference.file_sha256(config_path))
        assert manifest["reference_config"] == old["reference_config"]
        for key in ("source_sha256", "dataset_sha256"):
            manifest[key] = {path: reference.file_sha256(path) for path in old[key]}
            assert manifest[key] == old[key], key
        print(json.dumps(dict(event="graph_loading", dataset=args.dataset, seed=args.seed)), flush=True)
        dataset, metadata = reference.make_dataset(args, config, torch.float16)
        manifest["graph_loader"] = metadata
        count, batches = len(dataset._ordered_targets), len(dataset)
        manifest.update(train_targets=count, batches_per_epoch=batches)
        size, extra = divmod(count, batches)
        balanced_sizes = [size + (index < extra) for index in range(batches)]
        classes = dict(fixed_cyclic=FixedCyclicTreeProvider, fresh_uniform=ResampledTreeProvider,
                       frozen_uniform=FrozenUniformTreeProvider)
        providers = {name: classes[POLICIES[name]["neighbor"]](dataset, seed=args.seed,
                     batch_sizes=balanced_sizes if POLICIES[name]["balanced"] else None)
                     for name in args.conditions}
        manifest["batch_sizes"] = {name: obj.target_counts for name, obj in providers.items()}
        old_rows = read_rows(args.baseline / "fixed/steps.jsonl") if args.seed == 42 else None
        old_summary = next(run for run in old["runs"] if run["condition"] == "fixed") if args.seed == 42 else None
        assert args.max_steps == old["arguments"]["max_steps"]
        manifest["fixed_provider_checks"] = check_fixed_provider(dataset,
            FixedCyclicTreeProvider(dataset, seed=args.seed), old_rows)
        starts, ids = dataset._forward_index.row_offsets, dataset._ordered_targets
        anchors = ids[(starts[ids + 1] - starts[ids]) >= 30][:8]
        assert len(anchors) == 8
        manifest["sampling_audit_nodes"] = anchors.tolist()
        manifest["sampling_audit_degrees"] = (starts[anchors + 1] - starts[anchors]).tolist()
        manifest["cuda_feature_store"] = sampling.cuda_features(dataset, providers, args.seed)
        if args.seed == 42:
            previous = args.sampling_baseline or (args.output / "no-sampling-baseline")
            manifest["cells"].update({
                "000": dict(directory=str(args.baseline), condition="fixed"),
                "100": dict(directory=str(args.baseline), condition="reshuffle"),
                "110": dict(directory=str(previous), condition="resampled"),
                "111": dict(directory=str(previous), condition="balanced"),
            })
        for name in args.conditions:
            code = name[1] + name[3] + name[5] if name != "uniform_frozen" else name
            assert code not in manifest["cells"], "Existing completed cells must remain unchanged"
            manifest["cells"][code] = dict(directory=str(args.output), condition=name)
        records = {}
        reference.write_json(args.output / "manifest.json", manifest)
        for name, provider in providers.items():
            expected_first = records.get("o0n1b0") if name == "uniform_frozen" else None
            result, rows = run_arm(args, dataset, provider, config, name, anchors,
                                   old_summary, old_rows, expected_first)
            manifest["runs"].append(result)
            records[name] = rows
            if old_summary is None:
                old_summary, old_rows = result, rows
            reference.write_json(args.output / "manifest.json", manifest)
        if "uniform_frozen" in records and "o0n1b0" in records:
            manifest["uniform_refresh_pair"] = dict(first_epoch_equal=True, steps_checked=batches)
        manifest["paired_checks"] = dict(initial_model_equal=True, initial_rng_equal=True,
            all_step_torch_rng_equal=True, optimizer_policy_equal=True)
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
