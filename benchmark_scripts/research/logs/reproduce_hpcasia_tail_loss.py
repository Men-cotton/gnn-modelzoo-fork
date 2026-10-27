"""Run a paired, local GPU test of repeated GraphSAGE tail batches.

This is an input-order mechanism probe, not a CS-3 performance measurement.
It imports the original Model Zoo sampler, model, masked loss and native GPU
optimizer policy. Only the target permutation changes between the two arms.
Sampled IDs/masks may be cached; feature batches are always gathered afresh.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[3]


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def arrays_sha256(arrays):
    result = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        result.update(str((value.shape, value.dtype.str)).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def state_sha256(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def rng_sha256():
    return arrays_sha256([torch.get_rng_state().numpy(),
                          torch.cuda.get_rng_state().cpu().numpy()])


def target_batch(dataset, index):
    ids = np.full(dataset.batch_size, dataset.pad_id, dtype=np.int64)
    mask = np.zeros(dataset.batch_size, dtype=bool)
    targets = dataset._ordered_targets[index * dataset.batch_size:
                                       (index + 1) * dataset.batch_size]
    ids[:len(targets)] = targets
    mask[:len(targets)] = True
    return ids, mask


class TreeCache:
    """Cache original sampler outputs per real root; never cache features."""

    def __init__(self, dataset, *, progress=True):
        self.dataset = dataset
        self.original_order = dataset._ordered_targets.copy()
        widths = [1, *np.cumprod(dataset.fanouts).tolist()]
        self.nodes = [np.empty((len(self.original_order), width), dtype=np.int32)
                      for width in widths]
        self.masks = [np.empty_like(value, dtype=bool) for value in self.nodes]
        self.lookup = np.full(dataset.num_nodes, -1, dtype=np.int32)
        self.lookup[self.original_order] = np.arange(len(self.original_order))
        if dataset.num_nodes > np.iinfo(np.int32).max:
            raise ValueError("The ID cache requires 32-bit node IDs")
        for index in range(len(dataset)):
            ids, valid = target_batch(dataset, index)
            nodes, masks, neighbors = dataset._sample_layers(ids, valid)
            count = int(valid.sum())
            start = index * dataset.batch_size
            for depth, (node_ids, node_mask) in enumerate(zip(nodes, masks)):
                self.nodes[depth][start:start + count] = node_ids.reshape(
                    dataset.batch_size, -1)[:count]
                self.masks[depth][start:start + count] = node_mask.reshape(
                    dataset.batch_size, -1)[:count]
                if depth:
                    np.testing.assert_array_equal(node_mask.reshape(-1),
                                                  neighbors[depth - 1].reshape(-1))
            if progress and (index % 5 == 0 or index + 1 == len(dataset)):
                print(json.dumps({"event": "cache", "batch": index + 1,
                                  "batches": len(dataset)}), flush=True)
        self.bytes = sum(x.nbytes for x in self.nodes + self.masks) + self.lookup.nbytes

    def tree(self, index):
        ids, valid = target_batch(self.dataset, index)
        rows = self.lookup[ids[valid]]
        if (rows < 0).any():
            raise ValueError("Target not present in the original training split")
        nodes, masks, neighbors = [], [], []
        for depth, (cached_ids, cached_mask) in enumerate(zip(self.nodes, self.masks)):
            values = np.full((len(ids), cached_ids.shape[1]), self.dataset.pad_id,
                             dtype=np.int64)
            flags = np.zeros_like(values, dtype=bool)
            values[valid] = cached_ids[rows]
            flags[valid] = cached_mask[rows]
            nodes.append(values.reshape(-1))
            masks.append(flags.reshape(-1))
            if depth:
                neighbors.append(flags.reshape(-1, self.dataset.fanouts[depth - 1]))
        return nodes, masks, neighbors

    def payload(self, index, tree=None):
        dataset = self.dataset
        nodes, masks, neighbors = self.tree(index) if tree is None else tree
        features = dataset._gather_features(nodes, masks)
        labels = dataset.labels[torch.from_numpy(nodes[0])].clone().reshape(-1)
        target_mask = torch.from_numpy(masks[0]).bool()
        labels[~target_mask] = 0
        return {
            "node_features": [x.view(dataset.batch_size, -1, x.shape[-1])
                              for x in features],
            "node_masks": [torch.from_numpy(x).view(dataset.batch_size, -1)
                           for x in masks],
            "neighbor_masks": [torch.from_numpy(x).view(dataset.batch_size, -1,
                                                        x.shape[-1])
                               for x in neighbors],
            "labels": labels,
            "target_mask": target_mask,
        }


def assert_payload_equal(actual, expected):
    for key in actual:
        values_a, values_b = actual[key], expected[key]
        if torch.is_tensor(values_a):
            values_a, values_b = [values_a], [values_b]
        if len(values_a) != len(values_b):
            raise AssertionError(f"Payload length mismatch: {key}")
        for left, right in zip(values_a, values_b):
            torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)


def verify_cache(dataset, cache, seed):
    checks = []
    original = dataset._ordered_targets.copy()
    try:
        for order in ("original", "reshuffled"):
            dataset._ordered_targets = (original.copy() if order == "original" else
                                        np.random.default_rng(seed + 10001).permutation(original))
            for index in sorted({0, len(dataset) - 1}):
                ids, valid = target_batch(dataset, index)
                actual = cache.tree(index)
                expected = dataset._sample_layers(ids, valid)
                for a, b in zip(actual, expected):
                    for left, right in zip(a, b):
                        np.testing.assert_array_equal(left, right)
                assert_payload_equal(cache.payload(index, actual), dataset[index])
                checks.append({"order": order, "batch_index": index,
                               "valid_targets": int(valid.sum()), "equal": True})
    finally:
        dataset._ordered_targets = original
    return checks


def use_cuda_feature_store(dataset, cache, seed):
    """Run the original gather on CUDA and check against the CPU payload."""
    original = dataset._ordered_targets.copy()
    cpu_features = dataset.features
    gpu_features = cpu_features.cuda()
    checks = []
    try:
        for order in ("original", "reshuffled"):
            dataset._ordered_targets = (original.copy() if order == "original" else
                                        np.random.default_rng(seed + 10001).permutation(original))
            for index in sorted({0, len(dataset) - 1}):
                dataset.features = cpu_features
                expected = cache.payload(index)
                dataset.features = gpu_features
                actual = cache.payload(index)
                assert_payload_equal(actual, expected)
                del actual, expected
                checks.append({"order": order, "batch_index": index, "equal": True})
    finally:
        dataset._ordered_targets = original
        dataset.features = gpu_features
    torch.cuda.empty_cache()
    return {"bytes": gpu_features.numel() * gpu_features.element_size(), "checks": checks}


def configure(args):
    from cerebras.modelzoo.common.utils.run.config_loader import load_params_file
    config_path = (args.modelzoo_root / "src/cerebras/modelzoo/models/gnn/configs" /
                   f"params_graphsage_ogbn_{args.dataset}_accuracy_nocache.yaml")
    config = load_params_file(config_path)
    init = config["trainer"]["init"]
    init["seed"] = args.seed
    init["model"]["architecture"]["hidden_dim"] = args.hidden
    init["model"].setdefault("task", {})["compute_eval_metrics"] = False
    train = config["trainer"]["fit"]["train_dataloader"]
    train.update(data_dir=str(args.data_dir), dataset_name=f"ogbn-{args.dataset}",
                 sampler_seed=args.seed, num_workers=0, persistent_workers=False,
                 pin_memory=False, shuffle=True, cache_fraction=None,
                 static_batch_cache_size=0, drop_last_batch=False)
    if train["batch_size"] != 4096 or train["fanouts"] != [15, 10, 5]:
        raise ValueError("Reference batch/fanout configuration has changed")
    init["loop"]["max_steps"] = args.max_steps
    init["loop"]["eval_frequency"] = None
    init["logging"]["log_steps"] = 1
    return config, config_path


def make_dataset(args, config, dtype):
    if args.graph_loader == "bounded":
        from cerebras.modelzoo.models.gnn.research.ogb import load_local_ogb_graph
        from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
            GraphSAGENeighborSamplerDataset,
        )
        components = load_local_ogb_graph(f"ogbn-{args.dataset}", args.data_dir)
        metadata = components.pop("metadata")
        components["features"] = components["features"].to(dtype=dtype)
        dataset = GraphSAGENeighborSamplerDataset(
            **components, fanouts=[15, 10, 5], batch_size=4096, shuffle=True,
            pad_id=0, seed=args.seed, drop_last=False)
        return dataset, metadata
    if args.dataset == "products":
        raise ValueError("products requires --graph-loader bounded; native preprocessing exceeds local RAM")
    from cerebras.modelzoo.models.gnn.fixed_shape_gpu import make_loader
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    # OGB's local, trusted preprocessed files contain these PyG containers.
    with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
        loader = make_loader(config["trainer"]["fit"]["train_dataloader"],
                             num_layers=3, float_dtype=dtype)
    return loader.dataset, {"loader": "original Model Zoo make_loader"}


def run_arm(args, dataset, cache, config, condition, dtype):
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
    dataset._ordered_targets = cache.original_order.copy()
    init = config["trainer"]["init"]
    model = GNNModel(copy.deepcopy(init["model"])).cuda().train()
    initial_hash = state_sha256(model.state_dict())
    initial_rng_hash = rng_sha256()
    torch.save(model.state_dict(), output / "initial_model.pt")
    optimizer = torch.optim.AdamW(adamw_param_groups(model), **adamw_kwargs(init))
    scaler_options = grad_scaler_kwargs(init, dtype)
    scaler = torch.amp.GradScaler("cuda", **scaler_options)
    counter = OptimizerStepCounter(optimizer)
    policy = optimizer_policy(model, optimizer, scaler_options)
    write_json(output / "optimizer_policy.json", policy)
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    epoch_order = None
    recorded = []
    with (output / "steps.jsonl").open("x") as stream, \
            (output / "epoch_orders.jsonl").open("x") as orders:
        for step in range(1, args.max_steps + 1):
            epoch, batch_index = divmod(step - 1, len(dataset))
            if batch_index == 0:
                dataset._ordered_targets = (
                    np.random.default_rng(args.seed + 10000 + epoch).permutation(cache.original_order)
                    if condition == "reshuffle" and epoch else cache.original_order.copy())
                epoch_order = arrays_sha256([dataset._ordered_targets])
                orders.write(json.dumps({"epoch": epoch, "targets": dataset._ordered_targets.tolist(),
                                         "sha256": epoch_order}) + "\n")
                orders.flush()
            tree = cache.tree(batch_index)
            nodes, masks, _ = tree
            target_ids = nodes[0][masks[0]]
            target_hash = arrays_sha256([target_ids])
            neighbor_hash = arrays_sha256([*nodes[1:], *masks[1:]])
            payload = cache.payload(batch_index, tree)
            valid_count = int(payload["target_mask"].sum())
            supervised = int((payload["target_mask"] & (payload["labels"] != -100)).sum())
            batch = GraphSAGEBatch.from_payload(payload).to(torch.device("cuda"))
            del payload
            optimizer.zero_grad(set_to_none=True)
            before = counter.steps
            scale_before = scaler.get_scale()
            torch_rng = rng_sha256()
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                loss = model(batch)
            scalar_loss = float(loss.detach())
            if not np.isfinite(scalar_loss):
                raise FloatingPointError(f"nonfinite loss at {condition} step {step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            after = counter.steps
            record = {
                "condition": condition, "step": step, "epoch": epoch,
                "batch_index": batch_index, "loss": scalar_loss,
                "valid_targets": valid_count, "supervised_targets": supervised,
                "tail": batch_index == len(dataset) - 1,
                "target_sha256": target_hash, "neighbor_sha256": neighbor_hash,
                "epoch_order_sha256": epoch_order, "torch_rng_sha256": torch_rng,
                "optimizer_updates": after, "optimizer_updated": after > before,
                "loss_scale_before": scale_before, "loss_scale_after": scaler.get_scale(),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            if record["tail"]:
                record["target_ids"] = target_ids.tolist()
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            recorded.append(record)
            if step % 10 == 0 or record["tail"] or step == 1:
                print(json.dumps({key: record[key] for key in
                                  ("condition", "step", "loss", "valid_targets", "tail",
                                   "optimizer_updated", "elapsed_seconds")}), flush=True)
            del batch, loss, tree
    result = {
        "status": "complete", "condition": condition, "steps": len(recorded),
        "initial_model_sha256": initial_hash, "initial_rng_sha256": initial_rng_hash,
        "final_model_sha256": state_sha256(model.state_dict()),
        "optimizer_updates": counter.steps,
        "skipped_updates": len(recorded) - counter.steps,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "seconds": time.perf_counter() - start_time,
    }
    write_json(output / "summary.json", result)
    del model, optimizer, scaler, counter
    torch.cuda.empty_cache()
    return result, recorded


def self_test():
    from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
        GraphSAGENeighborSamplerDataset,
    )
    dataset = GraphSAGENeighborSamplerDataset(
        features=torch.arange(24, dtype=torch.float32).reshape(8, 3),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                                 [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]]),
        labels=torch.arange(8) % 2, mask=torch.arange(8) < 7, fanouts=[2, 2],
        batch_size=3, shuffle=True, pad_id=0, seed=42,
        edge_index_is_undirected=True,
    )
    cache = TreeCache(dataset, progress=False)
    checks = verify_cache(dataset, cache, 42)
    assert [int(cache.payload(i)["target_mask"].sum()) for i in range(len(dataset))] == [3, 3, 1]
    print(json.dumps({"self_test": "passed", "cache_checks": checks}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelzoo-root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dataset", choices=["arxiv", "products"], default="arxiv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--condition", choices=["fixed", "reshuffle", "both"], default="both")
    parser.add_argument("--graph-loader", choices=["native", "bounded"], default="bounded")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    args.modelzoo_root = args.modelzoo_root.resolve()
    if args.data_dir is None:
        args.data_dir = args.modelzoo_root / "src/cerebras/modelzoo/models/gnn/data/datasets"
    args.data_dir = args.data_dir.resolve()
    args.max_steps = args.max_steps or (500 if args.dataset == "arxiv" else 1000)
    if min(args.max_steps, args.hidden, args.threads) <= 0:
        parser.error("steps, hidden width, and thread count must be positive")
    if args.graph_loader == "bounded" and args.precision != "fp16":
        parser.error("the bounded loader currently supports only FP16 features")
    sys.path.insert(0, str(args.modelzoo_root / "src"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    global np, torch
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    if args.self_test:
        self_test()
        return
    if args.output is None:
        parser.error("--output is required for a GPU run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this runner does not silently fall back to CPU")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = {"status": "running", "scope": "local GPU input-order mechanism probe",
                "arguments": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "differences_from_cs3": ["native CUDA execution and torch.optim.AdamW/GradScaler",
                                         "no validation interruptions or CS-3 execution segments",
                                         "CPU ID/mask cache and GPU feature store; features gathered afresh",
                                         "deterministic CUDA algorithms; TF32 disabled",
                                         f"hidden width {args.hidden}; reference width 1024"],
                "loss_definition": "instantaneous pre-update supervised-target mean at every step",
                "control": "epoch target permutation only; sampler seed and neighbors remain fixed",
                "script_sha256": file_sha256(__file__),
                "bounded_loader_sha256": file_sha256(ROOT / "src/cerebras/modelzoo/models/gnn/research/ogb.py")}
    cgroup_line = next(line for line in Path("/proc/self/cgroup").read_text().splitlines()
                       if line.startswith("0::"))
    cgroup_root = Path("/sys/fs/cgroup") / cgroup_line[3:].lstrip("/")
    manifest["memory_limits"] = {name: (cgroup_root / name).read_text().strip()
                                 for name in ("memory.max", "memory.swap.max")}
    if manifest["memory_limits"]["memory.max"] == "max":
        raise RuntimeError("run inside a scope with a finite MemoryMax (recommended: 10G)")
    write_json(args.output / "manifest.json", manifest)
    try:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
        config, config_path = configure(args)
        write_json(args.output / "resolved_config.json", config)
        source_root = args.modelzoo_root / "src/cerebras/modelzoo/models/gnn"
        sources = ["architectures/graphsage.py", "task/wrapper.py", "task/adapters.py", "task/loss.py",
                   "data_processing/samplers/neighbor_tree.py", "data_processing/sources/base.py",
                   "data_processing/batches.py", "gpu_policy.py", "fixed_shape_gpu.py"]
        manifest["source_sha256"] = {str(source_root / path): file_sha256(source_root / path)
                                     for path in sources}
        manifest["reference_config"] = {"path": str(config_path), "sha256": file_sha256(config_path)}
        dataset_root = args.data_dir / f"ogbn-{args.dataset}" / f"ogbn_{args.dataset}"
        data_paths = [dataset_root / "processed/geometric_data_processed.pt",
                      dataset_root / "split" / ("time" if args.dataset == "arxiv" else "sales_ranking") / "train.csv.gz"]
        print(json.dumps({"event": "dataset_hashing", "files": [str(x) for x in data_paths]}), flush=True)
        manifest["dataset_sha256"] = {str(path): file_sha256(path) for path in data_paths}
        print(json.dumps({"event": "graph_loading", "loader": args.graph_loader}), flush=True)
        dataset, loader_metadata = make_dataset(args, config, dtype)
        manifest["graph_loader"] = loader_metadata
        manifest["batches_per_epoch"] = len(dataset)
        manifest["train_targets"] = len(dataset._ordered_targets)
        cache = TreeCache(dataset)
        manifest["cache_bytes"] = cache.bytes
        manifest["cache_verification"] = verify_cache(dataset, cache, args.seed)
        manifest["cuda_feature_store"] = use_cuda_feature_store(dataset, cache, args.seed)
        write_json(args.output / "manifest.json", manifest)
        results, records = [], []
        conditions = ["fixed", "reshuffle"] if args.condition == "both" else [args.condition]
        for condition in conditions:
            result, rows = run_arm(args, dataset, cache, config, condition, dtype)
            results.append(result)
            records.append(rows)
        manifest["runs"] = results
        if len(results) == 2:
            comparable = ["loss", "valid_targets", "target_sha256", "neighbor_sha256",
                          "torch_rng_sha256", "optimizer_updated", "loss_scale_before"]
            first_epoch_equal = all(all(a[key] == b[key] for key in comparable)
                                    for a, b in zip(records[0][:len(dataset)], records[1][:len(dataset)]))
            manifest["paired_check"] = {
                "initial_model_equal": results[0]["initial_model_sha256"] == results[1]["initial_model_sha256"],
                "initial_rng_equal": results[0]["initial_rng_sha256"] == results[1]["initial_rng_sha256"],
                "first_epoch_records_equal": first_epoch_equal,
                "first_epoch_steps_checked": min(args.max_steps, len(dataset)),
            }
            if not all(manifest["paired_check"][key] for key in
                       ("initial_model_equal", "initial_rng_equal", "first_epoch_records_equal")):
                raise AssertionError("Paired initial state / first-epoch equality failed")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
