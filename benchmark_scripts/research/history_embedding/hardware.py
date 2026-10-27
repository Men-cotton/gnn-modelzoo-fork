"""Cerebras SDK 2.10 entry point for immutable epoch histories.

`prepare` creates portable tensors. `train` executes one bounded job. `refresh`
loads its model checkpoint and creates the NEXT immutable snapshot on the host.
Workers never consume a client-mutated Python dictionary. Exported activations
are a transfer probe, not online worker feedback. A new process is used per job.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from benchmark_scripts.research.history_embedding.core import (
    HistorySAGE,
    HistoryStore,
    masked_loss,
    neighbor_mean,
    prepare,
    prepare_tree,
    prepare_tree_mean,
    refresh,
    tensor_bytes,
)


def pack_batch(batch):
    # The logical node batch is ONE SDK sample. Otherwise SDK microbatching
    # could split the roots while local_index still addresses the entire batch.
    return {
        k: tuple(x.unsqueeze(0) for x in v)
        if isinstance(v, (tuple, list))
        else v.unsqueeze(0)
        for k, v in batch.items()
    }


def unpack_batch(batch):
    return {
        k: tuple(x[0] for x in v) if isinstance(v, (tuple, list)) else v[0]
        for k, v in batch.items()
    }


class StaticModel(HistorySAGE):
    def forward(self, batch):
        batch = unpack_batch(batch)
        if "node_features" in batch:
            batch = dict(batch)
            batch["node_features"] = tuple(
                x.reshape(-1, x.shape[-1]) for x in batch["node_features"]
            )
            batch["neighbor_masks"] = tuple(
                x.reshape(-1, x.shape[-1]) for x in batch["neighbor_masks"]
            )
            if "node_means" in batch:
                batch["node_means"] = tuple(
                    x.reshape(-1, x.shape[-1]) for x in batch["node_means"]
                )
        return super().forward(batch)


class SnapshotDataset(torch.utils.data.Dataset):
    def __init__(self, bundle_path, snapshot_path, mode, batch_size, epoch, seed):
        self.bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
        self.mode, self.batch_size = mode, batch_size
        self.store = None
        if mode == "history":
            snap = torch.load(snapshot_path, map_location="cpu", weights_only=True)
            if snap["epoch"] != epoch:
                raise ValueError(
                    "Snapshot epoch does not match requested training epoch"
                )
            self.store = HistoryStore(snap["histories"], snap["weight_step"])
            self.store.means = snap["means"]
        ids = self.bundle["train_ids"]
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        self.order = ids[torch.randperm(len(ids), generator=generator)]

    def __len__(self):
        return math.ceil(len(self.order) / self.batch_size)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        b = self.bundle
        roots = self.order[index * self.batch_size : (index + 1) * self.batch_size]
        if self.mode.startswith("tree"):
            if self.mode == "tree_mean":
                batch = prepare_tree_mean(
                    b["features"],
                    b["x_mean"],
                    b["neighbors"],
                    b["mask"],
                    b["labels"],
                    roots,
                    b["model_config"]["num_layers"],
                    self.batch_size,
                )
                batch["node_means"] = tuple(
                    x.reshape(self.batch_size, -1, x.shape[-1])
                    for x in batch["node_means"]
                )
            else:
                batch = prepare_tree(
                    b["features"],
                    b["neighbors"],
                    b["mask"],
                    b["labels"],
                    roots,
                    b["model_config"]["num_layers"],
                    self.batch_size,
                )
            # CSX requires one common leading batch dimension on every input.
            batch["node_features"] = tuple(
                x.reshape(self.batch_size, -1, x.shape[-1])
                for x in batch["node_features"]
            )
            batch["neighbor_masks"] = tuple(
                x.reshape(self.batch_size, -1, x.shape[-1])
                for x in batch["neighbor_masks"]
            )
            return pack_batch(batch)
        return pack_batch(
            prepare(
                b["features"],
                b["x_mean"],
                b["neighbors"],
                b["mask"],
                b["labels"],
                roots,
                self.store,
                self.batch_size,
            )
        )


def input_fn(bundle_path, snapshot_path, mode, batch_size, epoch, seed):
    # One streaming worker is intentional for the first admission experiment.
    # Reject accidental multi-worker duplication before any learning result.
    import cerebras.pytorch as cstorch

    if cstorch.distributed.is_streamer() and cstorch.distributed.num_streamers() != 1:
        raise RuntimeError(
            "This pilot requires one input worker; multi-worker sharding is not implemented"
        )
    dataset = SnapshotDataset(bundle_path, snapshot_path, mode, batch_size, epoch, seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=0
    )


def make_snapshot(model, bundle, path, epoch, weight_step, source_checkpoint, device):
    start = time.perf_counter()
    model.to(device)
    n, m = bundle["neighbors"].to(device), bundle["mask"].to(device)
    histories = refresh(model, bundle["features"].to(device), n, m)
    means = [neighbor_mean(h, n, m).cpu() for h in histories]
    histories = [h.cpu() for h in histories]
    snapshot = dict(
        histories=histories,
        means=means,
        epoch=epoch,
        weight_step=weight_step,
        source_checkpoint=source_checkpoint,
    )
    torch.save(snapshot, path)
    return {
        "refresh_and_save_s": time.perf_counter() - start,
        "history_bytes": tensor_bytes(histories),
        "cached_means_bytes": tensor_bytes(means),
        "epoch": epoch,
        "weight_step": weight_step,
        "refresh_device": device,
    }


def prepare_bundle(args):
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.data_dir is None:
        raise ValueError("Real OGB preparation requires --data-dir")
    from benchmark_scripts.research.history_embedding.data import load_graph

    graph = load_graph(args.data_dir, args.dataset, args.fanout)
    features, neighbors, mask, labels = (
        graph[k] for k in ("features", "neighbors", "mask", "labels")
    )
    train_ids, source = graph["splits"]["train"], graph["source"]
    config = dict(
        input_dim=features.shape[1],
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_classes=source["classes"],
        dropout=args.dropout,
    )
    model = HistorySAGE(**config)
    bundle = dict(
        features=features,
        neighbors=neighbors,
        mask=mask,
        labels=labels,
        x_mean=neighbor_mean(features, neighbors, mask),
        train_ids=train_ids,
        splits=graph["splits"],
        adjacency_indptr=torch.from_numpy(graph["adjacency"].indptr),
        adjacency_indices=torch.from_numpy(graph["adjacency"].indices),
        model_config=config,
        initial_model=model.state_dict(),
        source=source,
    )
    torch.save(bundle, args.output / "bundle.pt")
    metrics = make_snapshot(
        model,
        bundle,
        args.output / "snapshot-0.pt",
        0,
        0,
        "initial_model",
        args.refresh_device,
    )
    (args.output / "prepare.json").write_text(
        json.dumps(dict(source=source, config=config, **metrics), indent=2) + "\n"
    )


def refresh_snapshot(args):
    import cerebras.pytorch as cstorch

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    checkpoint = cstorch.load(str(args.checkpoint))
    if checkpoint["next_epoch"] != args.epoch:
        raise ValueError("Checkpoint and next snapshot epoch disagree")
    model = HistorySAGE(**bundle["model_config"])
    model.load_state_dict(checkpoint["model"])
    if args.output.exists():
        raise FileExistsError(args.output)
    metrics = make_snapshot(
        model,
        bundle,
        args.output,
        args.epoch,
        checkpoint["weight_step"],
        str(args.checkpoint.resolve()),
        args.refresh_device,
    )
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")


def train(args):
    import cerebras.pytorch as cstorch

    if args.export_history and args.mode != "history":
        raise ValueError("History export requires history mode")
    bundle_path = str(args.bundle.resolve())
    snapshot_path = str(args.snapshot.resolve()) if args.snapshot else None
    if args.mode == "history" and not snapshot_path:
        raise ValueError("History training requires a snapshot")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    dataset = SnapshotDataset(
        bundle_path, snapshot_path, args.mode, args.batch_size, args.epoch, args.seed
    )
    steps = args.steps or len(dataset)
    if steps < 1 or steps > len(dataset):
        raise ValueError("Steps must fit one epoch; input repetition is not allowed")
    if args.epoch > 0 and not args.checkpoint:
        raise ValueError("Subsequent epochs require an optimizer/model checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed + args.epoch)
    backend_options = dict(artifact_dir=str(args.output / "runtime"))
    if args.backend == "CSX":
        if not args.cluster_config:
            raise ValueError(
                "CSX needs --cluster-config JSON with the site's address and mounts"
            )
        cluster_options = json.loads(args.cluster_config.read_text())
        if "REPLACE_WITH" in json.dumps(cluster_options):
            raise ValueError(
                "Replace the example cluster paths/address with the site's configuration"
            )
        if (
            cluster_options.get("num_workers_per_csx", 1) != 1
            or cluster_options.get("num_csx", 1) != 1
        ):
            raise ValueError("Pilot requires num_csx=1 and num_workers_per_csx=1")
        cluster_options.update(num_csx=1, num_workers_per_csx=1)
        backend_options.update(
            cluster_config=cstorch.distributed.ClusterConfig(**cluster_options),
            compile_dir=str(args.compile_dir.resolve()),
            compile_only=args.compile_only,
        )
    elif args.compile_only:
        raise ValueError("compile-only is a CSX operation")
    backend = cstorch.backend(args.backend, **backend_options)
    checkpoint = cstorch.load(str(args.checkpoint)) if args.checkpoint else None
    if checkpoint and checkpoint["next_epoch"] != args.epoch:
        raise ValueError("Checkpoint epoch mismatch")
    if args.mode == "history":
        snap = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        expected_source = (
            str(args.checkpoint.resolve()) if args.checkpoint else "initial_model"
        )
        if snap["source_checkpoint"] != expected_source:
            raise ValueError("History snapshot was not refreshed from this checkpoint")
    weight_step = checkpoint["weight_step"] if checkpoint else 0
    with backend.device:
        model = StaticModel(**bundle["model_config"])
    model.load_state_dict(
        checkpoint["model"] if checkpoint else bundle["initial_model"]
    )
    model.train()
    compiled = cstorch.compile(model, backend)
    optimizer = cstorch.optim.AdamW(
        model.parameters(), lr=args.lr, eps=1e-8, weight_decay=0.0
    )
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    loader = cstorch.utils.data.DataLoader(
        input_fn,
        bundle_path,
        snapshot_path,
        args.mode,
        args.batch_size,
        args.epoch,
        args.seed,
    )
    records = []
    captured = set()

    @cstorch.step_closure
    def capture(loss, root_ids, valid, labels, hidden):
        ids = root_ids[valid.bool()].long().cpu()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        row = dict(
            capture=len(records) + 1,
            loss=float(loss),
            supervised=int((valid.bool() & (labels >= 0)).sum()),
            valid_roots=len(ids),
            elapsed_since_execute_s=time.perf_counter() - start,
        )
        if args.export_history:
            duplicate = captured.intersection(ids.tolist())
            if duplicate:
                raise ValueError("Repeated root IDs within one execution epoch")
            captured.update(ids.tolist())
            outputs = [h[valid.bool()].cpu() for h in hidden]
            if not all(torch.isfinite(h).all() for h in outputs):
                raise ValueError("Nonfinite exported history")
            torch.save(
                {"root_ids": ids, "histories": outputs},
                args.output / f"capture-{len(records):05d}.pt",
            )
            row["history_bytes"] = tensor_bytes(outputs)
        records.append(row)
        with (args.output / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    @cstorch.checkpoint_closure
    def save_checkpoint():
        cstorch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "weight_step": weight_step + steps,
                "next_epoch": args.epoch + 1,
            },
            str(args.output / "checkpoint.mdl"),
        )

    @cstorch.trace
    def training_step(batch):
        optimizer.zero_grad()
        logits, hidden = compiled(batch)
        loss = masked_loss(logits, batch["labels"][0], batch["valid"][0])
        loss.backward()
        optimizer.step()
        capture(
            loss,
            batch["root_ids"][0],
            batch["valid"][0],
            batch["labels"][0],
            hidden if args.export_history else (),
        )
        save_checkpoint()
        return loss

    start = time.perf_counter()
    executor = cstorch.utils.data.DataExecutor(
        loader,
        num_steps=steps,
        checkpoint_steps=steps,
        activation_steps=1 if args.export_history else args.activation_steps,
    )
    iterations = 0
    for batch in executor:
        training_step(batch)
        iterations += 1
    if not args.compile_only and not (args.output / "checkpoint.mdl").exists():
        raise RuntimeError("Execution ended without its final checkpoint")
    observed_rate = None
    effective_activation_steps = 1 if args.export_history else args.activation_steps
    if (
        effective_activation_steps == 1
        and len(records) == steps
        and 0 < args.warmup_steps < steps
    ):
        elapsed = (
            records[-1]["elapsed_since_execute_s"]
            - records[args.warmup_steps - 1]["elapsed_since_execute_s"]
        )
        observed_rate = (
            sum(r["supervised"] for r in records[args.warmup_steps :]) / elapsed
        )
    result = dict(
        status="compiled_not_executed" if args.compile_only else "executed",
        backend=args.backend,
        mode=args.mode,
        seed=args.seed,
        learning_rate=args.lr,
        source=bundle["source"],
        bundle=bundle_path,
        snapshot=snapshot_path,
        model_config=bundle["model_config"],
        export_history=args.export_history,
        epoch=args.epoch,
        planned_steps=steps,
        python_iterations=iterations,
        captured_steps=len(records),
        elapsed_s=time.perf_counter() - start,
        complete_epoch=(steps == len(dataset)),
        full_batch_input_bytes=tensor_bytes(dataset[0]),
        batch_size=args.batch_size,
        sdk_batch_size=1,
        client_observed_targets_per_s_after_warmup=observed_rate,
        warmup_steps=args.warmup_steps,
        timing_scope="client activation receipts; includes fetch and capture-file overhead, not isolated device time",
        optimizer="cstorch.optim.AdamW eps=1e-8 weight_decay=0",
        sdk=cstorch.__version__,
        cpu_validation_is_not_csx_compilation=True,
        snapshot_policy="immutable within job; refreshed from final checkpoint between jobs",
        online_worker_feedback=False,
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "refresh"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--dataset", choices=["ogbn-arxiv", "ogbn-products"], default="ogbn-arxiv"
    )
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--refresh-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--backend", choices=["CPU", "CSX"], default="CPU")
    parser.add_argument(
        "--mode", choices=["tree", "tree_mean", "history"], default="history"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--export-history", action="store_true")
    parser.add_argument("--activation-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--cluster-config", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--compile-dir", type=Path, default=Path("/tmp/history-embedding-compile")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    {"prepare": prepare_bundle, "train": train, "refresh": refresh_snapshot}[
        args.action
    ](args)


if __name__ == "__main__":
    main()
