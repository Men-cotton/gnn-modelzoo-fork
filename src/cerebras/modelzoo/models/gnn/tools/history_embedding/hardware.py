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
import numpy as np

from cerebras.modelzoo.models.gnn.gpu_measurements import logical_tensor_bytes
from cerebras.modelzoo.models.gnn.tools.csx_backend import (
    configure_precision,
    configure_optimizer,
    loader_options,
    load_config,
    WARMUP_STEPS,
)
from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
    EpochBatchSampler,
)

from cerebras.modelzoo.models.gnn.tools.download_datasets import DEFAULT_ROOT as DATA

from cerebras.modelzoo.models.gnn.tools.history_embedding.core import (
    HistorySAGE,
    HistoryStore,
    masked_loss,
    neighbor_mean,
    prepare,
    prepare_tree,
    prepare_tree_mean,
    refresh,
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


class StaticModel(HistorySAGE):
    def forward(self, batch):
        batch = {
            k: tuple(x[0] for x in v) if isinstance(v, (tuple, list)) else v[0]
            for k, v in batch.items()
        }
        return super().forward(batch)


class SnapshotDataset(torch.utils.data.Dataset):
    def __init__(self, bundle_path, snapshot_path, mode, batch_size, epoch, seed):
        self.bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
        self.mode, self.batch_size = mode, batch_size
        self.seed, self.start_epoch, self.order_epoch = seed, epoch, epoch
        self.store = None
        if mode == "history":
            snap = torch.load(snapshot_path, map_location="cpu", weights_only=True)
            if snap["epoch"] != epoch:
                raise ValueError(
                    "Snapshot epoch does not match requested training epoch"
                )
            self.store = HistoryStore(snap["histories"], snap["weight_step"])
            self.store.means = snap["means"]
            self.source_checkpoint = snap["source_checkpoint"]
        ids = self.bundle["train_ids"]
        self.order = ids[
            torch.from_numpy(np.random.default_rng(seed + epoch).permutation(len(ids)))
        ]

    def __len__(self):
        return math.ceil(len(self.order) / self.batch_size)

    def __getitem__(self, index):
        epoch = self.start_epoch
        if isinstance(index, tuple):
            offset, index = index
            epoch += offset
        if epoch != self.order_epoch:
            ids = self.bundle["train_ids"]
            self.order = ids[
                torch.from_numpy(
                    np.random.default_rng(self.seed + epoch).permutation(len(ids))
                )
            ]
            self.order_epoch = epoch
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


def input_fn(bundle_path, snapshot_path, mode, batch_size, epoch, seed, backend="CSX"):
    # One streaming worker is intentional for the first admission experiment.
    # Reject accidental multi-worker duplication before any learning result.
    import cerebras.pytorch as cstorch

    if cstorch.distributed.is_streamer() and cstorch.distributed.num_streamers() != 1:
        raise RuntimeError(
            "This pilot requires one input worker; multi-worker sharding is not implemented"
        )
    dataset = SnapshotDataset(bundle_path, snapshot_path, mode, batch_size, epoch, seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        sampler=EpochBatchSampler(dataset),
        **loader_options(backend, dataset.bundle["source"]["dataset"]),
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
        "history_bytes": logical_tensor_bytes(histories),
        "cached_means_bytes": logical_tensor_bytes(means),
        "epoch": epoch,
        "weight_step": weight_step,
        "refresh_device": device,
    }


def prepare_bundle(args):
    torch.manual_seed(args.seed)
    from cerebras.modelzoo.models.gnn.tools.history_embedding.data import load_graph

    graph = load_graph(args.data_dir, args.dataset, args.fanout, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
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
    dataset = SnapshotDataset(
        bundle_path, snapshot_path, args.mode, args.batch_size, args.epoch, args.seed
    )
    bundle = dataset.bundle
    steps = len(dataset) if args.steps is None else args.steps
    if steps < 1:
        raise ValueError("steps must be positive")
    if args.epoch > 0 and not args.checkpoint:
        raise ValueError("Subsequent epochs require an optimizer/model checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed + args.epoch)
    backend_options = dict(artifact_dir=str(args.output / "runtime"))
    if args.backend == "CSX":
        from cerebras.modelzoo.models.gnn.tools.csx_backend import (
            backend_options as csx_backend_options,
        )

        backend_options.update(
            csx_backend_options(
                args.output,
                args.compile_dir,
                [args.bundle, args.snapshot, args.checkpoint],
                args.compile_only,
                dataset=bundle["source"]["dataset"],
            )
        )
    elif args.compile_only:
        raise ValueError("compile-only is a CSX operation")
    backend = cstorch.backend(args.backend, **backend_options)
    precision = configure_precision(args.backend, bundle["source"]["dataset"])
    checkpoint = cstorch.load(str(args.checkpoint)) if args.checkpoint else None
    if checkpoint and checkpoint["next_epoch"] != args.epoch:
        raise ValueError("Checkpoint epoch mismatch")
    if args.mode == "history":
        expected_source = (
            str(args.checkpoint.resolve()) if args.checkpoint else "initial_model"
        )
        if dataset.source_checkpoint != expected_source:
            raise ValueError("History snapshot was not refreshed from this checkpoint")
    weight_step = checkpoint["weight_step"] if checkpoint else 0
    with backend.device:
        model = StaticModel(**bundle["model_config"])
    model.load_state_dict(
        checkpoint["model"] if checkpoint else bundle["initial_model"]
    )
    model.train()
    compiled = cstorch.compile(model, backend)
    optimizer = configure_optimizer(model, bundle["source"]["dataset"], args.lr)
    scaler = (
        cstorch.amp.GradScaler(loss_scale=precision["loss_scaling_factor"])
        if precision["enabled"]
        else None
    )
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler:
            scaler.load_state_dict(checkpoint["grad_scaler"])
    loader = cstorch.utils.data.DataLoader(
        input_fn,
        bundle_path,
        snapshot_path,
        args.mode,
        args.batch_size,
        args.epoch,
        args.seed,
        args.backend,
    )
    records = []
    captured = set()

    @cstorch.step_closure
    def capture(loss, root_ids, valid, labels, hidden, skipped):
        ids = root_ids[valid.bool()].long().cpu()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        row = dict(
            capture=len(records) + 1,
            loss=float(loss),
            supervised=int((valid.bool() & (labels >= 0)).sum()),
            valid_roots=len(ids),
            skipped_optimizer_steps=int(skipped),
            elapsed_since_execute_s=time.perf_counter() - start,
        )
        if args.export_history:
            if len(records) % len(dataset) == 0:
                captured.clear()
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
            row["history_bytes"] = logical_tensor_bytes(outputs)
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
                "next_epoch": args.epoch + steps // len(dataset)
                if steps % len(dataset) == 0
                else None,
                "grad_scaler": scaler.state_dict() if scaler else None,
            },
            str(args.output / "checkpoint.mdl"),
        )

    @cstorch.trace
    def training_step(batch):
        optimizer.zero_grad()
        logits, hidden = compiled(batch)
        loss = masked_loss(logits, batch["labels"][0], batch["valid"][0])
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        capture(
            loss,
            batch["root_ids"][0],
            batch["valid"][0],
            batch["labels"][0],
            hidden if args.export_history else (),
            scaler.get_num_skipped_steps() if scaler else torch.tensor(0),
        )
        save_checkpoint()

    start = time.perf_counter()
    effective_activation_steps = 1 if args.export_history else args.activation_steps
    executor = cstorch.utils.data.DataExecutor(
        loader,
        num_steps=steps,
        checkpoint_steps=steps,
        activation_steps=effective_activation_steps,
    )
    iterations = 0
    for batch in executor:
        training_step(batch)
        iterations += 1
    if not args.compile_only and not (args.output / "checkpoint.mdl").exists():
        raise RuntimeError("Execution ended without its final checkpoint")
    observed_rate = None
    if (
        effective_activation_steps == 1
        and len(records) == steps
        and args.warmup_steps < steps
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
        precision=precision,
        mode=args.mode,
        seed=args.seed,
        learning_rate=optimizer.defaults["lr"],
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
        completed_epochs=steps // len(dataset),
        partial_epoch_steps=steps % len(dataset),
        full_batch_input_bytes=logical_tensor_bytes(dataset[0]),
        batch_size=args.batch_size,
        sdk_batch_size=1,
        client_observed_targets_per_s_after_warmup=observed_rate,
        warmup_steps=args.warmup_steps,
        timing_scope="client activation receipts; includes fetch and capture-file overhead, not isolated device time",
        optimizer={"name": "cstorch.optim.AdamW", "options": optimizer.defaults},
        sdk=cstorch.__version__,
        cpu_validation_is_not_csx_compilation=True,
        snapshot_policy="immutable within job; refreshed from final checkpoint between jobs",
        online_worker_feedback=False,
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def add_prepare_arguments(parser):
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--dataset",
        type=lambda name: "ogbn-" + name.removeprefix("ogbn-"),
        choices=["ogbn-arxiv", "ogbn-products"],
    )
    parser.add_argument("--fanout", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--refresh-device", choices=["cpu", "cuda"])


def apply_prepare_defaults(args, parser):
    args.dataset = args.dataset or "ogbn-arxiv"
    args.data_dir = args.data_dir or DATA
    args.refresh_device = args.refresh_device or "cpu"
    config = load_config(args.dataset)["trainer"]
    model = config["init"]["model"]["architecture"]
    loader = config["fit"]["train_dataloader"]
    for name, value in (
        ("hidden", model["hidden_dim"]),
        ("layers", model["num_layers"]),
        ("dropout", model["dropout"]),
        ("fanout", loader["fanouts"][0]),
        ("batch_size", loader["batch_size"]),
        ("seed", config["init"]["seed"]),
    ):
        if getattr(args, name) is None:
            setattr(args, name, value)
    if min(args.hidden, args.layers, args.fanout, args.batch_size) < 1:
        parser.error("hidden size, layers, fanout and batch size must be positive")
    if args.seed < 0 or not 0 <= args.dropout < 1:
        parser.error("nonnegative seed and dropout in [0, 1) required")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "refresh"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    add_prepare_arguments(parser)
    parser.add_argument("--backend", choices=["CPU", "CSX"], default="CPU")
    parser.add_argument(
        "--mode", choices=["tree", "tree_mean", "history"], default="history"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--export-history", action="store_true")
    parser.add_argument("--activation-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--compile-dir", type=Path)
    args = parser.parse_args()
    if args.action == "train" and args.warmup_steps < 1:
        parser.error("warmup must be positive")
    apply_prepare_defaults(args, parser)
    if args.action == "train" and args.backend == "CSX":
        from cerebras.modelzoo.tools.job_source import require_pinned

        require_pinned()
    torch.set_num_threads(4)
    {"prepare": prepare_bundle, "train": train, "refresh": refresh_snapshot}[
        args.action
    ](args)


if __name__ == "__main__":
    main()
