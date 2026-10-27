"""Matched local experiments; CPU timings and CUDA timings are labeled separately."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from cerebras.modelzoo.models.gnn.research import DATA

from cerebras.modelzoo.models.gnn.research.history_embedding.core import (
    HistorySAGE,
    HistoryStore,
    infer,
    masked_loss,
    neighbor_mean,
    prepare,
    prepare_tree,
    prepare_tree_mean,
    refresh,
    tensor_bytes,
    to_device,
)
from cerebras.modelzoo.models.gnn.research.history_embedding.data import (
    full_neighbor_infer,
    load_graph,
)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def trial(args, graph, mode, seed):
    device = torch.device(args.device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HistorySAGE(
        graph["source"]["input_dim"],
        args.hidden,
        args.layers,
        graph["source"]["classes"],
        args.dropout,
    ).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    x, n, m, labels = (graph[k] for k in ("features", "neighbors", "mask", "labels"))
    gx, gn, gm = x.to(device), n.to(device), m.to(device)
    x_mean = neighbor_mean(gx, gn, gm).cpu()
    roots = graph["splits"]["train"]
    store = None
    step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cumulative = 0.0
    best_valid = -1.0
    best_epoch = None
    best_test = None
    output = args.output / f"{mode}-seed{seed}"
    output.mkdir(exist_ok=False, parents=True)
    config = vars(args).copy()
    config["output"], config["data_dir"] = (
        str(config["output"]),
        str(config["data_dir"]),
    )
    metadata = dict(
        config=config,
        source=graph["source"],
        mode=mode,
        seed=seed,
        torch=torch.__version__,
        device=str(device),
        initial_model_sha256=initial_hash,
        device_name=torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        scope="sampled-history variant; native GPU/CPU, not Cerebras",
        float_dtype="float32",
        tf32=torch.backends.cuda.matmul.allow_tf32,
    )
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "epochs.jsonl").open("x") as stream:
        for epoch in range(args.epochs):
            model.train()
            sync(device)
            start = time.perf_counter()
            refresh_s = 0.0
            if not mode.startswith("tree") and (
                epoch % (5 if mode == "snapshot5" else 1) == 0
            ):
                t = time.perf_counter()
                store = HistoryStore(refresh(model, gx, gn, gm), step=step)
                if mode.startswith("snapshot"):
                    # GPU global means are copied once; input preparation remains host-side.
                    store.means = [
                        neighbor_mean(h.to(device), gn, gm).cpu() for h in store.values
                    ]
                sync(device)
                refresh_s = time.perf_counter() - t
            generator = torch.Generator().manual_seed(seed * 1000 + epoch)
            order = roots[torch.randperm(len(roots), generator=generator)]
            totals = dict(
                prepare_s=0.0,
                h2d_s=0.0,
                compute_s=0.0,
                history_write_s=0.0,
                input_bytes=0,
                output_history_bytes=0,
                age_sum=0,
                age_count=0,
                age_max=0,
                loss_sum=0.0,
                correct=0,
                supervised=0,
            )
            for batch_index, batch_roots in enumerate(order.split(args.batch_size)):
                if mode == "snapshot32" and batch_index and batch_index % 32 == 0:
                    t = time.perf_counter()
                    store = HistoryStore(refresh(model, gx, gn, gm), step=step)
                    store.means = [
                        neighbor_mean(h.to(device), gn, gm).cpu() for h in store.values
                    ]
                    sync(device)
                    refresh_s += time.perf_counter() - t
                t = time.perf_counter()
                if mode == "tree":
                    batch = prepare_tree(
                        x, n, m, labels, batch_roots, args.layers, args.batch_size
                    )
                elif mode == "tree_mean":
                    batch = prepare_tree_mean(
                        x,
                        x_mean,
                        n,
                        m,
                        labels,
                        batch_roots,
                        args.layers,
                        args.batch_size,
                    )
                else:
                    batch = prepare(
                        x, x_mean, n, m, labels, batch_roots, store, args.batch_size
                    )
                    if store.version is not None:
                        boundary = (
                            m[batch_roots] & ~batch["local_mask"][: len(batch_roots)]
                        )
                        age = (step - store.version[n[batch_roots]])[boundary]
                        totals["age_sum"] += int(age.sum())
                        totals["age_count"] += age.numel()
                        totals["age_max"] = max(
                            totals["age_max"], int(age.max()) if age.numel() else 0
                        )
                totals["input_bytes"] += tensor_bytes(batch)
                totals["prepare_s"] += time.perf_counter() - t
                t = time.perf_counter()
                batch = to_device(batch, device)
                sync(device)
                totals["h2d_s"] += time.perf_counter() - t
                t = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                logits, hidden = model(batch)
                loss = masked_loss(logits, batch["labels"], batch["valid"])
                loss.backward()
                optimizer.step()
                sync(device)
                totals["compute_s"] += time.perf_counter() - t
                # Embeddings are from pre-update weights: tag that exact weight version.
                if mode == "online1":
                    t = time.perf_counter()
                    store.write(
                        batch_roots, [h[: len(batch_roots)] for h in hidden], step
                    )
                    totals["output_history_bytes"] += sum(
                        h[: len(batch_roots)].numel() * h.element_size() for h in hidden
                    )
                    totals["history_write_s"] += time.perf_counter() - t
                step += 1
                totals["loss_sum"] += loss.item() * len(batch_roots)
                totals["correct"] += int(
                    (
                        logits[: len(batch_roots)].argmax(1)
                        == batch["labels"][: len(batch_roots)]
                    ).sum()
                )
                totals["supervised"] += len(batch_roots)
            sync(device)
            epoch_s = time.perf_counter() - start
            cumulative += epoch_s
            eval_start = time.perf_counter()
            logits = infer(model, gx, gn, gm).cpu()
            sampled_valid = float(
                (
                    logits[graph["splits"]["valid"]].argmax(1)
                    == labels[graph["splits"]["valid"]]
                )
                .float()
                .mean()
            )
            full_logits = full_neighbor_infer(model, x, graph["adjacency"], device)
            valid = float(
                (
                    full_logits[graph["splits"]["valid"]].argmax(1)
                    == labels[graph["splits"]["valid"]]
                )
                .float()
                .mean()
            )
            # Test is recorded only for the checkpoint selected by validation, never used in optimization.
            if valid > best_valid:
                best_valid, best_epoch = valid, epoch + 1
                best_test = float(
                    (
                        full_logits[graph["splits"]["test"]].argmax(1)
                        == labels[graph["splits"]["test"]]
                    )
                    .float()
                    .mean()
                )
                torch.save(
                    {
                        "model": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        "epoch": epoch + 1,
                    },
                    output / "best.pt",
                )
            sync(device)
            row = dict(
                epoch=epoch + 1,
                step=step,
                epoch_s=epoch_s,
                cumulative_train_s=cumulative,
                refresh_s=refresh_s,
                evaluation_s=time.perf_counter() - eval_start,
                train_loss=totals["loss_sum"] / totals["supervised"],
                train_accuracy=totals["correct"] / totals["supervised"],
                sampled_valid_accuracy=sampled_valid,
                full_valid_accuracy=valid,
                best_full_valid_accuracy=best_valid,
                best_epoch=best_epoch,
                boundary_age_mean_steps=totals["age_sum"] / max(1, totals["age_count"]),
                peak_cuda_bytes=torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else None,
                **totals,
            )
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(
                json.dumps(
                    dict(
                        mode=mode,
                        seed=seed,
                        epoch=epoch + 1,
                        loss=round(row["train_loss"], 4),
                        valid=round(valid, 4),
                        seconds=round(epoch_s, 2),
                    )
                ),
                flush=True,
            )
    summary = dict(
        mode=mode,
        seed=seed,
        epochs=args.epochs,
        best_epoch=best_epoch,
        best_full_valid_accuracy=best_valid,
        test_at_best_valid=best_test,
        total_train_refresh_s=cumulative,
        initial_model_sha256=initial_hash,
        final_model_sha256=state_hash(model),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument(
        "--dataset", default="ogbn-arxiv", choices=["ogbn-arxiv", "ogbn-products"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["tree", "snapshot1", "snapshot5", "online1"],
        choices=[
            "tree",
            "tree_mean",
            "snapshot1",
            "snapshot5",
            "snapshot32",
            "online1",
        ],
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no implicit CPU fallback")
    graph = load_graph(args.data_dir, args.dataset, args.fanout)
    for seed in args.seeds:
        for mode in args.modes:
            trial(args, graph, mode, seed)


if __name__ == "__main__":
    main()
