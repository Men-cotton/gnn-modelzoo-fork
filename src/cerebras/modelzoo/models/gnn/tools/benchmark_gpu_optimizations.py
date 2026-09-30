"""Bounded synthetic fixed-shape CUDA comparison; never downloads or falls back."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import traceback

import torch

from ..gpu_policy import adamw_param_groups

from ..architectures.graphsage import GraphSAGE
from ..data_processing.batches import GraphSAGEBatch
from ..data_processing.samplers.neighbor_tree import GraphSAGENeighborSamplerDataset
from ..gpu_optimizations import enable_topology_cache, fuse_depths
from ..task.loss import masked_classification_loss


def summary(values):
    return dict(
        mean=statistics.mean(values),
        median=statistics.median(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0,
        min=min(values),
        max=max(values),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", choices=("smoke", "representative"), default="smoke")
    p.add_argument("--timeout-sec", type=int, default=150)
    args = p.parse_args()
    if not 1 <= args.timeout_sec <= 180:
        p.error("timeout-sec must be 1..180")
    paths = [
        Path.cwd(),
        args.output.resolve(),
        Path(os.environ.get("TMPDIR", "/tmp")).resolve(),
    ]
    if any(str(x).startswith(("/mnt/", "//")) for x in paths):
        p.error("all benchmark I/O must be on the WSL filesystem")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = args.output.open("x")
    result = {
        "case": args.case,
        "status": "started",
        "torch": torch.__version__,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                Path(__file__).parents[1] / "gpu_optimizations.py",
                Path(__file__).parents[1] / "architectures/graphsage.py",
                Path(__file__).parents[1] / "data_processing/samplers/neighbor_tree.py",
            ]
        },
        "cwd": str(Path.cwd()),
        "tmpdir": str(paths[-1]),
        "results": [],
    }

    def timed_out(*_):
        raise TimeoutError("bounded benchmark time limit reached")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(args.timeout_sec)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(42)
        b, fanouts, d, h, n, steps = (
            (8, [3, 2, 2], 16, 32, 127, 4)
            if args.case == "smoke"
            else (64, [5, 3, 2], 64, 128, 509, 20)
        )
        warmup, repeats, dropout = (3, 3, 0.5) if args.case == "smoke" else (10, 4, 0.5)
        features = torch.randn(n, d)
        # Directed input treated as implicit undirected. Include degree zero.
        rows = torch.arange(n - 1).repeat_interleave(6)
        cols = (rows + torch.arange(1, 7).repeat(n - 1)) % (n - 1)
        edges = torch.stack((rows, cols))
        labels = torch.arange(n) % 4
        split = torch.ones(n, dtype=torch.bool)

        def make_dataset():
            return GraphSAGENeighborSamplerDataset(
                features,
                edges,
                labels,
                split,
                fanouts,
                b,
                shuffle=True,
                pad_id=0,
                seed=42,
                edge_index_is_undirected=False,
            )

        template = GraphSAGE(d, h, 3, dropout, "mean", 4)
        adamw = dict(lr=0.003, eps=1e-6, betas=(0.9, 0.999), weight_decay=0.01)
        result.update(
            device=torch.cuda.get_device_name(0),
            precision="fp32",
            tf32=False,
            batch_size=b,
            fanouts=fanouts,
            input_dim=d,
            hidden_dim=h,
            nodes=n,
            bias_weight_decay=0.0,
            dropout=dropout,
            warmup=warmup,
            steps=steps,
            repeats=repeats,
            adamw=adamw,
            boundary="serial host sample/gather + synchronous H2D + CUDA forward/backward/AdamW; excludes setup/cache/warmup",
            static_batch_cache_size=0,
            num_workers=0,
            variants=["baseline", "topology", "fusion", "combined"],
        )
        # Rotate execution order across repeats; each variant sees identical seeds/steps.
        datasets = {name: make_dataset() for name in result["variants"]}
        cache_info = {}
        for name in ("topology", "combined"):
            cache = enable_topology_cache(datasets[name])
            cache_info[name] = dict(
                bytes=cache.nbytes, build_seconds=cache.build_seconds
            )
        result["cache"] = cache_info
        for repeat in range(repeats):
            order = result["variants"][repeat:] + result["variants"][:repeat]
            for name in order:
                ds = datasets[name]
                model = copy.deepcopy(template).cuda().train()
                if name in ("fusion", "combined"):
                    fuse_depths(model)
                optimizer = torch.optim.AdamW(adamw_param_groups(model), **adamw)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = {
                    key: []
                    for key in (
                        "host_ms",
                        "h2d_wall_ms",
                        "device_stage_wall_ms",
                        "e2e_ms",
                        "h2d_cuda_ms",
                        "device_cuda_ms",
                    )
                }
                losses = []
                target_count = 0
                payload_digest = hashlib.sha256()
                transfer_start_event = torch.cuda.Event(enable_timing=True)
                transfer_end_event = torch.cuda.Event(enable_timing=True)
                device_start_event = torch.cuda.Event(enable_timing=True)
                device_end_event = torch.cuda.Event(enable_timing=True)
                for step in range(warmup + steps):
                    # Identical dropout RNG per logical step, outside timing.
                    torch.manual_seed(1000 + repeat * 100 + step)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    payload = ds[step // len(ds), step % len(ds)]
                    batch = GraphSAGEBatch.from_payload(payload)
                    host_end = time.perf_counter()
                    transfer_start_event.record()
                    batch = batch.to(torch.device("cuda"))
                    transfer_end_event.record()
                    torch.cuda.synchronize()
                    transfer_end = time.perf_counter()
                    device_start_event.record()
                    optimizer.zero_grad(set_to_none=True)
                    loss = masked_classification_loss(
                        model(batch), batch.labels, batch.target_mask
                    )
                    loss.backward()
                    optimizer.step()
                    device_end_event.record()
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    if step >= warmup:
                        for key, value in zip(
                            list(times)[:4],
                            (
                                host_end - start,
                                transfer_end - host_end,
                                end - transfer_end,
                                end - start,
                            ),
                        ):
                            times[key].append(value * 1000)
                        times["h2d_cuda_ms"].append(
                            transfer_start_event.elapsed_time(transfer_end_event)
                        )
                        times["device_cuda_ms"].append(
                            device_start_event.elapsed_time(device_end_event)
                        )
                        losses.append(loss.item())
                        target_count += int(payload["target_mask"].sum())
                        for key in sorted(payload):
                            tensors = (
                                payload[key]
                                if isinstance(payload[key], list)
                                else [payload[key]]
                            )
                            for tensor in tensors:
                                payload_digest.update(tensor.numpy().tobytes())
                result["results"].append(
                    dict(
                        variant=name,
                        repeat=repeat,
                        samples=times,
                        summary={k: summary(v) for k, v in times.items()},
                        losses=losses,
                        valid_targets=target_count,
                        payload_sha256=payload_digest.hexdigest(),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    )
                )
                del model, optimizer, batch, loss
                torch.cuda.empty_cache()
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        json.dump(result, stream, indent=2)
        stream.write("\n")
        stream.close()
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
