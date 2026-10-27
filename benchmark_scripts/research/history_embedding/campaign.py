"""Run separate train/host-refresh processes and include their full wall time.

This is the correctness-first CSX epoch-snapshot path. Frequent job startup and
checkpoint costs are deliberately visible; they must not be called free I/O.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
from benchmark_scripts.research.runtime import module_command
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Prepared bundle.pt and snapshot-0.pt directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["CPU", "CSX"], default="CPU")
    parser.add_argument(
        "--mode", choices=["tree", "tree_mean", "history"], default="history"
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--cluster-config", type=Path)
    parser.add_argument("--evaluation-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.backend == "CSX" and not args.cluster_config:
        raise ValueError("CSX requires --cluster-config")
    entry = "benchmark_scripts.research.history_embedding.hardware"
    bundle = args.input.resolve() / "bundle.pt"
    snapshot = args.input.resolve() / "snapshot-0.pt"
    checkpoint = None
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=False)
    overall = time.perf_counter()
    events = []
    for epoch in range(args.epochs):
        destination = args.output.resolve() / f"epoch-{epoch}"
        command = module_command(entry) + [
            "train",
            "--backend",
            args.backend,
            "--mode",
            args.mode,
            "--bundle",
            str(bundle),
            "--output",
            str(destination),
            "--epoch",
            str(epoch),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--lr",
            str(args.lr),
            "--compile-dir",
            str(args.output.resolve() / "compile"),
        ]
        if args.mode == "history":
            command += ["--snapshot", str(snapshot)]
        if checkpoint:
            command += ["--checkpoint", str(checkpoint)]
        if args.cluster_config:
            command += ["--cluster-config", str(args.cluster_config.resolve())]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            start = time.perf_counter()
            with (args.output / f"train-{epoch}.log").open("x") as stream:
                subprocess.run(
                    command, check=True, stdout=stream, stderr=subprocess.STDOUT
                )
            events.append(
                dict(
                    epoch=epoch,
                    phase="train_process",
                    seconds=time.perf_counter() - start,
                )
            )
        checkpoint = destination / "checkpoint.mdl"
        evaluation = module_command(
            "benchmark_scripts.research.history_embedding.evaluate"
        ) + [
            "--bundle",
            str(bundle),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(destination / "validation.json"),
            "--device",
            args.evaluation_device,
        ]
        print(shlex.join(evaluation), flush=True)
        if not args.dry_run:
            start = time.perf_counter()
            with (args.output / f"validate-{epoch}.log").open("x") as stream:
                subprocess.run(
                    evaluation, check=True, stdout=stream, stderr=subprocess.STDOUT
                )
            events.append(
                dict(
                    epoch=epoch,
                    phase="full_neighbor_validation_process",
                    seconds=time.perf_counter() - start,
                    metrics=json.loads((destination / "validation.json").read_text()),
                )
            )
        if args.mode == "history" and epoch + 1 < args.epochs:
            snapshot = args.output.resolve() / f"snapshot-{epoch + 1}.pt"
            command = module_command(entry) + [
                "refresh",
                "--bundle",
                str(bundle),
                "--checkpoint",
                str(checkpoint),
                "--epoch",
                str(epoch + 1),
                "--output",
                str(snapshot),
            ]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                start = time.perf_counter()
                with (args.output / f"refresh-{epoch + 1}.log").open("x") as stream:
                    subprocess.run(
                        command, check=True, stdout=stream, stderr=subprocess.STDOUT
                    )
                events.append(
                    dict(
                        epoch=epoch + 1,
                        phase="host_refresh_process",
                        seconds=time.perf_counter() - start,
                    )
                )
    if not args.dry_run:
        result = dict(
            backend=args.backend,
            mode=args.mode,
            epochs=args.epochs,
            wall_s_including_process_startup_and_checkpoints=time.perf_counter()
            - overall,
            events=events,
            initial_prepare_excluded=True,
        )
        (args.output / "campaign.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
