"""Run separate train/host-refresh processes and include their full wall time.

This is the correctness-first CSX epoch-snapshot path. Frequent job startup and
checkpoint costs are deliberately visible; they must not be called free I/O.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from cerebras.modelzoo.models.gnn.tools.history_embedding.hardware import (
    add_prepare_arguments,
    apply_prepare_defaults,
)
from cerebras.modelzoo.tools import benchmark_launcher
from cerebras.modelzoo.tools.job_source import require_pinned
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Reuse prepared bundle.pt and snapshot-0.pt directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["CPU", "CSX"], default="CPU")
    parser.add_argument(
        "--mode", choices=["tree", "tree_mean", "history"], default="history"
    )
    add_prepare_arguments(parser)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--evaluation-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    benchmark_launcher.add_arguments(parser)
    args = parser.parse_args(argv)
    config = apply_prepare_defaults(args)
    if args.lr is None:
        args.lr = config["init"]["optimizer"]["AdamW"]["learning_rate"]
    if args.input:
        args.input = args.input.resolve()
    args.output = args.output.resolve()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.backend == "CSX" and not args.dry_run:
        require_pinned()
    if args.detach and not args.dry_run:
        return benchmark_launcher.launch(
            [sys.executable, "-u", "-m", __spec__.name, *argv, "--foreground"],
            args.output.with_name(args.output.name + "-launcher"),
            name="history-campaign",
            session=args.tmux_session,
        )
    entry = "cerebras.modelzoo.models.gnn.tools.history_embedding.hardware"
    checkpoint = None
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=False)
    input_dir = args.input or args.output / "input"
    if not args.input:
        command = [
            sys.executable,
            "-u",
            "-m",
            entry,
            "prepare",
            "--output",
            str(input_dir),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--dataset",
            args.dataset,
            "--fanout",
            str(args.fanout),
            "--hidden",
            str(args.hidden),
            "--layers",
            str(args.layers),
            "--dropout",
            str(args.dropout),
            "--seed",
            str(args.seed),
            "--refresh-device",
            args.refresh_device,
        ]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            with (args.output / "prepare.log").open("x") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
    bundle = input_dir / "bundle.pt"
    snapshot = input_dir / "snapshot-0.pt"
    overall = time.perf_counter()
    events = []
    for epoch in range(args.epochs):
        destination = args.output.resolve() / f"epoch-{epoch}"
        command = [sys.executable, "-u", "-m", entry] + [
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
        evaluation = [
            sys.executable,
            "-u",
            "-m",
            "cerebras.modelzoo.models.gnn.tools.history_embedding.evaluate",
        ] + [
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
            command = [sys.executable, "-u", "-m", entry] + [
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
    raise SystemExit(main())
