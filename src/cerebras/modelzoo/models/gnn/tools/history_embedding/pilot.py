"""Prepare and run the serial representation benchmark."""

import argparse
from datetime import datetime, timezone
from cerebras.modelzoo.models.gnn.tools.csx_backend import WARMUP_STEPS
import json
import shlex
from pathlib import Path
import subprocess
import sys

from cerebras.modelzoo.models.gnn.tools.history_embedding.hardware import (
    add_prepare_arguments,
    apply_prepare_defaults,
)
from cerebras.modelzoo.tools import benchmark_launcher
from cerebras.modelzoo.tools.job_plan import ROOT
from cerebras.modelzoo.tools.job_source import require_pinned
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Reuse prepared inputs")
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: model_dirs/history_benchmark/<UTC timestamp>",
    )
    parser.add_argument("--backend", choices=["CPU", "CSX"], default="CSX")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int)
    add_prepare_arguments(parser)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["tree", "tree_mean"],
        choices=["tree", "tree_mean", "history", "history-export"],
    )
    parser.add_argument("--dry-run", action="store_true")
    benchmark_launcher.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.input and any(
        value is not None
        for value in (
            args.dataset,
            args.data_dir,
            args.hidden,
            args.layers,
            args.fanout,
            args.dropout,
            args.refresh_device,
        )
    ):
        parser.error("--input cannot be combined with input preparation options")
    config = apply_prepare_defaults(args, parser)
    if args.steps is None:
        args.steps = config["init"]["loop"]["max_steps"]
    if args.input:
        args.input = args.input.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    args.output = (
        args.output or ROOT / "model_dirs/history_benchmark" / stamp
    ).resolve()
    if len(args.cases) != len(set(args.cases)):
        parser.error("duplicate cases; use --repeats to repeat each case")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.batch_size < 1 or not 0 < args.warmup_steps < args.steps:
        parser.error("positive batch size and 0 < warmup < steps required")
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.backend == "CSX" and not args.dry_run:
        require_pinned()
    if args.detach and not args.dry_run:
        return benchmark_launcher.launch(
            [
                sys.executable,
                "-u",
                "-m",
                __spec__.name,
                *argv,
                "--foreground",
                "--output",
                str(args.output),
            ],
            args.output.with_name(args.output.name + "-launcher"),
            name="history-pilot",
            session=args.tmux_session,
        )
    entry = "cerebras.modelzoo.models.gnn.tools.history_embedding.hardware"
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
    records = []
    cases = [
        (name.removesuffix("-export"), name.endswith("-export")) for name in args.cases
    ]
    for repeat in range(args.repeats):
        for mode, export in cases:
            name = f"{mode}{'-export' if export else ''}-r{repeat}"
            command = [sys.executable, "-u", "-m", entry] + [
                "train",
                "--backend",
                args.backend,
                "--mode",
                mode,
                "--bundle",
                str(input_dir / "bundle.pt"),
                "--snapshot",
                str(input_dir / "snapshot-0.pt"),
                "--output",
                str(args.output.resolve() / name),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--steps",
                str(args.steps),
                "--activation-steps",
                "1",
                "--warmup-steps",
                str(args.warmup_steps),
                "--compile-dir",
                str(args.output.resolve() / "compile"),
            ]
            if export:
                command.append("--export-history")
            if args.dry_run:
                print(shlex.join(command))
                continue
            start = time.perf_counter()
            with (args.output / f"{name}.log").open("x") as log:
                completed = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT
                )
            record = dict(
                case=name,
                command=command,
                returncode=completed.returncode,
                whole_process_s=time.perf_counter() - start,
            )
            records.append(record)
            (args.output / "pilot.json").write_text(
                json.dumps(
                    dict(backend=args.backend, cases=records),
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(record), flush=True)
            if completed.returncode:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
