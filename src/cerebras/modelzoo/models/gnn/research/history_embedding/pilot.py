"""Serial compiler/transfer pilot; invoke through the existing tmux launcher."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from cerebras.modelzoo.tools import benchmark_launcher
from cerebras.modelzoo.tools.job_source import require_pinned
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["CPU", "CSX"], default="CSX")
    parser.add_argument("--phase", choices=["compile", "execute"], required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["tree", "tree_mean"],
        choices=["tree", "tree_mean", "history", "history-export"],
    )
    parser.add_argument("--dry-run", action="store_true")
    benchmark_launcher.add_arguments(parser)
    args = parser.parse_args(argv)
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.phase == "compile" and args.backend != "CSX":
        raise ValueError("CPU is not a CSX compiler check")
    if args.batch_size < 1 or args.steps < 1 or not 0 <= args.warmup_steps < args.steps:
        parser.error("positive batch size/steps and warmup < steps required")
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
            name="history-pilot",
            session=args.tmux_session,
        )
    entry = "cerebras.modelzoo.models.gnn.research.history_embedding.hardware"
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=False)
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
                str(args.input.resolve() / "bundle.pt"),
                "--snapshot",
                str(args.input.resolve() / "snapshot-0.pt"),
                "--output",
                str(args.output.resolve() / name),
                "--batch-size",
                str(args.batch_size),
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
            if args.phase == "compile":
                command.append("--compile-only")
            if args.dry_run:
                import shlex

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
                    dict(backend=args.backend, phase=args.phase, cases=records),
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(record), flush=True)
            if completed.returncode:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
