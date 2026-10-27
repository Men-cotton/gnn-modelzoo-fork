"""Prepare inputs, then run and compare dense/dictionary jobs in one campaign."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import torch
from cerebras.modelzoo.models.gnn.tools.csx_backend import load_config, WARMUP_STEPS
import cerebras.pytorch as cstorch
from cerebras.modelzoo.tools import benchmark_launcher
from cerebras.modelzoo.tools.job_source import require_pinned
from cerebras.modelzoo.models.gnn.tools.download_datasets import DEFAULT_ROOT
from cerebras.modelzoo.models.gnn.tools.feature_sharing.core import (
    FrozenInputs,
    file_hash,
)


def compare(output, steps, rtol, atol):
    dirs = [output / m for m in ("dense", "dictionary")]
    records = [
        [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
        for d in dirs
    ]
    if any(len(rows) != steps for rows in records):
        raise ValueError("incomplete activation capture")
    loss_diff = logit_diff = 0.0
    for index, (a, b) in enumerate(zip(*records)):
        if (
            a["step"],
            a["sample_id"],
            a["valid_targets"],
            a["skipped_optimizer_steps"],
        ) != (
            b["step"],
            b["sample_id"],
            b["valid_targets"],
            b["skipped_optimizer_steps"],
        ):
            raise ValueError("paired input order mismatch")
        torch.testing.assert_close(
            torch.tensor(a["loss"]), torch.tensor(b["loss"]), rtol=rtol, atol=atol
        )
        loss_diff = max(loss_diff, abs(a["loss"] - b["loss"]))
        logits = [
            torch.load(d / f"logits-{index + 1:05d}.pt", weights_only=True).float()
            for d in dirs
        ]
        torch.testing.assert_close(*logits, rtol=rtol, atol=atol)
        logit_diff = max(logit_diff, float((logits[0] - logits[1]).abs().max()))
    checkpoints = [cstorch.load(str(d / "checkpoint.mdl")) for d in dirs]
    if any(c["steps"] != steps for c in checkpoints):
        raise ValueError("checkpoint step mismatch")
    torch.testing.assert_close(
        checkpoints[0]["model"], checkpoints[1]["model"], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        checkpoints[0]["optimizer"], checkpoints[1]["optimizer"], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        checkpoints[0]["grad_scaler"], checkpoints[1]["grad_scaler"], rtol=0, atol=0
    )
    parameter_diff = max(
        float((checkpoints[0]["model"][n] - checkpoints[1]["model"][n]).abs().max())
        for n in checkpoints[0]["model"]
    )
    return dict(
        rtol=rtol,
        atol=atol,
        max_loss_absolute_difference=loss_diff,
        max_logit_absolute_difference=logit_diff,
        max_parameter_absolute_difference=parameter_diff,
        optimizer_state_compared=True,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="Reuse a prepared frozen window")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--dataset", choices=["arxiv", "products"], default="arxiv")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--fanouts", type=int, nargs="+")
    p.add_argument("--hidden", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--capacity", type=int)
    p.add_argument("--backend", choices=["CPU", "CSX"], default="CSX")
    p.add_argument("--compile-dir", type=Path)
    p.add_argument("--steps", type=int)
    p.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--rtol", type=float)
    p.add_argument("--atol", type=float)
    p.add_argument("--dry-run", action="store_true")
    benchmark_launcher.add_arguments(p)
    args = p.parse_args(argv)
    config = load_config(args.dataset)["trainer"]
    model = config["init"]["model"]["architecture"]
    loader = config["fit"]["train_dataloader"]
    if args.batch_size is None:
        args.batch_size = loader["batch_size"]
    if args.fanouts is None:
        args.fanouts = loader["fanouts"]
    if args.hidden is None:
        args.hidden = model["hidden_dim"]
    if args.dropout is None:
        args.dropout = model["dropout"]
    if args.seed is None:
        args.seed = config["init"]["seed"]
    if args.steps is None:
        args.steps = config["init"]["loop"]["max_steps"]
    args.output = args.output.resolve()
    if args.input:
        args.input = args.input.resolve()
    if min(args.batch_size, args.hidden, *args.fanouts) < 1:
        p.error("positive sizes required")
    if not 0 <= args.warmup_steps < args.steps:
        p.error("warmup must be less than steps")
    if any(value is not None and value < 0 for value in (args.rtol, args.atol)):
        p.error("negative comparison tolerance")
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
            name="shared-tree",
            session=args.tmux_session,
        )
    input_dir = args.input or args.output / "input"
    prepare = [
        sys.executable,
        "-u",
        "-m",
        "cerebras.modelzoo.models.gnn.tools.feature_sharing.prepare",
        "--output",
        str(input_dir),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--dataset",
        args.dataset,
        "--batch-size",
        str(args.batch_size),
        "--hidden",
        str(args.hidden),
        "--seed",
        str(args.seed),
        "--dropout",
        str(args.dropout),
        "--fanouts",
        *map(str, args.fanouts),
    ]
    if args.capacity is not None:
        prepare += ["--capacity", str(args.capacity)]
    commands = {}
    for mode in ("dense", "dictionary"):
        commands[mode] = [
            sys.executable,
            "-u",
            "-m",
            "cerebras.modelzoo.models.gnn.tools.feature_sharing.run",
            "--input",
            str(input_dir),
            "--output",
            str(args.output / mode),
            "--mode",
            mode,
            "--backend",
            args.backend,
            "--steps",
            str(args.steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--compile-dir",
            str(((args.compile_dir or args.output / "compile") / mode).resolve()),
        ]
    if args.dry_run:
        for command in ([prepare] if not args.input else []) + list(commands.values()):
            print(shlex.join(command))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="starting", backend=args.backend, planned_steps=args.steps, cases={}
    )
    status_path = args.output / "campaign.json"
    try:
        if not args.input:
            result["active_mode"] = "prepare"
            status_path.write_text(json.dumps(result, indent=2) + "\n")
            with (args.output / "prepare.log").open("w") as log:
                subprocess.run(
                    prepare, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        dataset = FrozenInputs(input_dir, "dictionary")
        deterministic = dataset.metadata["model_config"]["dropout"] == 0
        if args.steps < len(dataset):
            raise ValueError("steps must cover the complete frozen window")
        result["input_sha256"] = file_hash(input_dir / "input.json")
        for mode, cmd in commands.items():
            result["active_mode"] = mode
            status_path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Starting {mode}", flush=True)
            with (args.output / f"{mode}.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            case = json.loads((args.output / mode / "result.json").read_text())
            if (
                case["status"] != "executed"
                or case["input_sha256"] != result["input_sha256"]
            ):
                raise ValueError("case completion/input mismatch")
            result["cases"][mode] = case
        default_tolerance = 0.0 if args.backend == "CPU" else 0.01
        if deterministic:
            result["comparison"] = compare(
                args.output,
                args.steps,
                args.rtol if args.rtol is not None else default_tolerance,
                args.atol if args.atol is not None else default_tolerance,
            )
            result["status"] = "passed"
        else:
            result["comparison"] = {
                "status": "not_checked",
                "reason": "dropout is enabled; compiler RNG parity is not established",
            }
            result["status"] = "executed"
        result.pop("active_mode", None)
    except BaseException as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        status_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
