"""Run fixed-input CSX learning and throughput measurements in seed-major order."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import signal
import statistics
import sys
import time

from cerebras.modelzoo.models.gnn.tools import autotune, learning_campaign
from cerebras.modelzoo.models.gnn.tools.autotune_backends import CSXBackend
from cerebras.modelzoo.models.gnn.tools.job_labels import apply_job_labels
from cerebras.modelzoo.models.gnn.worker_validation import get_available_cpu_cores
from cerebras.modelzoo.tools import benchmark_launcher

KNOBS = dict(num_workers=4, prefetch_factor=2, persistent_workers=True)


def plan(args, base: dict) -> list[dict]:
    runs = []
    for seed in args.seeds:
        seeded = deepcopy(base)
        seeded["trainer"]["init"]["seed"] = seed
        for name in ("train_dataloader", "val_dataloader"):
            seeded["trainer"]["fit"][name]["sampler_seed"] = seed
        for kind, repeat in [
            ("learning", 1),
            ("throughput", 1),
            ("throughput", 2),
            ("throughput", 3),
            ("cache", 1),
        ]:
            name = f"seed_{seed}/{kind}_r{repeat}"
            folder = args.output / name
            if kind == "learning":
                config = learning_campaign.learning_config(seeded, args, KNOBS, folder)
            else:
                config = CSXBackend.prepare_config(
                    seeded,
                    KNOBS,
                    folder / "model",
                    args.warmup_steps + args.measure_steps,
                    args.job_time_sec,
                )
                config["trainer"]["init"]["model"]["task"][
                    "compute_eval_metrics"
                ] = False
                # prepare_config intentionally disables cache; apply this control last.
                if kind == "cache":
                    config["trainer"]["fit"]["train_dataloader"]["cache_fraction"] = 1.0
            apply_job_labels(
                config,
                mode=kind,
                repeat=repeat,
                trial_dir=folder,
                study_dir=args.output,
            )
            runs.append(
                dict(
                    id=name,
                    kind=kind,
                    seed=seed,
                    repeat=repeat,
                    directory=str(folder),
                    config=config,
                    command=CSXBackend.command(
                        folder / "params.yaml", folder / "model"
                    ),
                )
            )
    return runs


def summarize(output: Path, state: dict) -> None:
    seeds = []
    for seed in state["settings"]["seeds"]:
        rows = [r for r in state["runs"] if r["seed"] == seed]
        learning = [
            r for r in rows if r["kind"] == "learning" and r["status"] == "completed"
        ]
        main = [
            r for r in rows if r["kind"] == "throughput" and r["status"] == "completed"
        ]
        cache = [r for r in rows if r["kind"] == "cache" and r["status"] == "completed"]
        rates = [r["measurement"]["seed_nodes_per_second"] for r in main]
        row = dict(
            seed=seed,
            throughput_repeats_completed=len(main),
            throughput_seed_nodes_per_second=rates,
            final_validation_accuracy=(
                learning[0]["curves"]["final_validation_accuracy"] if learning else None
            ),
        )
        if rates:
            row["throughput_mean"] = statistics.mean(rates)
            row["throughput_sample_stddev"] = (
                statistics.stdev(rates) if len(rates) > 1 else None
            )
        if cache:
            row["cache_seed_nodes_per_second"] = cache[0]["measurement"][
                "seed_nodes_per_second"
            ]
            if len(main) == 3:
                row["cache_to_uncached_ratio"] = (
                    row["cache_seed_nodes_per_second"] / row["throughput_mean"]
                )
        seeds.append(row)
    autotune.write_json(
        output / "summary.json",
        dict(
            status=state["status"],
            seeds=seeds,
            interpretation="Within-seed throughput repeats and between-seed learning results are separate. Retain all completed measurements, including failed half-window stability checks. Cache effects are observations, not a pass criterion.",
        ),
    )


def settings(args) -> dict:
    return {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"budget_sec", "detach", "tmux_session", "dry_run"}
    }


def execute(args, runs: list[dict], provenance: dict) -> int:
    fingerprint = autotune.digest(
        dict(
            settings=settings(args),
            runs=runs,
            environment={k: v for k, v in provenance.items() if k != "git_status"},
        )
    )
    path = args.output / "campaign.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["fingerprint"] != fingerprint:
            raise ValueError(
                "Settings, sources or environment changed; use a new output"
            )
        if any(r["status"] not in {"pending", "completed"} for r in state["runs"]):
            raise ValueError(
                "Previous client did not finish cleanly; inspect its jobs and use a new output"
            )
        if args.budget_sec < state["budget_sec"]:
            raise ValueError("The cumulative budget cannot decrease")
    else:
        state = dict(
            fingerprint=fingerprint,
            environment=provenance,
            settings=settings(args),
            status="ready",
            used_seconds=0.0,
            runs=[{**run, "status": "pending"} for run in runs],
        )
    state["budget_sec"] = args.budget_sec

    def save():
        autotune.write_json(path, state)
        summarize(args.output, state)

    save()
    for row in state["runs"]:
        if row["status"] == "completed":
            continue
        if args.budget_sec - state["used_seconds"] < args.trial_timeout_sec + 10:
            state["status"] = "budget_exhausted"
            save()
            return 2
        folder = Path(row["directory"])
        learning_campaign.write_config(folder / "params.yaml", row["config"])
        row.update(
            status="running",
            started_at=autotune.timestamp(),
            config_sha256=autotune.digest(row["config"]),
        )
        state["status"] = "running"
        save()
        print(f"Start {row['id']}: {folder / 'train.log'}", flush=True)
        started = time.monotonic()
        try:
            result = autotune.execute(
                row["command"], folder / "train.log", args.trial_timeout_sec
            )
            row.update(result, client_status=result["status"])
            if row["status"] == "completed":
                if row["kind"] == "learning":
                    row["curves"] = learning_campaign.collect_learning(
                        folder / "train.log", args
                    )
                    autotune.write_json(folder / "learning_curves.json", row["curves"])
                else:
                    row["measurement"] = CSXBackend.measure(
                        folder / "train.log",
                        args.warmup_steps,
                        args.warmup_steps + args.measure_steps,
                        args.stability_tolerance_percent,
                    )
        except KeyboardInterrupt:
            row.update(status="interrupted")
        except Exception as exc:
            row.update(
                status="invalid_measurement", reason=f"{type(exc).__name__}: {exc}"
            )
        finally:
            row.update(
                client_wall_seconds=time.monotonic() - started,
                finished_at=autotune.timestamp(),
            )
            log = folder / "train.log"
            row["job_ids"] = sorted(
                set(
                    re.findall(
                        r"\bwsjob-(?!dashboard\b)[A-Za-z0-9-]+",
                        log.read_text(errors="replace") if log.exists() else "",
                    )
                )
            )
            state["used_seconds"] += row["client_wall_seconds"]
            if row["status"] != "completed":
                state["status"] = row["status"]
            autotune.write_json(folder / "result.json", row)
            save()
        if row["status"] != "completed":
            return 2
    state["status"] = "completed"
    save()
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    benchmark_launcher.add_arguments(parser)
    parser.add_argument("--dataset", choices=("arxiv", "products"), default="arxiv")
    parser.add_argument("--seeds", type=int, nargs=3, default=[42, 43, 44])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--learning-steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--measure-steps", type=int, default=800)
    parser.add_argument("--stability-tolerance-percent", type=float, default=2.0)
    parser.add_argument("--budget-sec", type=int, default=172800)
    parser.add_argument("--job-time-sec", type=int, default=7200)
    parser.add_argument("--trial-timeout-sec", type=int, default=9000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.seeds[0] != 42
        or len(set(args.seeds)) != 3
        or any(s < 0 or s >= 2**32 for s in args.seeds)
    ):
        parser.error("Require three distinct seeds in [0, 2**32), starting with 42")
    if args.learning_steps is None:
        args.learning_steps = 500 if args.dataset == "arxiv" else 1000
    if args.eval_every is None:
        # IA3 accuracy profiles: arxiv every 20 steps, products every 40.
        args.eval_every = 20 if args.dataset == "arxiv" else 40
    if (
        args.learning_steps < 60
        or args.eval_every <= 0
        or args.eval_every % 10
        or args.learning_steps % args.eval_every
        or args.learning_steps // args.eval_every < 2
    ):
        parser.error(
            "Require >=60 learning steps and >=2 full evaluations at multiples of 10 steps"
        )
    if (
        args.warmup_steps < 10
        or args.warmup_steps % 10
        or args.measure_steps < 20
        or args.measure_steps % 20
    ):
        parser.error(
            "Warmup must be a positive multiple of 10; measure steps a positive multiple of 20"
        )
    if not 0 < args.stability_tolerance_percent < 100:
        parser.error("Stability tolerance must be between 0 and 100 percent")
    if (
        args.job_time_sec <= 0
        or args.trial_timeout_sec <= args.job_time_sec
        or args.budget_sec < args.trial_timeout_sec + 10
    ):
        parser.error(
            "Require 0 < job time < client timeout and budget >= client timeout + 10"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    args.output = (
        args.output
        or autotune.ROOT / f"model_dirs/hpcasia_final/{args.dataset}_{stamp}"
    ).resolve()
    return args


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.detach and not args.dry_run:
        return benchmark_launcher.launch(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                *argv,
                "--foreground",
                "--output",
                str(args.output),
            ],
            args.output,
            name="gnn-hpcasia",
            session=args.tmux_session,
        )

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    runs = plan(args, learning_campaign.shared_base(args.dataset))
    with autotune.study_lock(args.output):
        if args.dry_run:
            if (args.output / "campaign.json").exists():
                raise ValueError("Use a separate output for a dry-run")
            for run in runs:
                learning_campaign.write_config(
                    Path(run["directory"]) / "params.yaml", run["config"]
                )
            autotune.write_json(
                args.output / "plan.json", dict(settings=settings(args), runs=runs)
            )
            print(
                f"Preview: {args.output}; {len(runs)} runs; no jobs submitted",
                flush=True,
            )
            return 0
        if (args.output / "plan.json").exists() and not (
            args.output / "campaign.json"
        ).exists():
            raise ValueError(
                "Dry-run output exists; use a fresh output for measurements"
            )
        cores = get_available_cpu_cores()
        if cores is not None and cores < KNOBS["num_workers"]:
            raise ValueError(f"Four workers exceed launch affinity ({cores})")
        return execute(args, runs, autotune.environment("csx"))


if __name__ == "__main__":
    raise SystemExit(main())
