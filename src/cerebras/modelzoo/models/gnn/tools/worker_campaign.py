"""One-launch CSX worker, input-intervention and resource-observation campaign."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

import yaml

from cerebras.modelzoo.common.utils.run.config_loader import load_params_file
from cerebras.modelzoo.models.gnn.worker_validation import get_available_cpu_cores
from cerebras.modelzoo.tools import benchmark_launcher

if __package__:
    from . import autotune, measure_window
    from .autotune_backends import CSXBackend
    from .job_labels import apply_job_labels
else:
    import autotune
    import measure_window
    from autotune_backends import CSXBackend
    from job_labels import apply_job_labels

GNN, ROOT = autotune.GNN, autotune.ROOT
CONTROLS = {
    "prefetch1": {"prefetch_factor": 1},
    "persistent_off": {"persistent_workers": False},
    "feature_cache": {"cache_fraction": 1.0},
    "static_batch": {"static_batch_cache_size": 1},
}


def configuration(
    base: dict,
    workers: int,
    model: Path,
    steps: int,
    job_seconds: int,
    control: str | None = None,
    diagnostic: bool = False,
    *,
    repeat: int = 1,
    study_dir: Path | None = None,
) -> dict:
    config = CSXBackend.prepare_config(
        base, autotune.candidate(workers, persistent=True), model, steps, job_seconds
    )
    config["trainer"]["init"]["backend"]["cluster_config"]["num_workers_per_csx"] = 1
    loader = config["trainer"]["fit"]["train_dataloader"]
    loader["worker_diagnostics"] = {"enabled": False}
    if control:
        loader.update(CONTROLS[control])
    if diagnostic:
        loader["worker_diagnostics"] = dict(
            enabled=True,
            output_dir=str(model.parent / "worker_diagnostics"),
            max_batches=steps,
            max_snapshots=120,
            snapshot_interval_seconds=5.0,
            resource_monitor=True,
            resource_monitor_pss=True,
        )
    apply_job_labels(
        config,
        mode=(
            "diagnostic" if diagnostic else "intervention" if control else "sensitivity"
        ),
        repeat=repeat,
        trial_dir=model.parent,
        study_dir=study_dir or model.parent.parent,
    )
    return config


def plan(args: argparse.Namespace, base: dict) -> list[dict]:
    stages = []

    def ordinary(workers: int) -> None:
        folder = args.output / f"workers_w{workers:02d}"
        command = [
            sys.executable,
            "-u",
            str(Path(autotune.__file__).resolve()),
            "--backend",
            "csx",
            "--mode",
            "sensitivity",
            "--dataset",
            args.dataset,
            "--cache",
            "none",
            "--wsc-workers",
            "1",
            "--continue-on-failure",
            "--workers",
            *(["4"] if workers == 4 else ["4", str(workers)]),
            "--repeats",
            str(args.repeats),
            "--warmup-steps",
            "40",
            "--measure-steps",
            "400",
            "--job-time-sec",
            str(args.job_time_sec),
            "--trial-timeout-sec",
            str(args.trial_timeout_sec),
            "--budget-sec",
            str(args.budget_sec),
            "--output",
            str(folder),
            "--job-label-study",
            str(args.output),
        ]
        stages.append(
            dict(
                id=folder.name,
                kind="sensitivity",
                workers=workers,
                trials=args.repeats * (1 if workers == 4 else 2),
                directory=str(folder),
                command=command,
            )
        )

    def direct(
        name: str,
        workers: int,
        control: str | None = None,
        diagnostic: bool = False,
        repeat: int = 1,
    ) -> None:
        folder = args.output / name
        config = configuration(
            base,
            workers,
            folder / "model",
            80 if diagnostic else 440,
            args.job_time_sec,
            control,
            diagnostic,
            repeat=repeat,
            study_dir=args.output,
        )
        stages.append(
            dict(
                id=name,
                kind="diagnostic" if diagnostic else "intervention",
                workers=workers,
                control=control,
                repeat=repeat,
                trials=1,
                directory=str(folder),
                config=config,
                config_sha256=autotune.digest(config),
                command=CSXBackend.command(folder / "params.yaml", folder / "model"),
            )
        )

    ordinary(4)
    # Finish mechanism controls before exposing the campaign to larger-worker OOMs.
    for repeat in range(1, args.repeats + 1):
        controls = args.controls if repeat % 2 else list(reversed(args.controls))
        for control in controls:
            direct(f"control_{control}_r{repeat}", 4, control, repeat=repeat)
    direct("diagnostic_w04", 4, diagnostic=True)
    for workers in args.workers:
        if workers == 4:
            continue
        ordinary(workers)
        direct(f"diagnostic_w{workers:02d}", workers, diagnostic=True)
    return stages


def run_sensitivity(command: list[str], log: Path) -> int:
    """Allow the existing tuner to clean up its own separately-sessioned client."""
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    autotune.stop_process(process)
            raise


def summarize(output: Path, state: dict) -> None:
    rows = []
    for stage in state["stages"]:
        directory = Path(stage["directory"])
        if stage["kind"] == "sensitivity":
            source = directory / "study.json"
            trials = json.loads(source.read_text())["trials"] if source.exists() else []
        else:
            trials = [stage]
        for trial in trials:
            measurement = trial.get("measurement") or {}
            rows.append(
                dict(
                    stage=stage["id"],
                    kind=stage["kind"],
                    workers=trial.get("knobs", {}).get("num_workers", stage["workers"]),
                    control=stage.get("control"),
                    repeat=trial.get("repeat"),
                    status=trial["status"],
                    nominal_slots_per_second=measurement.get(
                        "nominal_slots_per_second"
                    ),
                    unstable=(
                        trial["status"] == "unstable"
                        or (measurement.get("half_window_check") or {}).get(
                            "within_tolerance"
                        )
                        is False
                    ),
                    window_seconds=measurement.get("training_window_seconds"),
                    source=str(directory),
                )
            )
    groups = {}
    for row in rows:
        scope = row["stage"] if row["kind"] == "sensitivity" else row["kind"]
        key = f'{scope}/w{row["workers"]}/{row["control"] or "none"}'
        group = groups.setdefault(key, dict(runs=0, measured=0, rates=[]))
        group["runs"] += 1
        if row["nominal_slots_per_second"] is not None:
            group["measured"] += 1
            group["rates"].append(row["nominal_slots_per_second"])
    for group in groups.values():
        values = group["rates"]
        group.update(
            mean=statistics.mean(values) if values else None,
            sample_stddev=statistics.stdev(values) if len(values) > 1 else None,
        )
    autotune.write_json(
        output / "summary.json",
        dict(
            status=state["status"],
            runs=rows,
            groups=groups,
            interpretation="Diagnostics include instrumentation overhead. Static-batch runs reuse one batch and are mechanism probes, not accuracy results. Missing measurements are not zero.",
        ),
    )
    if rows:
        with (output / "runs.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def execute(args: argparse.Namespace, stages: list[dict], fingerprint: str) -> int:
    path = args.output / "campaign.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["fingerprint"] != fingerprint:
            raise ValueError(
                "Sources, settings or configurations changed; use a new output directory"
            )
        if any(s["status"] in {"running", "interrupted"} for s in state["stages"]):
            raise ValueError(
                "Previous campaign was interrupted while a stage was running. Inspect its jobs before starting a new campaign."
            )
    else:
        state = dict(
            fingerprint=fingerprint,
            status="ready",
            used_seconds=0,
            stages=[{**stage, "status": "pending"} for stage in stages],
        )
    autotune.write_json(path, state)
    try:
        for stage in state["stages"]:
            if stage["status"] in {"completed", "completed_with_failures", "failed"}:
                continue
            remaining = args.budget_sec - state["used_seconds"]
            if remaining < args.trial_timeout_sec + 10:
                state["status"] = "budget_exhausted"
                break
            directory = Path(stage["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            stage.update(status="running", started_at=autotune.timestamp())
            state["status"] = "running"
            autotune.write_json(path, state)
            started = time.monotonic()
            print(f'Start {stage["id"]} ({stage["trials"]} run(s))', flush=True)
            try:
                if stage["kind"] == "sensitivity":
                    command = list(stage["command"])
                    nested_path = directory / "study.json"
                    spent = (
                        json.loads(nested_path.read_text()).get("used_sec", 0)
                        if nested_path.exists()
                        else 0
                    )
                    # The inner tuner interprets its budget cumulatively across resumes.
                    command[command.index("--budget-sec") + 1] = str(
                        int(remaining + spent)
                    )
                    code = run_sensitivity(command, directory / "driver.log")
                    stage["returncode"] = code
                    stage["status"] = "completed" if code == 0 else "failed"
                    study_path = directory / "study.json"
                    if code != 0 and study_path.exists():
                        nested = json.loads(study_path.read_text())
                        if nested.get("status") == "budget_exhausted":
                            stage["status"] = "pending"
                            state["status"] = "budget_exhausted"
                        elif (
                            nested.get("status")
                            == "completed_with_missing_measurements"
                        ):
                            stage["status"] = "completed_with_failures"
                        elif any(
                            r["status"] == "interrupted"
                            for r in nested.get("trials", [])
                        ):
                            stage["status"] = "interrupted"
                else:
                    (directory / "params.yaml").write_text(
                        yaml.safe_dump(stage["config"], sort_keys=False)
                    )
                    result = autotune.execute(
                        stage["command"],
                        directory / "train.log",
                        args.trial_timeout_sec,
                    )
                    stage.update(result)
                    stage["client_status"] = result["status"]
                    if result["status"] == "completed":
                        measurement = measure_window.summarize(
                            directory / "train.log",
                            40,
                            80 if stage["kind"] == "diagnostic" else 440,
                        )
                        stage["measurement"] = measurement
                        points = measure_window.read_points(directory / "train.log")
                        stage["step_timestamps_utc"] = {
                            str(p.step): p.timestamp.replace(
                                tzinfo=timezone.utc
                            ).isoformat()
                            for p in points
                        }
                    else:
                        stage["status"] = (
                            "interrupted"
                            if result["status"] == "interrupted"
                            else "failed"
                        )
                    autotune.write_json(directory / "result.json", stage)
            except (KeyboardInterrupt, Exception) as exc:
                stage.update(
                    status=(
                        "interrupted"
                        if isinstance(exc, KeyboardInterrupt)
                        else "failed"
                    ),
                    error=type(exc).__name__ + ": " + str(exc),
                )
            finally:
                elapsed = time.monotonic() - started
                state["used_seconds"] += elapsed
                stage.update(
                    client_wall_seconds=elapsed, finished_at=autotune.timestamp()
                )
                autotune.write_json(path, state)
                summarize(args.output, state)
            if state["status"] == "budget_exhausted":
                break
            if stage["status"] == "interrupted":
                state["status"] = "interrupted"
                break
            if stage["status"] != "completed":
                stage["continued_after_failure_at"] = autotune.timestamp()
                print(
                    f'Recorded failure in {stage["id"]}; continuing with the next planned stage.',
                    flush=True,
                )
        else:
            state["status"] = (
                "completed"
                if all(s["status"] == "completed" for s in state["stages"])
                else "completed_with_failures"
            )
    finally:
        autotune.write_json(path, state)
        summarize(args.output, state)
    return 0 if state["status"] == "completed" else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    benchmark_launcher.add_arguments(parser)
    parser.add_argument("--dataset", choices=("arxiv", "products"), default="arxiv")
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument(
        "--controls", choices=list(CONTROLS), nargs="*", default=list(CONTROLS)
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-sec", type=int, default=86400)
    parser.add_argument("--job-time-sec", type=int, default=7200)
    parser.add_argument("--trial-timeout-sec", type=int, default=9000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--archive", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    if args.repeats < 3 or 4 not in args.workers or any(w < 4 for w in args.workers):
        parser.error(
            "Require >=3 repeats and worker counts >=4 including the reference 4"
        )
    if (
        args.job_time_sec <= 0
        or args.trial_timeout_sec < args.job_time_sec
        or args.budget_sec < args.trial_timeout_sec + 10
    ):
        parser.error(
            "Require budget >= trial timeout + 10, trial timeout >= positive job time"
        )
    args.workers = sorted(set(args.workers))
    args.controls = list(dict.fromkeys(args.controls))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    args.output = (
        args.output or ROOT / f"model_dirs/hpcasia_r04/campaign_{stamp}"
    ).resolve()
    return args


def main(argv: list[str] | None = None) -> int:
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
            name="gnn-worker-campaign",
            session=args.tmux_session,
        )

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    base = load_params_file(GNN / "configs/autotune" / f"{args.dataset}_w40.yaml")
    stages = plan(args, base)
    with autotune.study_lock(args.output):
        if args.dry_run and (args.output / "campaign.json").exists():
            raise ValueError("Use a separate output for a dry-run")
        cores = get_available_cpu_cores()
        if not args.dry_run and cores is not None and max(args.workers) > cores:
            raise ValueError(
                f"Requested workers exceed launch affinity ({cores}); no jobs submitted"
            )
        settings = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key
            not in {
                "dry_run",
                "archive",
                "budget_sec",
                "detach",
                "tmux_session",
            }
        }
        provenance = {} if args.dry_run else autotune.environment("csx")
        identity = {k: v for k, v in provenance.items() if k != "git_status"}
        fingerprint = autotune.digest(
            dict(settings=settings, base=base, environment=identity)
        )
        description = dict(
            settings=settings,
            failure_policy="skip_failed_or_timed_out_runs; stop_on_interrupt_or_budget",
            planned_trials=sum(s["trials"] for s in stages),
            budget_sec=args.budget_sec,
            stages=stages,
            environment=provenance,
            launch_cpu_cores=cores,
        )
        if not (args.output / "campaign.json").exists():
            autotune.write_json(args.output / "plan.json", description)
            (args.output / "base.yaml").write_text(
                yaml.safe_dump(base, sort_keys=False)
            )
        print(
            f'Output: {args.output}; {description["planned_trials"]} planned training runs',
            flush=True,
        )
        if args.dry_run:
            for stage in stages:
                if "config" in stage:
                    folder = Path(stage["directory"])
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "params.yaml").write_text(
                        yaml.safe_dump(stage["config"], sort_keys=False)
                    )
                else:
                    subprocess.run(
                        stage["command"] + ["--dry-run"], cwd=ROOT, check=True
                    )
            return 0
        code = execute(args, stages, fingerprint)
    if args.archive:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive = shutil.make_archive(
            str(args.output) + "_" + stamp, "zip", args.output.parent, args.output.name
        )
        print(f"Archive: {archive}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
