"""Sequential CSX/PyG input tuning with budgets, resume and repeated measurements."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

import yaml

from cerebras.modelzoo.common.utils.run.config_loader import load_params_file
from cerebras.modelzoo.models.gnn.worker_validation import (
    get_available_cpu_cores,
)

# Also support direct execution from the documented GNN directory.
if __package__:
    from .autotune_backends import get_backend
    from .job_labels import apply_job_labels
else:
    from autotune_backends import get_backend
    from job_labels import apply_job_labels

GNN = Path(__file__).resolve().parents[1]
ROOT = GNN.parents[4]
UNCERTAIN = {"running", "interrupted", "timeout", "failed"}


def prepare_config(backend, args, base, knobs, model_dir, steps, *, repeat=1):
    config = backend.prepare_config(
        base, knobs, model_dir, steps, args.job_time_sec, args.warmup_steps
    )
    if args.cache == "full":
        config["trainer"]["fit"]["train_dataloader"]["cache_fraction"] = 1.0
    if args.wsc_workers is not None:
        config["trainer"]["init"]["backend"]["cluster_config"][
            "num_workers_per_csx"
        ] = args.wsc_workers
    if backend.name == "csx":
        apply_job_labels(
            config,
            mode=args.mode,
            repeat=repeat,
            trial_dir=model_dir.parent,
            study_dir=Path(args.job_label_study or args.output),
        )
    return config


def sensitivity_report(state, output):
    """Derive tables from existing trial records, retaining unstable/failed runs."""
    settings = state["settings"]
    rows = []
    summaries = []
    for workers in settings["workers"]:
        group = []
        for repeat in range(1, settings["repeats"] + 1):
            trial = next(
                (
                    r
                    for r in state["trials"]
                    if r["phase"] == "sensitivity"
                    and r["knobs"]["num_workers"] == workers
                    and r["repeat"] == repeat
                ),
                None,
            )
            measurement = (trial or {}).get("measurement", {})
            # An unstable but complete finite measurement remains in the statistics.
            usable = trial is not None and trial["status"] in {
                "completed",
                "unstable",
            }
            rate = measurement.get("throughput") if usable else None
            row = dict(
                num_workers=workers,
                repeat=repeat,
                status=trial["status"] if trial else "not_run",
                throughput=rate,
                metric=measurement.get("metric"),
                training_window_seconds=measurement.get(
                    "training_window_seconds"
                ),
                trial_id=trial["trial_id"] if trial else None,
                job_ids=";".join(trial.get("job_ids", [])) if trial else "",
                reason=trial.get("failure_reason") if trial else None,
            )
            rows.append(row)
            group.append(row)
        rates = [r["throughput"] for r in group if r["throughput"] is not None]
        metrics = {r["metric"] for r in group if r["throughput"] is not None}
        if len(metrics) > 1:
            raise ValueError("Cannot combine different throughput metrics")
        summaries.append(
            dict(
                num_workers=workers,
                planned_runs=settings["repeats"],
                measured_runs=len(rates),
                unstable_runs=sum(r["status"] == "unstable" for r in group),
                failed_or_invalid_runs=sum(
                    r["status"]
                    not in {"completed", "unstable", "not_run", "running"}
                    for r in group
                ),
                not_run=sum(r["status"] == "not_run" for r in group),
                running_runs=sum(r["status"] == "running" for r in group),
                metric=next(iter(metrics), None),
                mean=statistics.mean(rates) if rates else None,
                sample_stddev=(
                    statistics.stdev(rates) if len(rates) > 1 else None
                ),
            )
        )
    write_json(
        output / "sensitivity.json",
        {
            "settings": settings,
            "status": state["status"],
            "definition": "Equal-weight run mean and sample standard deviation (ddof=1); includes unstable finite runs. Missing runs are not zero.",
            "runs": rows,
            "summary": summaries,
        },
    )
    for name, values in (("runs", rows), ("summary", summaries)):
        with (output / f"sensitivity_{name}.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()


@contextmanager
def study_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another tuner holds this study's lock") from exc
        yield


def candidate(workers, prefetch=2, persistent=False):
    return {
        "num_workers": workers,
        "prefetch_factor": prefetch if workers else None,
        "persistent_workers": persistent if workers else False,
    }


def candidate_id(knobs):
    return f"w{knobs['num_workers']:02d}_p{knobs['prefetch_factor'] or 0}_s{int(knobs['persistent_workers'])}"


def environment(backend="csx"):
    # Inspect the environment used by the child command without syncing it.
    code = """import importlib.metadata as m, json, sys, os
import cerebras.modelzoo as mz
info = {"python": sys.version, "executable": sys.executable,
    "prefix": sys.prefix, "modelzoo_path": mz.__file__,
    "runtime_environment": {k: os.environ.get(k) for k in
        ("CUDA_VISIBLE_DEVICES", "NO_COMPILE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    "packages": sorted((d.metadata["Name"], d.version) for d in m.distributions())}
"""
    if backend == "csx":
        code += """import cerebras.pytorch, shutil
info.update(cszoo=shutil.which("cszoo"), sdk=m.version("cerebras-pytorch"))
"""
    else:
        code += """import torch, torch_geometric
from cerebras.modelzoo.models.gnn.reference.pyg.data import check_pyg_lib
from cerebras.modelzoo.models.gnn.reference.pyg.runner import main
check_pyg_lib()
if not torch.cuda.is_available():
    raise RuntimeError("PyG tuning requires an available CUDA GPU")
if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
    raise RuntimeError("PyG tuning currently supports one GPU per trial; launch without torchrun")
p = torch.cuda.get_device_properties(0)
info.update(torch=torch.__version__, pyg=torch_geometric.__version__, cuda=torch.version.cuda,
            gpu=p.name, gpu_memory_bytes=p.total_memory,
            gpu_capability=[p.major, p.minor])
"""
    code += "print(json.dumps(info))"
    result = subprocess.run(
        ["uv", "run", "--no-sync", "--", "python", "-c", code],
        cwd=GNN,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(
            f"{backend} environment check failed:\n{result.stderr[-3000:]}"
        )
    info = json.loads(result.stdout.strip().splitlines()[-1])
    if Path(info["modelzoo_path"]).resolve().parent != GNN.parents[1]:
        raise ValueError("Prepare an editable installation of this checkout")
    if backend == "csx" and (
        not info["python"].startswith("3.11.")
        or info["sdk"] != "2.10.0"
        or not info["cszoo"]
        or Path(info["cszoo"]).parent != Path(info["prefix"]) / "bin"
    ):
        raise ValueError(
            "Prepare Python 3.11, Cerebras 2.10.0 and editable cszoo before tuning"
        )
    info["uv"] = subprocess.check_output(
        ["uv", "--version"], text=True
    ).strip()
    info["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    info["git_status"] = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    )
    # Include uncommitted and untracked implementation changes without storing their contents.
    sources = list((ROOT / "src").rglob("*.py"))
    sources.extend(
        ROOT / "benchmark_scripts/cerebras" / name
        for name in (
            "worker_launcher.py",
            "run_worker_campaign.sh",
            "run_worker_sensitivity.sh",
        )
    )
    info["source_sha256"] = digest(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(sources)
        }
    )
    return info


def stop_process(process):
    # Stop the whole private session, including loader workers left by a failed client.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def execute(cmd, log, timeout):
    """Run one client; a stopped client does not prove its remote job has stopped."""
    started = time.monotonic()
    process = None
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(
                cmd,
                cwd=GNN,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            code = process.wait(timeout=timeout)
        return {
            "status": "completed" if code == 0 else "failed",
            "returncode": code,
            "failure_reason": (
                None if code == 0 else f"Client exited with {code}"
            ),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "failure_reason": "Client wall-time limit reached",
        }
    except KeyboardInterrupt:
        return {"status": "interrupted", "failure_reason": "Tuner interrupted"}
    except OSError as exc:
        return {"status": "failed", "failure_reason": str(exc)}
    finally:
        if process is not None:
            stop_process(process)
        print(
            f"  client elapsed: {time.monotonic() - started:.1f}s", flush=True
        )


def rank(records, phase, repeats=1):
    groups = {}
    for row in records:
        if row["phase"] == phase:
            groups.setdefault(row["candidate_id"], []).append(row)
    ranked = []
    for key, rows in groups.items():
        # A failed or unstable repeat disqualifies the candidate; never cherry-pick it away.
        if len(rows) != repeats or any(
            r["status"] != "completed" for r in rows
        ):
            continue
        rates = [r["measurement"]["throughput"] for r in rows]
        ranked.append(
            {
                "candidate_id": key,
                "knobs": rows[0]["knobs"],
                "median_throughput": statistics.median(rates),
                "min_throughput": min(rates),
                "max_throughput": max(rates),
                "repeats": len(rates),
                "metric": rows[0]["measurement"]["metric"],
            }
        )
    return sorted(
        ranked, key=lambda r: (-r["median_throughput"], r["candidate_id"])
    )


class Study:
    def __init__(self, args, base, output, provenance):
        self.args, self.base, self.output = args, base, output
        self.backend = get_backend(args.backend)
        settings = {
            k: v
            for k, v in vars(args).items()
            if k
            not in {
                "output",
                "dry_run",
                "acknowledge_stopped_jobs",
                "budget_sec",
            }
        }
        # git_status may change when generated results are stored inside the checkout.
        identity = {k: v for k, v in provenance.items() if k != "git_status"}
        self.fingerprint = digest(
            {"settings": settings, "base": base, "environment": identity}
        )
        path = output / "study.json"
        if path.exists():
            self.state = json.loads(path.read_text())
            if self.state["fingerprint"] != self.fingerprint:
                raise ValueError(
                    "Settings, source, configuration or environment changed; use a new output directory"
                )
            if args.budget_sec < self.state["budget_sec"]:
                raise ValueError(
                    "A resumed study's total budget cannot decrease"
                )
            if args.acknowledge_stopped_jobs:
                for row in self.state["trials"]:
                    if row["status"] in UNCERTAIN:
                        if row["status"] == "running":
                            # The prior client disappeared before recording its elapsed time.
                            self.state["used_sec"] += self.state["settings"][
                                "trial_timeout_sec"
                            ]
                            row.update(
                                status="interrupted",
                                failure_reason="Previous tuner disappeared",
                            )
                        row["job_stop_acknowledged_at"] = timestamp()
        else:
            self.state = {
                "fingerprint": self.fingerprint,
                "created_at": timestamp(),
                "settings": settings,
                "environment": provenance,
                "used_sec": 0.0,
                "trials": [],
                "status": "ready",
            }
            (output / "base.yaml").write_text(
                yaml.safe_dump(base, sort_keys=False)
            )
        self.state["budget_sec"] = args.budget_sec
        self.save()

    def save(self):
        write_json(self.output / "study.json", self.state)
        if self.args.mode == "sensitivity":
            sensitivity_report(self.state, self.output)

    def trial(self, knobs, phase, repeat, measured_steps):
        key = candidate_id(knobs)
        trial_id = f"{phase}_{key}_r{repeat}"
        prior = next(
            (r for r in self.state["trials"] if r["trial_id"] == trial_id),
            None,
        )
        if prior:
            return prior
        if (
            self.state["budget_sec"] - self.state["used_sec"]
            < self.args.trial_timeout_sec + 10
        ):
            self.state["status"] = "budget_exhausted"
            self.save()
            raise StopIteration
        folder = self.output / trial_id
        folder.mkdir(exist_ok=False)
        config = prepare_config(
            self.backend,
            self.args,
            self.base,
            knobs,
            folder / "model",
            self.args.warmup_steps + measured_steps,
            repeat=repeat,
        )
        config_path = folder / "params.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        row = {
            "trial_id": trial_id,
            "candidate_id": key,
            "phase": phase,
            "repeat": repeat,
            "knobs": knobs,
            "status": "running",
            "started_at": timestamp(),
            "command": self.backend.command(config_path, folder / "model"),
            "config_sha256": digest(config),
        }
        self.state["trials"].append(row)
        self.save()
        print(
            f"{trial_id}: {self.args.warmup_steps} warm-up + {measured_steps} measured steps; {folder / 'train.log'}",
            flush=True,
        )
        start = time.monotonic()
        row.update(
            execute(
                row["command"],
                folder / "train.log",
                self.args.trial_timeout_sec,
            )
        )
        row["client_wall_seconds"] = time.monotonic() - start
        row["finished_at"] = timestamp()
        self.state["used_sec"] += row["client_wall_seconds"]
        log = folder / "train.log"
        contents = log.read_text(errors="replace") if log.exists() else ""
        row["job_ids"] = sorted(
            set(re.findall(r"\bwsjob-[A-Za-z0-9-]+", contents))
        )
        row["performance_files"] = [
            str(p.relative_to(folder))
            for p in sorted(folder.rglob("performance.json"))
        ]
        if row["status"] == "completed":
            try:
                row["measurement"] = self.backend.measure(
                    log,
                    self.args.warmup_steps,
                    self.args.warmup_steps + measured_steps,
                    self.args.stability_tolerance_percent,
                )
                check = row["measurement"]["half_window_check"]
                if not check or not check["within_tolerance"]:
                    row.update(
                        status="unstable",
                        failure_reason="Half-window stability check failed or midpoint missing",
                    )
            except (ValueError, OSError) as exc:
                row.update(
                    status="invalid_measurement", failure_reason=str(exc)
                )
        if row["status"] in UNCERTAIN and not self.backend.remote:
            row["job_stop_acknowledged_at"] = timestamp()
        continuing = self.args.continue_on_failure and row["status"] in {
            "failed",
            "timeout",
        }
        if continuing:
            row["continued_after_failure_at"] = timestamp()
            row["remote_stop_unconfirmed"] = self.backend.remote
        write_json(folder / "result.json", row)
        self.save()
        if (
            row["status"] in UNCERTAIN
            and self.backend.remote
            and not continuing
        ):
            self.state["status"] = "job_stop_unconfirmed"
            self.save()
            raise StopIteration
        if row["status"] == "interrupted":
            self.state["status"] = "interrupted"
            self.save()
            raise StopIteration
        return row

    def run(self):
        if any(
            r["status"] in UNCERTAIN
            and not r.get("job_stop_acknowledged_at")
            and not (
                self.args.continue_on_failure
                and r["status"] in {"failed", "timeout"}
            )
            for r in self.state["trials"]
        ):
            raise ValueError(
                "Previous client failed/stopped. Confirm its job/process has ended, then use --acknowledge-stopped-jobs"
            )
        if self.args.mode == "sensitivity":
            try:
                # Every candidate, once per round; no screening or winner selection.
                for repeat in range(1, self.args.repeats + 1):
                    order = (
                        self.args.workers
                        if repeat % 2
                        else list(reversed(self.args.workers))
                    )
                    for workers in order:
                        self.trial(
                            candidate(
                                workers,
                                persistent=self.args.persistent_workers,
                            ),
                            "sensitivity",
                            repeat,
                            self.args.measure_steps,
                        )
                self.state["status"] = (
                    "completed"
                    if all(
                        r["status"] in {"completed", "unstable"}
                        for r in self.state["trials"]
                    )
                    else "completed_with_missing_measurements"
                )
                self.save()
            except StopIteration:
                pass
            print(
                json.dumps(
                    {
                        k: self.state[k]
                        for k in ("status", "used_sec", "budget_sec")
                    },
                    indent=2,
                )
            )
            return 0 if self.state["status"] == "completed" else 2
        try:
            for workers in self.args.workers:
                self.trial(
                    candidate(
                        workers, persistent=self.args.persistent_workers
                    ),
                    "workers",
                    1,
                    self.args.measure_steps,
                )
            coarse = rank(self.state["trials"], "workers")
            pool = coarse[: self.args.top_k]
            if self.args.prefetch_factors:
                for row in pool:
                    workers = row["knobs"]["num_workers"]
                    if not workers:
                        continue
                    for prefetch in self.args.prefetch_factors:
                        for persistent in (False, True):
                            self.trial(
                                candidate(workers, prefetch, persistent),
                                "loader",
                                1,
                                self.args.measure_steps,
                            )
                # Prefer the later measurement when a baseline setting was measured again.
                combined = {r["candidate_id"]: r for r in coarse}
                loader_rows = [
                    r for r in self.state["trials"] if r["phase"] == "loader"
                ]
                for row in loader_rows:
                    combined.pop(row["candidate_id"], None)
                combined.update(
                    {
                        r["candidate_id"]: r
                        for r in rank(self.state["trials"], "loader")
                    }
                )
                pool = sorted(
                    combined.values(),
                    key=lambda r: -r["median_throughput"],
                )[: self.args.top_k]
            self.state["finalists"] = pool
            self.save()
            # Alternate order between rounds so candidates are not measured in three-job blocks.
            for repeat in range(1, self.args.repeats + 1):
                for row in pool if repeat % 2 else list(reversed(pool)):
                    self.trial(
                        row["knobs"],
                        "confirm",
                        repeat,
                        self.args.confirm_steps,
                    )
            self.state["ranking"] = rank(
                self.state["trials"], "confirm", self.args.repeats
            )
            if self.state["ranking"]:
                best = self.state["ranking"][0]
                # Emit a reproducible benchmark config, not an unbounded production training config.
                config = prepare_config(
                    self.backend,
                    self.args,
                    self.base,
                    best["knobs"],
                    self.output / "best_model",
                    self.args.warmup_steps + self.args.confirm_steps,
                )
                (self.output / "best.yaml").write_text(
                    yaml.safe_dump(config, sort_keys=False)
                )
                self.state["status"] = "completed"
            else:
                self.state["status"] = "no_valid_candidate"
            self.save()
        except StopIteration:
            pass
        print(
            json.dumps(
                {
                    k: self.state[k]
                    for k in ("status", "used_sec", "budget_sec")
                },
                indent=2,
            )
        )
        return 0 if self.state["status"] == "completed" else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("autotune", "sensitivity"), default="autotune"
    )
    parser.add_argument("--cache", choices=("none", "full"), default="none")
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--wsc-workers",
        type=int,
        help="Fixed WSC Worker replicas per CSX; distinct from DataLoader workers",
    )
    parser.add_argument("--backend", choices=("csx", "pyg"), default="csx")
    parser.add_argument(
        "--dataset", choices=("arxiv", "products"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--job-label-study",
        help="Shared CSX study directory for job labels; defaults to --output",
    )
    parser.add_argument("--workers", type=int, nargs="+", default=None)
    parser.add_argument("--prefetch-factors", type=int, nargs="+", default=[])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--measure-steps", type=int)
    parser.add_argument("--confirm-steps", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--stability-tolerance-percent", type=float, default=2.0
    )
    parser.add_argument("--job-time-sec", type=int, default=7200)
    parser.add_argument(
        "--trial-timeout-sec",
        type=int,
        default=9000,
        help="Client wall-time cap including queue and compile; separate from job_time_sec",
    )
    parser.add_argument(
        "--budget-sec",
        type=int,
        required=True,
        help="Total client wall-time budget across resumes; can be increased on resume",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the resolved plan without launching a training client",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Sensitivity only: skip failed/timed-out trials and continue; remote termination is not asserted",
    )
    parser.add_argument(
        "--acknowledge-stopped-jobs",
        action="store_true",
        help="Resume after independently verifying all previous unconfirmed jobs/processes have ended",
    )
    args = parser.parse_args(argv)
    if args.continue_on_failure and args.mode != "sensitivity":
        parser.error("--continue-on-failure requires --mode sensitivity")
    backend = get_backend(args.backend)
    if args.workers is None:
        args.workers = (
            [40, 4, 8, 10, 12, 16, 20]
            if args.mode == "sensitivity"
            else [40, 0, 4, 8, 16, 32]
        )
    if args.persistent_workers is None:
        args.persistent_workers = (
            True if args.mode == "sensitivity" else backend.persistent_workers
        )
    if args.wsc_workers is not None and (
        args.backend != "csx" or args.wsc_workers < 1
    ):
        parser.error("wsc-workers requires CSX and a positive replica count")
    if args.mode == "sensitivity":
        if args.backend == "csx" and args.wsc_workers is None:
            parser.error(
                "Sensitivity requires an explicit --wsc-workers count fixed across all trials"
            )
        if args.prefetch_factors:
            parser.error(
                "Sensitivity fixes prefetch_factor=2; omit --prefetch-factors"
            )
        if args.confirm_steps is not None:
            parser.error(
                "Sensitivity uses --measure-steps for every run; omit --confirm-steps"
            )
    if args.measure_steps is None:
        args.measure_steps = (
            400 if args.mode == "sensitivity" else backend.measure_steps
        )
    if args.confirm_steps is None:
        args.confirm_steps = (
            args.measure_steps
            if args.mode == "sensitivity"
            else backend.confirm_steps
        )
    positive = (
        "top_k",
        "warmup_steps",
        "measure_steps",
        "confirm_steps",
        "job_time_sec",
        "trial_timeout_sec",
        "budget_sec",
    )
    if any(getattr(args, k) <= 0 for k in positive) or args.repeats < 3:
        parser.error(
            "Budgets and step counts must be positive; confirmation requires at least 3 repeats"
        )
    if (
        args.warmup_steps % 10
        or args.measure_steps % 20
        or args.confirm_steps % 20
    ):
        parser.error(
            "warmup must be a multiple of 10; measured windows must be multiples of 20 (exact endpoints/midpoints)"
        )
    if args.confirm_steps < args.measure_steps:
        parser.error("confirm-steps must be at least measure-steps")
    if (
        args.backend == "csx" and args.trial_timeout_sec < args.job_time_sec
    ) or args.budget_sec < args.trial_timeout_sec + 10:
        parser.error(
            "Require budget-sec >= trial-timeout-sec + 10 (cleanup reserve); CSX also requires trial-timeout-sec >= job-time-sec"
        )
    if any(w < 0 for w in args.workers) or any(
        p < 1 for p in args.prefetch_factors
    ):
        parser.error(
            "Workers must be non-negative and prefetch factors positive"
        )
    if (
        not math.isfinite(args.stability_tolerance_percent)
        or args.stability_tolerance_percent <= 0
    ):
        parser.error("Stability tolerance must be finite and positive")
    args.workers = list(dict.fromkeys(args.workers))
    if 40 in args.workers:
        args.workers.remove(40)
        args.workers.insert(0, 40)
    args.prefetch_factors = list(dict.fromkeys(args.prefetch_factors))
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    backend = get_backend(args.backend)
    base = load_params_file(
        GNN / "configs/autotune" / f"{args.dataset}_w40.yaml"
    )
    cores = get_available_cpu_cores()
    eligible = [w for w in args.workers if cores is None or w <= cores]
    skipped = [w for w in args.workers if w not in eligible]
    if not eligible and not (args.mode == "sensitivity" and args.dry_run):
        raise ValueError(
            "No worker candidates fit this process's CPU allocation"
        )
    if skipped and args.mode == "autotune":
        print(
            f"CPU affinity allows {cores} cores; skipping workers {skipped}. Execution worker memory/CPU allocation still needs to fit."
        )
    if args.mode == "sensitivity":
        if skipped and not args.dry_run:
            raise ValueError(
                f"CPU affinity allows {cores} cores, but requested workers {skipped} do not fit. "
                "No jobs submitted; use a suitable launch allocation or explicitly change --workers. "
                "WSC Worker CPU/memory allocation must also fit."
            )
        if skipped:
            print(
                f"Preview only: workers {skipped} exceed launch CPU allocation ({cores}); execution would stop before submission."
            )
    else:
        args.workers = eligible
    with study_lock(output):
        if args.dry_run:
            if (output / "study.json").exists():
                raise ValueError(
                    "Use a separate output directory for a dry run of an existing study"
                )
            plan = {
                "backend": args.backend,
                "dataset": args.dataset,
                "mode": args.mode,
                "continue_on_failure": args.continue_on_failure,
                "workers": args.workers,
                "skipped_workers": skipped if args.mode == "autotune" else [],
                "blocked_workers": (
                    skipped if args.mode == "sensitivity" else []
                ),
                "launch_cpu_cores": cores,
                "cache": args.cache,
                "persistent_workers": args.persistent_workers,
                "wsc_workers": args.wsc_workers,
                "planned_trials": (
                    len(args.workers) * args.repeats
                    if args.mode == "sensitivity"
                    else None
                ),
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "confirm_steps": args.confirm_steps,
                "repeats": args.repeats,
                "prefetch_factors": args.prefetch_factors,
                "top_k": args.top_k,
                "budget_sec": args.budget_sec,
                "trial_timeout_sec": args.trial_timeout_sec,
                "commands": [],
            }
            for workers in args.workers:
                knobs = candidate(workers, persistent=args.persistent_workers)
                phase = (
                    "sensitivity" if args.mode == "sensitivity" else "workers"
                )
                folder = output / f"{phase}_{candidate_id(knobs)}_r1"
                path = output / f"preview_w{workers:02d}.yaml"
                path.write_text(
                    yaml.safe_dump(
                        prepare_config(
                            backend,
                            args,
                            base,
                            knobs,
                            folder / "model",
                            args.warmup_steps + args.measure_steps,
                        ),
                        sort_keys=False,
                    )
                )
                plan["commands"].append(
                    backend.command(path, folder / "model")
                )
            write_json(output / "plan.json", plan)
            print(f"Plan saved to {output / 'plan.json'}; no jobs submitted")
            return 0
        provenance = environment(args.backend)
        provenance["available_cpu_cores"] = cores
        return Study(args, base, output, provenance).run()


if __name__ == "__main__":

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"autotune: {exc}", file=sys.stderr)
        sys.exit(2)
