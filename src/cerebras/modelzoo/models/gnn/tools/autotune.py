"""Sequential CS-3 input tuning using the cs3_autotune_overrides measurement windows."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
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
from cerebras.modelzoo.models.gnn.worker_validation import get_available_cpu_cores

# Also support direct execution from the documented GNN directory.
if __package__:
    from .measure_window import summarize
else:
    from measure_window import summarize

GNN = Path(__file__).resolve().parents[1]
ROOT = GNN.parents[4]
UNCERTAIN = {"running", "interrupted", "timeout", "failed"}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


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


def prepare_config(base, knobs, model_dir, steps, job_time_sec):
    config = deepcopy(base)
    trainer = config["trainer"]
    init = trainer["init"]
    init["model_dir"] = str(model_dir)
    init["loop"].update(
        num_steps=None,
        max_steps=steps,
        num_epochs=None,
        steps_per_epoch=None,
        eval_frequency=None,
    )
    init["checkpoint"].update(
        steps=None, autoload_last_checkpoint=False, save_initial_checkpoint=False
    )
    init["logging"]["log_steps"] = 10
    init["backend"]["cluster_config"].update(num_csx=1, job_time_sec=job_time_sec)
    loader = trainer["fit"]["train_dataloader"]
    loader.update(knobs)
    loader.update(
        batch_size=4096,
        cache_fraction=None,
        static_batch_cache_size=0,
        use_fake_data=False,
    )
    trainer["fit"].update(ckpt_path=None, val_dataloader=None)
    trainer.update(validate=None, validate_all=None)
    return config


def command(config, model_dir):
    return [
        "uv",
        "run",
        "--no-sync",
        "--",
        "cszoo",
        "fit",
        str(config),
        "--target_device",
        "CSX",
        "--model_dir",
        str(model_dir),
    ]


def environment():
    # Inspect the same uv environment that will run cszoo, without syncing it.
    code = """import importlib.metadata as m, json, sys, shutil
import cerebras.pytorch
import cerebras.modelzoo as mz
print(json.dumps({"python": sys.version, "executable": sys.executable,
    "prefix": sys.prefix, "modelzoo_path": mz.__file__,
    "cszoo": shutil.which("cszoo"), "sdk": m.version("cerebras-pytorch"),
    "packages": sorted((d.metadata["Name"], d.version) for d in m.distributions())}))"""
    result = subprocess.run(
        ["uv", "run", "--no-sync", "--", "python", "-c", code],
        cwd=GNN,
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout.strip().splitlines()[-1])
    if (
        not info["python"].startswith("3.11.")
        or info["sdk"] != "2.10.0"
        or not info["cszoo"]
        or Path(info["cszoo"]).parent != Path(info["prefix"]) / "bin"
        or Path(info["modelzoo_path"]).resolve().parent != GNN.parents[1]
    ):
        raise ValueError(
            "Prepare Python 3.11, Cerebras 2.10.0 and editable cszoo before tuning"
        )
    info["uv"] = subprocess.check_output(["uv", "--version"], text=True).strip()
    info["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    info["git_status"] = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    )
    # Include uncommitted and untracked implementation changes without storing their contents.
    sources = list((ROOT / "src").rglob("*.py"))
    info["source_sha256"] = digest(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources)
        }
    )
    return info


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
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
            "failure_reason": None if code == 0 else f"cszoo exited with {code}",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "failure_reason": "Client wall-time limit reached"}
    except KeyboardInterrupt:
        return {"status": "interrupted", "failure_reason": "Tuner interrupted"}
    except OSError as exc:
        return {"status": "failed", "failure_reason": str(exc)}
    finally:
        if process is not None:
            stop_process(process)
        print(f"  client elapsed: {time.monotonic() - started:.1f}s", flush=True)


def rank(records, phase, repeats=1):
    groups = {}
    for row in records:
        if row["phase"] == phase:
            groups.setdefault(row["candidate_id"], []).append(row)
    ranked = []
    for key, rows in groups.items():
        # A failed or unstable repeat disqualifies the candidate; never cherry-pick it away.
        if len(rows) != repeats or any(r["status"] != "completed" for r in rows):
            continue
        rates = [r["measurement"]["nominal_slots_per_second"] for r in rows]
        ranked.append(
            {
                "candidate_id": key,
                "knobs": rows[0]["knobs"],
                "median_nominal_slots_per_second": statistics.median(rates),
                "min_nominal_slots_per_second": min(rates),
                "max_nominal_slots_per_second": max(rates),
                "repeats": len(rates),
            }
        )
    return sorted(
        ranked, key=lambda r: (-r["median_nominal_slots_per_second"], r["candidate_id"])
    )


class Study:
    def __init__(self, args, base, output, provenance):
        self.args, self.base, self.output = args, base, output
        settings = {
            k: v
            for k, v in vars(args).items()
            if k not in {"output", "dry_run", "acknowledge_stopped_jobs", "budget_sec"}
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
                raise ValueError("A resumed study's total budget cannot decrease")
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
            (output / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
        self.state["budget_sec"] = args.budget_sec
        self.save()

    def save(self):
        write_json(self.output / "study.json", self.state)

    def trial(self, knobs, phase, repeat, measured_steps):
        key = candidate_id(knobs)
        trial_id = f"{phase}_{key}_r{repeat}"
        prior = next(
            (r for r in self.state["trials"] if r["trial_id"] == trial_id), None
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
            self.base,
            knobs,
            folder / "model",
            self.args.warmup_steps + measured_steps,
            self.args.job_time_sec,
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
            "command": command(config_path, folder / "model"),
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
            execute(row["command"], folder / "train.log", self.args.trial_timeout_sec)
        )
        row["client_wall_seconds"] = time.monotonic() - start
        row["finished_at"] = timestamp()
        self.state["used_sec"] += row["client_wall_seconds"]
        log = folder / "train.log"
        contents = log.read_text(errors="replace") if log.exists() else ""
        row["job_ids"] = sorted(set(re.findall(r"\bwsjob-[A-Za-z0-9-]+", contents)))
        row["performance_files"] = [
            str(p.relative_to(folder)) for p in sorted(folder.rglob("performance.json"))
        ]
        if row["status"] == "completed":
            try:
                row["measurement"] = summarize(
                    log,
                    self.args.warmup_steps,
                    self.args.warmup_steps + measured_steps,
                    4096,
                    self.args.stability_tolerance_percent,
                )
                check = row["measurement"]["half_window_check"]
                if not check or not check["within_tolerance"]:
                    row.update(
                        status="unstable",
                        failure_reason="Half-window stability check failed or midpoint missing",
                    )
            except (ValueError, OSError) as exc:
                row.update(status="invalid_measurement", failure_reason=str(exc))
        write_json(folder / "result.json", row)
        self.save()
        if row["status"] in UNCERTAIN:
            self.state["status"] = "job_stop_unconfirmed"
            self.save()
            raise StopIteration
        return row

    def run(self):
        if any(
            r["status"] in UNCERTAIN and not r.get("job_stop_acknowledged_at")
            for r in self.state["trials"]
        ):
            raise ValueError(
                "Previous client failed/stopped. Confirm its CSX job has ended, then use --acknowledge-stopped-jobs"
            )
        try:
            for workers in self.args.workers:
                self.trial(candidate(workers), "workers", 1, self.args.measure_steps)
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
                    {r["candidate_id"]: r for r in rank(self.state["trials"], "loader")}
                )
                pool = sorted(
                    combined.values(),
                    key=lambda r: -r["median_nominal_slots_per_second"],
                )[: self.args.top_k]
            self.state["finalists"] = pool
            self.save()
            # Alternate order between rounds so candidates are not measured in three-job blocks.
            for repeat in range(1, self.args.repeats + 1):
                for row in pool if repeat % 2 else list(reversed(pool)):
                    self.trial(row["knobs"], "confirm", repeat, self.args.confirm_steps)
            self.state["ranking"] = rank(
                self.state["trials"], "confirm", self.args.repeats
            )
            if self.state["ranking"]:
                best = self.state["ranking"][0]
                # Emit a reproducible benchmark config, not an unbounded production training config.
                config = prepare_config(
                    self.base,
                    best["knobs"],
                    self.output / "best_model",
                    self.args.warmup_steps + self.args.confirm_steps,
                    self.args.job_time_sec,
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
                {k: self.state[k] for k in ("status", "used_sec", "budget_sec")},
                indent=2,
            )
        )
        return 0 if self.state["status"] == "completed" else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("arxiv", "products"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[40, 0, 4, 8, 16, 32])
    parser.add_argument("--prefetch-factors", type=int, nargs="+", default=[])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--confirm-steps", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stability-tolerance-percent", type=float, default=2.0)
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
        help="Write the resolved plan without launching uv/cszoo",
    )
    parser.add_argument(
        "--acknowledge-stopped-jobs",
        action="store_true",
        help="Resume after independently verifying all previous failed/stopped CSX jobs have ended",
    )
    args = parser.parse_args(argv)
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
    if args.warmup_steps % 10 or args.measure_steps % 20 or args.confirm_steps % 20:
        parser.error(
            "warmup must be a multiple of 10; measured windows must be multiples of 20 (exact endpoints/midpoints)"
        )
    if args.confirm_steps < args.measure_steps:
        parser.error("confirm-steps must be at least measure-steps")
    if (
        args.trial_timeout_sec < args.job_time_sec
        or args.budget_sec < args.trial_timeout_sec + 10
    ):
        parser.error(
            "Require budget-sec >= trial-timeout-sec + 10 (cleanup reserve), and trial-timeout-sec >= job-time-sec"
        )
    if any(w < 0 for w in args.workers) or any(p < 1 for p in args.prefetch_factors):
        parser.error("Workers must be non-negative and prefetch factors positive")
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
    base = load_params_file(GNN / "configs/autotune" / f"{args.dataset}_w40.yaml")
    cores = get_available_cpu_cores()
    eligible = [w for w in args.workers if cores is None or w <= cores]
    skipped = [w for w in args.workers if w not in eligible]
    if not eligible:
        raise ValueError("No worker candidates fit this process's CPU allocation")
    if skipped:
        print(
            f"CPU affinity allows {cores} cores; skipping workers {skipped}. Remote worker memory/CPU allocation still needs to fit."
        )
    args.workers = eligible
    with study_lock(output):
        if args.dry_run:
            if (output / "study.json").exists():
                raise ValueError(
                    "Use a separate output directory for a dry run of an existing study"
                )
            plan = {
                "dataset": args.dataset,
                "workers": eligible,
                "skipped_workers": skipped,
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
            for workers in eligible:
                knobs = candidate(workers)
                folder = output / f"workers_{candidate_id(knobs)}_r1"
                path = output / f"preview_w{workers:02d}.yaml"
                path.write_text(
                    yaml.safe_dump(
                        prepare_config(
                            base,
                            knobs,
                            folder / "model",
                            args.warmup_steps + args.measure_steps,
                            args.job_time_sec,
                        ),
                        sort_keys=False,
                    )
                )
                plan["commands"].append(command(path, folder / "model"))
            write_json(output / "plan.json", plan)
            print(f"Plan saved to {output / 'plan.json'}; no jobs submitted")
            return 0
        provenance = environment()
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
