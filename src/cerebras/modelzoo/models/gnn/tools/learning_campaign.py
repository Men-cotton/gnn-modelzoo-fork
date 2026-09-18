"""Record CSX learning curves, tune input workers, and export configurations."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import signal
import sys
import time

import yaml

from cerebras.modelzoo.common.utils.run.config_loader import load_params_file
from cerebras.modelzoo.models.gnn.tools import autotune, measure_window
from cerebras.modelzoo.models.gnn.tools.autotune_backends import CSXBackend
from cerebras.modelzoo.models.gnn.tools.job_labels import apply_job_labels
from cerebras.modelzoo.models.gnn.worker_validation import get_available_cpu_cores
from cerebras.modelzoo.tools import benchmark_launcher


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def shared_base(dataset: str) -> dict:
    """Keep the same architecture, optimization and precision in every phase."""
    base = load_params_file(
        autotune.GNN
        / "configs"
        / f"params_graphsage_ogbn_{dataset}_accuracy_nocache.yaml"
    )
    init = base["trainer"]["init"]
    # SDK and native torch defaults differ; encode the comparison contract explicitly.
    init["optimizer"]["AdamW"].update(eps=1.0e-6, betas=[0.9, 0.999])
    init.setdefault("checkpoint", {})
    init["backend"] = dict(
        backend_type="CSX", cluster_config=dict(num_csx=1, num_workers_per_csx=1)
    )
    for name in ("train_dataloader", "val_dataloader"):
        loader = base["trainer"]["fit"][name]
        loader.update(
            cache_fraction=None,
            static_batch_cache_size=0,
            use_fake_data=False,
            drop_last_batch=False,
            worker_diagnostics={"enabled": False},
        )
    return base


def learning_config(base: dict, args, knobs: dict, folder: Path) -> dict:
    config = CSXBackend.prepare_config(
        base, knobs, folder / "model", args.learning_steps, args.job_time_sec
    )
    trainer = config["trainer"]
    trainer["init"]["model"]["task"]["compute_eval_metrics"] = True
    trainer["init"]["loop"].update(eval_frequency=args.eval_every, eval_steps=None)
    validation = deepcopy(base["trainer"]["fit"]["val_dataloader"])
    validation.update(knobs, split="valid", shuffle=False)
    trainer["fit"]["val_dataloader"] = validation
    trainer["fit"]["train_dataloader"]["split"] = "train"
    apply_job_labels(
        config,
        mode="learning",
        repeat=1,
        trial_dir=folder,
        study_dir=args.output,
    )
    return config


def collect_learning(log: Path, args) -> dict:
    """Collect complete curves; researchers decide whether learning is adequate."""
    input_contract = measure_window.read_input_contract(log)
    training, evaluations = [], []
    pending = {}
    for line in log.read_text(errors="replace").splitlines():
        match = re.search(r"Train Device=CSX, Step=(\d+), Loss=([^,\s]+)", line)
        if match:
            training.append((int(match[1]), float(match[2])))
        match = re.search(r"Eval Device=CSX, GlobalStep=(\d+),.*?Loss=([^,\s]+)", line)
        if match:
            if not math.isfinite(float(match[2])):
                raise ValueError("Nonfinite validation batch loss")
            pending["step"] = int(match[1])
        match = re.search(r"Avg Eval Loss:\s*(\S+)", line)
        if match:
            pending["loss"] = float(match[1])
        match = re.search(r"eval/masked_accuracy\s*=\s*(\S+)", line)
        if match:
            if "accuracy" in pending:
                raise ValueError("Duplicate validation accuracy")
            pending["accuracy"] = float(match[1])
        if "Evaluation completed successfully!" in line:
            if set(pending) != {"step", "loss", "accuracy"}:
                raise ValueError("Validation completed without step, loss and accuracy")
            evaluations.append(pending)
            pending = {}
    if pending:
        raise ValueError("Incomplete final validation")
    if [step for step, _ in training] != list(range(10, args.learning_steps + 1, 10)):
        raise ValueError("Missing, duplicate or out-of-order training steps")
    expected = list(range(args.eval_every, args.learning_steps + 1, args.eval_every))
    if [row["step"] for row in evaluations] != expected or len(expected) < 2:
        raise ValueError("Missing, duplicate or unexpected held-out evaluations")
    losses = [loss for _, loss in training]
    if any(not math.isfinite(loss) or loss < 0 for loss in losses):
        raise ValueError("Nonfinite or negative training loss")
    if any(
        not math.isfinite(row["loss"])
        or row["loss"] < 0
        or not math.isfinite(row["accuracy"])
        or not 0 <= row["accuracy"] <= 1
        for row in evaluations
    ):
        raise ValueError("Invalid held-out loss or accuracy")
    return dict(
        review_status="pending_human_review",
        final_validation_accuracy=evaluations[-1]["accuracy"],
        evaluations=evaluations,
        training_losses=training,
        input_contract=input_contract,
        interpretation="Single-seed loss and full valid-split curves. No automated judgment of learning or accuracy equivalence.",
    )


def tune_arguments(args, budget: int):
    return autotune.parse_args(
        [
            "--dataset",
            args.dataset,
            "--output",
            str(args.output / "tuning"),
            "--job-label-study",
            str(args.output),
            "--workers",
            *map(str, args.workers),
            "--prefetch-factors",
            *map(str, args.prefetch_factors),
            "--persistent-workers",
            "--wsc-workers",
            "1",
            "--top-k",
            str(args.top_k),
            "--repeats",
            str(args.repeats),
            "--measure-steps",
            str(args.measure_steps),
            "--confirm-steps",
            str(args.confirm_steps),
            "--stability-tolerance-percent",
            str(args.stability_tolerance_percent),
            "--job-time-sec",
            str(args.job_time_sec),
            "--trial-timeout-sec",
            str(args.trial_timeout_sec),
            "--budget-sec",
            str(budget),
        ]
    )


def handoff(args, config: dict, state: dict) -> None:
    folder = args.output / "handoff"
    folder.mkdir(exist_ok=True)
    csx = deepcopy(config)
    csx["trainer"]["init"]["model_dir"] = str(folder / "csx_model")
    apply_job_labels(
        csx,
        mode="selected",
        repeat=1,
        trial_dir=folder / "selected_csx",
        study_dir=args.output,
    )
    write_config(folder / "selected_csx.yaml", csx)
    fixed = deepcopy(csx)
    fixed["trainer"]["init"].pop("backend", None)
    fixed["trainer"]["init"]["model_dir"] = str(folder / "fixed_shape_gpu_model")
    write_config(folder / "selected_fixed_shape_gpu.yaml", fixed)
    pyg = deepcopy(fixed)
    pyg["trainer"]["init"]["model_dir"] = str(folder / "pyg_model")
    pyg["trainer"]["init"]["benchmark"] = {"warmup_steps": 40}
    # PyG interprets None as automatic GPU caching; explicit 0 means uncached.
    pyg["trainer"]["fit"]["train_dataloader"]["cache_fraction"] = 0.0
    write_config(folder / "pyg_reference.yaml", pyg)
    diagnostic = deepcopy(csx)
    diagnostic["trainer"]["init"]["model_dir"] = str(folder / "diagnostic_model")
    diagnostic["trainer"]["init"]["loop"]["max_steps"] = 80
    diagnostic["trainer"]["fit"]["train_dataloader"]["worker_diagnostics"] = dict(
        enabled=True,
        output_dir=str(folder / "worker_diagnostics"),
        max_batches=80,
        max_snapshots=120,
        snapshot_interval_seconds=5.0,
        resource_monitor=True,
        resource_monitor_pss=True,
    )
    apply_job_labels(
        diagnostic,
        mode="diagnostic",
        repeat=1,
        trial_dir=folder / "diagnostic",
        study_dir=args.output,
    )
    write_config(folder / "selected_csx_diagnostics.yaml", diagnostic)
    write_config(
        folder / "selected_learning.yaml",
        learning_config(
            shared_base(args.dataset),
            args,
            state["ranking"][0]["knobs"],
            folder / "selected_learning",
        ),
    )
    autotune.write_json(
        folder / "selection.json",
        dict(
            fingerprint=state["fingerprint"],
            dataset=args.dataset,
            review_status="pending_human_review",
            baseline_learning=state["baseline"]["curves"],
            selected_learning=state.get("selected", {}).get("curves"),
            ranking=state["ranking"],
            config_sha256=autotune.digest(config),
            gpu_status="Unmeasured starting configurations; tune GPU input parameters independently before comparison.",
            comparison="Report fixed-shape GPU and ordinary PyG separately; retain numerator definitions and execution provenance.",
        ),
    )


def execute(args, base: dict, provenance: dict) -> int:
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"budget_sec", "detach", "tmux_session", "dry_run"}
    }
    fingerprint = autotune.digest(
        dict(
            settings=settings,
            base=base,
            environment={k: v for k, v in provenance.items() if k != "git_status"},
        )
    )
    path = args.output / "learning_campaign.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["fingerprint"] != fingerprint:
            raise ValueError(
                "Settings, sources or environment changed; use a new output"
            )
        if args.budget_sec < state["budget_sec"]:
            raise ValueError("The cumulative budget cannot decrease")
        if any(
            state.get(name, {}).get("status")
            in {"running", "interrupted", "failed", "timeout"}
            for name in ("baseline", "selected")
        ):
            raise ValueError(
                "Previous learning client did not finish. Inspect its jobs and use a new output"
            )
    else:
        state = dict(
            fingerprint=fingerprint,
            environment=provenance,
            settings=settings,
            status="ready",
            review_status="pending_human_review",
            learning_seconds=0.0,
        )
    state["budget_sec"] = args.budget_sec

    def save():
        autotune.write_json(path, state)

    def learning(name: str, knobs: dict, tuning_used: float = 0) -> bool:
        if name in state:
            return state[name]["status"] == "completed"
        if (
            args.budget_sec - state["learning_seconds"] - tuning_used
            < args.trial_timeout_sec + 10
        ):
            state["status"] = "budget_exhausted"
            save()
            return False
        folder = args.output / name
        folder.mkdir(exist_ok=False)
        config = learning_config(base, args, knobs, folder)
        write_config(folder / "params.yaml", config)
        command = CSXBackend.command(folder / "params.yaml", folder / "model")
        row = dict(
            status="running",
            command=command,
            knobs=knobs,
            config_sha256=autotune.digest(config),
            started_at=autotune.timestamp(),
        )
        state[name] = row
        state["status"] = name
        save()
        print(f"Learning curves: {name}; {folder / 'train.log'}", flush=True)
        started = time.monotonic()
        try:
            row.update(
                autotune.execute(command, folder / "train.log", args.trial_timeout_sec)
            )
            if row["status"] == "completed":
                row["curves"] = collect_learning(folder / "train.log", args)
                autotune.write_json(folder / "learning_curves.json", row["curves"])
        except (ValueError, OSError) as exc:
            row.update(status="invalid_measurement", reason=str(exc))
        finally:
            row["client_wall_seconds"] = time.monotonic() - started
            row["finished_at"] = autotune.timestamp()
            contents = (
                (folder / "train.log").read_text(errors="replace")
                if (folder / "train.log").exists()
                else ""
            )
            row["job_ids"] = sorted(set(re.findall(r"\bwsjob-[A-Za-z0-9-]+", contents)))
            state["learning_seconds"] += row["client_wall_seconds"]
            if row["status"] != "completed":
                state["status"] = f"{name}_failed"
            autotune.write_json(folder / "result.json", row)
            save()
        return row["status"] == "completed"

    save()
    if state["status"] == "completed":
        return 0
    if not learning(
        "baseline", autotune.candidate(args.baseline_workers, persistent=True)
    ):
        return 2
    tuning_folder = args.output / "tuning"
    tuning_folder.mkdir(exist_ok=True)
    # Optional selected-config curves have their own reserved client budget.
    reserve = args.trial_timeout_sec + 10 if args.repeat_learning_with_selected else 0
    tuning_budget = int(
        args.budget_sec - state["baseline"]["client_wall_seconds"] - reserve
    )
    if tuning_budget < args.trial_timeout_sec + 10:
        state["status"] = "budget_exhausted"
        save()
        return 2
    with autotune.study_lock(tuning_folder):
        tuning_base = deepcopy(base)
        tuning_base["trainer"]["init"]["model"]["task"]["compute_eval_metrics"] = False
        study = autotune.Study(
            tune_arguments(args, tuning_budget), tuning_base, tuning_folder, provenance
        )
        state["status"] = "tuning"
        save()
        code = study.run()
    state["tuning_status"] = study.state["status"]
    state["tuning_seconds"] = study.state["used_sec"]
    if code:
        state["status"] = study.state["status"]
        save()
        return 2
    state["ranking"] = study.state["ranking"]
    save()
    if args.repeat_learning_with_selected and not learning(
        "selected", state["ranking"][0]["knobs"], state["tuning_seconds"]
    ):
        return 2
    handoff(args, yaml.safe_load((tuning_folder / "best.yaml").read_text()), state)
    state["status"] = "completed"
    state["used_seconds"] = state["learning_seconds"] + state["tuning_seconds"]
    save()
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    benchmark_launcher.add_arguments(parser)
    parser.add_argument("--dataset", choices=("arxiv", "products"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8, 12, 16])
    parser.add_argument("--baseline-workers", type=int, default=2)
    parser.add_argument("--prefetch-factors", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=400)
    parser.add_argument("--confirm-steps", type=int, default=800)
    parser.add_argument("--stability-tolerance-percent", type=float, default=2.0)
    parser.add_argument("--learning-steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--repeat-learning-with-selected", action="store_true")
    parser.add_argument("--budget-sec", type=int, default=86400)
    parser.add_argument("--job-time-sec", type=int, default=7200)
    parser.add_argument("--trial-timeout-sec", type=int, default=9000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.learning_steps is None:
        args.learning_steps = 500 if args.dataset == "arxiv" else 1000
    if args.eval_every is None:
        args.eval_every = args.learning_steps // 5
    if (
        args.learning_steps < 60
        or args.eval_every <= 0
        or args.eval_every % 10
        or args.learning_steps % args.eval_every
        or args.learning_steps // args.eval_every < 2
    ):
        parser.error(
            "Require >=60 learning steps and >=2 evenly spaced full evaluations; eval-every must be a multiple of 10"
        )
    if args.baseline_workers < 0:
        parser.error("baseline-workers must be nonnegative")
    args.workers = list(dict.fromkeys(args.workers))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    args.output = (
        args.output
        or autotune.ROOT / f"model_dirs/hpcasia_learning/{args.dataset}_{stamp}"
    ).resolve()
    tune_arguments(args, args.budget_sec)  # Reuse the tuner's budget/window validation.
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
            name="gnn-learning-campaign",
            session=args.tmux_session,
        )

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    base = shared_base(args.dataset)
    with autotune.study_lock(args.output):
        if args.dry_run:
            if (args.output / "learning_campaign.json").exists():
                raise ValueError("Use a separate output for a dry-run")
            config = learning_config(
                base,
                args,
                autotune.candidate(args.baseline_workers, persistent=True),
                args.output / "baseline",
            )
            write_config(args.output / "baseline" / "params.yaml", config)
            write_config(args.output / "tuning_base.yaml", base)
            autotune.write_json(
                args.output / "plan.json",
                dict(
                    settings={
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    stages=["baseline_learning_curves", "input_autotune"]
                    + (
                        ["selected_learning_curves"]
                        if args.repeat_learning_with_selected
                        else []
                    )
                    + ["export_handoff"],
                    review_status="pending_human_review",
                    failure_policy="Stop after client failure or incomplete/nonfinite measurements and on remote job uncertainty. Accuracy values never determine progression.",
                ),
            )
            print(f"Preview written: {args.output}; no training submitted", flush=True)
            return 0
        cores = get_available_cpu_cores()
        if cores is not None and max(args.workers + [args.baseline_workers]) > cores:
            raise ValueError(
                f"Worker counts exceed launch affinity ({cores}); no jobs submitted"
            )
        return execute(args, base, autotune.environment("csx"))


if __name__ == "__main__":
    raise SystemExit(main())
