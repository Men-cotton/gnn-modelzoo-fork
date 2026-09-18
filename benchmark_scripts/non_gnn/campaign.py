#!/usr/bin/env python3
"""Prepare shared text data and measure selected profiles with repeated trials."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import uuid

import yaml

import prepare_data
import run
import measurements
import records

from cerebras.modelzoo.tools import benchmark_launcher


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("CSX", "GPU"), required=True)
    p.add_argument(
        "--only",
        action="append",
        choices=tuple(yaml.safe_load(run.PROFILES.read_text())),
        help="Repeat to select profiles; default: all four",
    )
    p.add_argument("--list-configs", action="store_true")
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--max-documents", type=run.positive, default=10000)
    p.add_argument(
        "--raw-text",
        type=Path,
        help="Optional UTF-8 file, blank-line-separated documents",
    )
    p.add_argument(
        "--llama-tokenizer",
        default=os.environ.get("NON_GNN_LLAMA_TOKENIZER", "meta-llama/Llama-3.2-1B"),
    )
    p.add_argument("--tokenizer-revision", default="main")
    p.add_argument(
        "--data-root", type=Path, default=run.ROOT / "model_dirs/non_gnn/data"
    )
    p.add_argument("--output-dir", type=Path, help="New campaign directory")
    p.add_argument(
        "--gpu-implementation", choices=("native", "modelzoo", "both"), default="native"
    )
    p.add_argument("--max-steps", type=run.positive, default=200)
    p.add_argument("--warmup-steps", type=run.nonnegative, default=20)
    p.add_argument("--num-workers", type=run.nonnegative, default=2)
    p.add_argument("--seed", type=run.nonnegative, default=42)
    p.add_argument("--repeats", type=run.positive, default=3)
    p.add_argument("--trial-timeout-sec", type=run.positive, default=9000)
    p.add_argument("--job-time-sec", type=run.positive, default=7200)
    p.add_argument("--effective-batch-size", type=run.positive)
    p.add_argument("--gpu-micro-batch-size", type=run.positive)
    p.add_argument("--csx-micro-batch-size", default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--mount-dir", type=Path, action="append", default=[])
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare data and all configs without starting jobs",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without downloads, writes, or launches",
    )
    benchmark_launcher.add_arguments(p)
    return p


def jobs(args, profiles, data):
    result = []
    implementations = (
        ("native", "modelzoo")
        if args.gpu_implementation == "both"
        else (args.gpu_implementation,)
    )
    if args.backend == "CSX":
        implementations = ("native",)  # GPU-only selection is irrelevant here.
    for repeat in range(1, args.repeats + 1):
        ordered_profiles = args.only if repeat % 2 else list(reversed(args.only))
        for profile in ordered_profiles:
            length = profiles[profile]["sequence_length"]
            for implementation in implementations:
                name = (
                    f"{profile}_{implementation}" if args.backend == "GPU" else profile
                )
                output = args.output_dir / f"r{repeat:02d}" / name
                argv = [
                    "--backend",
                    args.backend,
                    "--profile",
                    profile,
                    "--data-dir",
                    str(
                        data
                        / (
                            f"bert_msl{length}"
                            if profile.startswith("bert_")
                            else "llama_corpus"
                        )
                    ),
                    "--output-dir",
                    str(output),
                    "--llama-data-format",
                    "corpus",
                    "--gpu-implementation",
                    implementation,
                    "--max-steps",
                    str(args.max_steps),
                    "--warmup-steps",
                    str(args.warmup_steps),
                    "--num-workers",
                    str(args.num_workers),
                    "--seed",
                    str(args.seed),
                    "--csx-micro-batch-size",
                    args.csx_micro_batch_size,
                    "--repeat",
                    str(repeat),
                    "--study-dir",
                    str(args.output_dir),
                    "--trial-timeout-sec",
                    str(args.trial_timeout_sec),
                    "--job-time-sec",
                    str(args.job_time_sec),
                    "--prepare-only",
                ]
                if profile.startswith("bert_"):
                    argv += ["--vocab-file", str(data / "bert_vocab.txt")]
                for option in ("effective_batch_size", "gpu_micro_batch_size"):
                    if getattr(args, option) is not None:
                        argv += [
                            "--" + option.replace("_", "-"),
                            str(getattr(args, option)),
                        ]
                if args.backend == "GPU" and implementation == "native":
                    if args.compile:
                        argv.append("--compile")
                    if args.gradient_checkpointing:
                        argv.append("--gradient-checkpointing")
                for path in args.mount_dir:
                    argv += ["--mount-dir", str(path.resolve())]
                # Check options before any downloads or externally visible submissions.
                run.build_config(run.parser().parse_args(argv))
                result.append(
                    {
                        "name": f"{name}_r{repeat:02d}",
                        "profile": profile,
                        "repeat": repeat,
                        "gpu_implementation": implementation
                        if args.backend == "GPU"
                        else None,
                        "prepare_arguments": argv,
                        "config": str(output / "params.yaml"),
                        "state": "planned",
                    }
                )
    return result


def submit_gpu(job):
    output = Path(job["config"]).parent
    command = [
        "qsub",
        "-v",
        f"NON_GNN_CONFIG={job['config']}",
        str(run.ROOT / "benchmark_scripts/pegasus/run_non_gnn_nqsv.pbs"),
    ]
    result = subprocess.run(command, cwd=run.ROOT, text=True, capture_output=True)
    (output / "qsub.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"qsub failed for {job['name']}: {result.stderr.strip()}")
    job.update(state="qsub_returned", scheduler_response=result.stdout.strip())
    print(f"[submit] {job['name']}: {result.stdout.strip()}", flush=True)


class CampaignInterrupted(KeyboardInterrupt):
    def __init__(self, number):
        self.number = number


def main(argv=None):
    def interrupt(number, _frame):
        raise CampaignInterrupted(number)

    handlers = {
        number: signal.signal(number, interrupt)
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        return run_campaign(argv)
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


def run_campaign(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    profiles = yaml.safe_load(run.PROFILES.read_text())
    if args.list_configs:
        print(run.PROFILES.read_text(), end="")
        return 0
    args.only = list(dict.fromkeys(args.only or profiles))
    args.data_root = args.data_root.resolve()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = (
        args.output_dir
        or run.ROOT
        / "model_dirs/non_gnn"
        / f"campaign_{args.backend.lower()}_{stamp}_{uuid.uuid4().hex[:8]}"
    ).resolve()
    state = None
    active_job = None
    try:
        records.validate_output_dir(args.output_dir)
        if args.backend == "GPU" and (
            "," in str(args.output_dir) or "\n" in str(args.output_dir)
        ):
            raise ValueError("Output paths must not contain commas or newlines")
        if args.backend == "GPU" and args.mount_dir:
            raise ValueError("--mount-dir applies only to CSX")
        spec = prepare_data.request_spec(args, profiles, run.VOCAB)
        data = prepare_data.cache_path(args.data_root, spec)
        plan = jobs(args, profiles, data)
        if args.dry_run:
            print(
                f"[data] verify {data}/manifest.json; reuse matching checksummed data; otherwise download/preprocess"
            )
            print(f"[source] {json.dumps(spec['source'])}")
            for job in plan:
                if args.prepare_only:
                    print(f"[prepare] {job['name']}")
                elif args.backend == "GPU":
                    print(
                        shlex.join(
                            [
                                "qsub",
                                "-v",
                                f"NON_GNN_CONFIG={job['config']}",
                                str(
                                    run.ROOT
                                    / "benchmark_scripts/pegasus/run_non_gnn_nqsv.pbs"
                                ),
                            ]
                        )
                    )
                else:
                    print(
                        shlex.join(
                            [
                                sys.executable,
                                str(Path(__file__).with_name("run.py")),
                                "--backend",
                                "CSX",
                                "--execute-config",
                                job["config"],
                            ]
                        )
                    )
            return 0
        if args.output_dir.exists():
            raise ValueError(
                f"Campaign directory already exists: {args.output_dir}; use a new output directory (data cache is reused)"
            )
        if (
            args.backend == "GPU"
            and not args.prepare_only
            and shutil.which("qsub") is None
        ):
            raise ValueError("qsub not found; run on Pegasus or use --prepare-only")
        run.check_runtime(args.backend)
        run.check_dependencies()
        if args.detach:
            return run.detach_driver(args, argv, Path(__file__).resolve(), "non-gnn")
        data, manifest = prepare_data.prepare(args, profiles, run.VOCAB)
        for profile in args.only:
            effective = (
                args.effective_batch_size or profiles[profile]["effective_batch_size"]
            )
            length = profiles[profile]["sequence_length"]
            available = (
                manifest["stats"]["bert"][str(length)]["samples"]
                if profile.startswith("bert_")
                else (manifest["stats"]["llama"]["tokens"] - 1) // length
            )
            if available < effective:
                raise ValueError(
                    f"{profile}: {available} samples, fewer than one effective batch ({effective}); increase --max-documents"
                )
        args.output_dir.mkdir(parents=True)
        state = {
            "backend": args.backend,
            "state": "preparing",
            "repeats": args.repeats,
            "seed": args.seed,
            "created_at": benchmark_launcher.timestamp(),
            **records.source_identity(),
            "data": str(data),
            "data_manifest_sha256": prepare_data.sha256(data / "manifest.json"),
            "jobs": plan,
        }
        state_path = args.output_dir / "campaign.json"

        def save():
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(state_path)

        save()
        # Validate ALL configurations before launching the first job.
        for job in plan:
            active_job = job
            job["state"] = "preparing"
            save()
            run.main(job["prepare_arguments"])
            job["state"] = "prepared"
            active_job = None
            save()
        if args.prepare_only:
            state["state"] = "prepared"
            save()
            print(f"[prepared] {len(plan)} configurations: {state_path}")
            return 0
        state["state"] = "running"
        save()
        failed = False
        interrupted = False
        interrupted_code = 0
        for job in plan:
            active_job = job
            try:
                if args.backend == "GPU":
                    submit_gpu(job)
                else:
                    print(f"[run] {job['name']}", flush=True)
                    result = run.execute_prepared(Path(job["config"]), backend="CSX")
                    job.update(result)
                    failed = failed or run.execution_exit_code(result) != 0
                    interrupted = result["state"] == "interrupted"
                    if interrupted:
                        interrupted_code = result["exit_code"]
            except (OSError, ValueError, RuntimeError) as exc:
                job.update(state="launch_failed", error=str(exc))
                failed = True
                print(f"[failed] {job['name']}: {exc}", flush=True)
            active_job = None
            save()
            measurements.summarize_campaign(args.output_dir, plan)
            if interrupted:
                break
        state["state"] = (
            "interrupted"
            if interrupted
            else "failed"
            if failed
            else "submitted"
            if args.backend == "GPU"
            else "completed"
        )
        state["finished_at"] = benchmark_launcher.timestamp()
        save()
        measurements.summarize_campaign(args.output_dir, plan)
        print(f"[campaign] {state['state']}: {state_path}", flush=True)
        return interrupted_code if interrupted else 2 if failed else 0
    except CampaignInterrupted as exc:
        if state is not None:
            state.update(
                state="interrupted", finished_at=benchmark_launcher.timestamp()
            )
            if active_job is not None:
                active_job.update(state="interrupted", exit_code=128 + exc.number)
            save()
            measurements.summarize_campaign(args.output_dir, plan)
        print(f"[campaign] interrupted by signal {exc.number}", flush=True)
        return 128 + exc.number
    except (OSError, ValueError, RuntimeError, SystemExit) as exc:
        if state is not None:
            state.update(
                state="failed",
                error=str(exc),
                finished_at=benchmark_launcher.timestamp(),
            )
            if active_job is not None:
                active_job.update(state="prepare_failed", error=str(exc))
            save()
            measurements.summarize_campaign(args.output_dir, plan)
        p.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
