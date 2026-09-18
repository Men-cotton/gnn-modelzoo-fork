#!/usr/bin/env python3
"""Prepare or reuse public text data, then launch all selected training profiles."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid

import yaml

import prepare_data
import run


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
    for profile in args.only:
        length = profiles[profile]["sequence_length"]
        for implementation in implementations:
            name = f"{profile}_{implementation}" if args.backend == "GPU" else profile
            output = args.output_dir / name
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
                    "name": name,
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


def start_csx(job):
    output = Path(job["config"]).parent
    command = [
        sys.executable,
        str(Path(__file__).with_name("execute_csx.py")),
        "--config",
        job["config"],
    ]
    with (output / "client.log").open("x") as log:
        proc = subprocess.Popen(
            command,
            cwd=run.ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    job.update(state="client_started", client_pid=proc.pid)
    print(
        f"[launch] {job['name']}: client PID {proc.pid}; {output / 'client.log'}",
        flush=True,
    )


def main(argv=None):
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
    try:
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
                                str(Path(__file__).with_name("execute_csx.py")),
                                "--config",
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
            run.main(job["prepare_arguments"])
            job["state"] = "prepared"
            save()
        if args.prepare_only:
            print(f"[prepared] {len(plan)} configurations: {state_path}")
            return 0
        for job in plan:
            try:
                submit_gpu(job) if args.backend == "GPU" else start_csx(job)
            except Exception as exc:
                job.update(state="launch_failed", error=str(exc))
                save()
                raise
            save()
        print(f"[campaign] {state_path}", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
