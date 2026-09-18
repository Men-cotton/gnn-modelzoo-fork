#!/usr/bin/env python3
"""Prepare one paired Model Zoo training configuration and optionally run CSX.

Pegasus uses --prepare-only before qsub, then runs the saved config on the GPU.
--dry-run only prints the resolved YAML and command; it creates no files.
"""

import argparse
import copy
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time

import yaml

from cerebras.modelzoo.tools import benchmark_launcher

import records

ROOT = Path(__file__).resolve().parents[2]
PROFILES = Path(__file__).with_name("profiles.yaml")
VOCAB = (
    ROOT
    / "src/cerebras/modelzoo/models/vocab/google_research_uncased_L-12_H-768_A-12.txt"
)


def check_runtime(backend):
    if backend == "CSX" and shutil.which("torch-cirh-opt") is None:
        raise ValueError(
            "CSX compiler torch-cirh-opt is not on PATH. "
            "Launch through the benchmark shell script (uv run --no-sync), "
            "or use uv run --no-sync --project "
            + shlex.quote(str(ROOT))
            + " python benchmark_scripts/non_gnn/campaign.py --backend CSX. "
            "If it is still missing under uv run, check that cerebras_pytorch "
            "is installed completely and .venv/bin/torch-cirh-opt is executable."
        )


def check_dependencies():
    """Check real imports before downloading data or submitting any jobs."""
    failures = []
    for name in ("datasets", "transformers", "torchvision", "h5py", "filelock"):
        try:
            importlib.import_module(name)
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        setup = ROOT / "benchmark_scripts/non_gnn/setup.sh"
        raise ValueError(
            "Non-GNN dependencies are missing or cannot be imported in "
            f"{sys.executable}:\n  "
            + "\n  ".join(failures)
            + f"\nRun: bash {shlex.quote(str(setup))}"
            + "\nThis adds dependencies to the existing .venv and preserves its PyTorch build."
            + " No data has been downloaded and no jobs have been submitted."
        )


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("CSX", "GPU"), required=True)
    p.add_argument("--profile", choices=tuple(yaml.safe_load(PROFILES.read_text())))
    p.add_argument("--list-configs", action="store_true")
    p.add_argument("--check-dependencies", action="store_true")
    p.add_argument("--preflight-error", help=argparse.SUPPRESS)
    p.add_argument(
        "--preflight-exit-code", type=positive, default=2, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--execute-config",
        type=Path,
        help="Execute an unchanged prepared configuration",
    )
    p.add_argument(
        "--data-dir", type=Path, help="Preprocessed training data on this system"
    )
    p.add_argument("--vocab-file", type=Path, default=VOCAB, help="BERT vocabulary")
    p.add_argument(
        "--llama-data-format", choices=("sample", "corpus"), default="sample"
    )
    p.add_argument("--effective-batch-size", type=positive)
    p.add_argument("--gpu-micro-batch-size", type=positive)
    p.add_argument(
        "--gpu-implementation", choices=("native", "modelzoo"), default="native"
    )
    p.add_argument(
        "--compile", action="store_true", help="torch.compile for the native GPU model"
    )
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Activation checkpointing for native GPU",
    )
    p.add_argument(
        "--warmup-steps",
        type=nonnegative,
        default=20,
        help="Updates excluded from the measured window within --max-steps",
    )
    p.add_argument(
        "--csx-micro-batch-size",
        default="auto",
        help="auto or a positive divisor of the effective batch",
    )
    p.add_argument(
        "--max-steps",
        type=positive,
        default=200,
        help="Optimizer updates, on both backends",
    )
    p.add_argument("--num-workers", type=nonnegative, default=2)
    p.add_argument("--seed", type=nonnegative, default=42)
    p.add_argument("--repeat", type=positive, default=1)
    p.add_argument("--study-dir", type=Path)
    p.add_argument("--trial-timeout-sec", type=positive, default=9000)
    p.add_argument("--job-time-sec", type=positive, default=7200)
    p.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; existing directories are rejected",
    )
    p.add_argument(
        "--mount-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional CSX worker mount (repeatable)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and save; do not start training",
    )
    benchmark_launcher.add_arguments(p)
    return p


def build_config(args):
    profile = yaml.safe_load(PROFILES.read_text())[args.profile]
    base = ROOT / profile["base_config"]
    original = yaml.safe_load(base.read_text())
    init = copy.deepcopy(original["trainer"]["init"])
    loader = copy.deepcopy(original["trainer"]["fit"]["train_dataloader"])
    length = profile["sequence_length"]
    effective = args.effective_batch_size or profile["effective_batch_size"]
    micro = args.gpu_micro_batch_size or profile["gpu_micro_batch_size"]
    if args.backend == "GPU" and (micro > effective or effective % micro):
        raise ValueError("GPU micro batch must divide the effective batch size")
    csx_micro = args.csx_micro_batch_size
    if csx_micro != "auto":
        try:
            csx_micro = positive(csx_micro)
        except (ValueError, argparse.ArgumentTypeError) as exc:
            raise ValueError(
                "CSX micro batch must be auto or a positive integer"
            ) from exc
        if effective % csx_micro:
            raise ValueError("CSX micro batch must divide the effective batch size")
    if args.backend == "GPU" and args.mount_dir:
        raise ValueError("--mount-dir applies only to CSX")
    if args.warmup_steps >= args.max_steps:
        raise ValueError("--warmup-steps must be smaller than --max-steps")
    if (
        args.backend == "CSX" or args.gpu_implementation == "modelzoo"
    ) and args.warmup_steps < 1:
        raise ValueError(
            "Model Zoo timing requires --warmup-steps >= 1 for a logged start endpoint"
        )
    if args.gpu_implementation == "modelzoo" and (
        args.compile or args.gradient_checkpointing
    ):
        raise ValueError(
            "--compile and --gradient-checkpointing require the native GPU implementation"
        )

    init["backend"] = {"backend_type": args.backend}
    init["callbacks"] = []
    if args.backend == "CSX":
        # Workers also need the checkout when image building falls back to
        # mounting the client venv; the client's PYTHONPATH is not sufficient.
        init["backend"]["cluster_config"] = {
            "num_csx": 1,
            "job_time_sec": args.job_time_sec,
            "mount_dirs": list(
                dict.fromkeys(str(path.resolve()) for path in [ROOT, *args.mount_dir])
            ),
            "python_paths": [str(ROOT / "src")],
        }
        init["callbacks"] = [
            {"ScopedTrainFlags": {"csx.performance.micro_batch_size": csx_micro}}
        ]
    init["model_dir"] = str(args.output_dir)
    init["seed"] = args.seed
    init["precision"] = {
        "enabled": True,
        "fp16_type": "bfloat16",
        "loss_scaling_factor": 1.0,
        "max_gradient_norm": 1.0,
    }
    init["loop"] = {
        "max_steps": args.max_steps,
        "eval_frequency": None,
        "grad_accum_steps": effective // micro if args.backend == "GPU" else 1,
    }
    init["checkpoint"] = {
        "steps": None,
        "save_initial_checkpoint": False,
        "autoload_last_checkpoint": False,
    }
    init["logging"] = {"log_steps": 1}
    # A bounded throughput run, with the same constant LR on both backends.
    init.pop("schedulers", None)
    init["optimizer"]["AdamW"]["lr"] = profile["learning_rate"]
    # Supplied with the real model parameters by Trainer at construction time.
    init["optimizer"]["AdamW"]["params"] = []
    loader.update(
        data_dir=str(args.data_dir),
        batch_size=micro if args.backend == "GPU" else effective,
        shuffle=True,
        shuffle_seed=args.seed,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers else None,
        persistent_workers=bool(args.num_workers),
    )
    if init["model"]["name"] == "bert":
        # Keep the original 512-position BERT-Large architecture in both runs.
        init["model"]["mlm_loss_weight"] = 0.058 if length == 128 else 0.019
        loader.update(
            vocab_file=str(args.vocab_file.resolve()),
            max_sequence_length=length,
            max_predictions_per_seq=20 if length == 128 else 80,
        )
    else:
        # Preserve Llama 3.2's RoPE scaling and architecture; shorten the input.
        init["model"]["max_position_embeddings"] = length
        loader["max_sequence_length"] = (
            length if args.llama_data_format == "corpus" else None
        )
    params = {"trainer": {"init": init, "fit": {"train_dataloader": loader}}}
    metadata = {
        "profile": args.profile,
        "backend": args.backend,
        "base_config": str(base.relative_to(ROOT)),
        "base_config_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "sequence_length": length,
        "effective_batch_size": effective,
        "loader_batch_size": loader["batch_size"],
        "grad_accum_steps": init["loop"]["grad_accum_steps"],
        "nominal_tokens_per_update": effective * length,
        "max_optimizer_steps": args.max_steps,
        "precision": "bfloat16",
        "gpu_implementation": (
            args.gpu_implementation if args.backend == "GPU" else None
        ),
        "compile": args.compile,
        "gradient_checkpointing": args.gradient_checkpointing,
        "warmup_steps": args.warmup_steps,
        "repeat": args.repeat,
        "study_dir": str(args.study_dir or args.output_dir),
        "trial_timeout_sec": args.trial_timeout_sec,
        "job_time_sec": args.job_time_sec,
        "data_dir": str(args.data_dir),
        "llama_data_format": (
            args.llama_data_format if init["model"]["name"] == "llama" else None
        ),
    }
    if args.backend == "CSX":
        labels = records.job_labels(args, metadata)
        init["backend"]["cluster_config"]["job_labels"] = labels
        metadata["job_labels"] = labels
    return params, metadata


def check_data(args, metadata):
    if not args.data_dir.is_dir():
        raise ValueError(f"Training data directory not found: {args.data_dir}")
    if args.profile.startswith("bert_"):
        if not args.vocab_file.is_file():
            raise ValueError(f"BERT vocabulary not found: {args.vocab_file}")
        if not (args.data_dir / "meta.dat").is_file():
            raise ValueError(
                "BERT expects dynamic-mask CSV data with meta.dat; see README.md"
            )
    else:
        import h5py

        files = sorted(args.data_dir.glob("*.h5"))
        if not files:
            raise ValueError("Llama data directory must contain *.h5 files")
        for path in files:
            with h5py.File(path, "r") as handle:
                if "data" not in handle:
                    raise ValueError(f"Missing HDF5 'data' dataset: {path}")
                shape = handle["data"].shape
            if args.llama_data_format == "sample":
                expected = (3, metadata["sequence_length"])
                if len(shape) != 3 or shape[1:] != expected or shape[0] == 0:
                    raise ValueError(
                        f"{path}: expected (N, {expected[0]}, {expected[1]}), N > 0; got {shape}"
                    )
            elif len(shape) != 1 or shape[0] <= metadata["sequence_length"]:
                raise ValueError(
                    f"{path}: corpus expects a 1-D token stream longer than one sequence; got {shape}"
                )


def prepared_config(config, backend=None, *, evidence=None):
    """Reject changed inputs before starting a previously prepared command."""
    config = Path(config).resolve()
    launch = json.loads((config.parent / "launch.json").read_text())
    if backend is not None and backend != launch["backend"]:
        raise ValueError("Prepared configuration backend does not match --backend")
    if hashlib.sha256(config.read_bytes()).hexdigest() != launch["params_sha256"]:
        raise ValueError("params.yaml changed after preparation; prepare a new run")
    evidence = {} if evidence is None else evidence
    if launch.get("source_sha256"):
        current = records.source_identity()["source_sha256"]
        if launch["source_sha256"] != current:
            evidence["observed_source_sha256"] = current
            raise ValueError(
                "Benchmark source changed after preparation; prepare a new run"
            )
    if launch.get("dataset_identity"):
        expected = launch["dataset_identity"]
        vocabulary = expected.get("vocabulary") or {}
        observed = records.dataset_identity(
            expected["data_dir"], vocabulary.get("path")
        )
        if expected != observed:
            evidence["observed_dataset_identity"] = observed
            raise ValueError(
                "Training data or vocabulary changed after preparation; prepare a new run"
            )
    return config, launch


def execute_prepared(
    config,
    *,
    backend=None,
    timeout_sec=None,
    preflight_error=None,
    preflight_exit_code=2,
):
    """Execute exactly one prepared client and retain partial and final evidence."""
    import measurements

    config = Path(config).resolve()
    output = config.parent
    if (output / "client_status.json").exists() or (output / "console.log").exists():
        raise ValueError(
            "This prepared run already has execution artifacts; prepare a new run"
        )
    launch = json.loads((output / "launch.json").read_text())
    # Claim execution before writing metadata. Losing a concurrent claim must
    # leave the first client's status and logs untouched.
    log = (output / "console.log").open("xb", buffering=0)
    status = {
        "state": "client_running",
        "phase": "preflight",
        "started_at": benchmark_launcher.timestamp(),
    }
    status_path = output / "client_status.json"
    process = None
    interrupted = None
    stop_deadline = None
    timed_out = False
    start = time.monotonic()
    client_start = None

    def stop_client():
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def interrupt(number, _frame):
        nonlocal interrupted, stop_deadline
        if interrupted is None:
            interrupted = number
            stop_deadline = time.monotonic() + 10
            stop_client()

    handlers = {
        number: signal.signal(number, interrupt)
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    code = preflight_exit_code if preflight_error else 2
    try:
        with log:
            benchmark_launcher.write_json(status_path, status)
            measurements.refresh_campaign(launch)
            if preflight_error:
                raise ValueError(preflight_error)
            config, launch = prepared_config(config, backend, evidence=status)
            check_runtime(launch["backend"])
            timeout_sec = (
                launch.get("trial_timeout_sec", 9000)
                if timeout_sec is None
                else timeout_sec
            )
            if timeout_sec <= 0:
                raise ValueError("Client timeout must be positive")
            benchmark_launcher.write_json(
                output / "environment.json", records.environment()
            )
            if interrupted is not None:
                raise InterruptedError("Execution cancelled before client startup")
            status["phase"] = "training"
            code = 1
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            )
            client_start = time.monotonic()
            status["client_started_at"] = benchmark_launcher.timestamp()
            status["preflight_seconds"] = client_start - start
            process = subprocess.Popen(
                launch["command"],
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status["client_pid"] = process.pid
            benchmark_launcher.write_json(status_path, status)
            if interrupted is not None:
                stop_client()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                drain_deadline = None
                while selector.get_map() or process.poll() is None:
                    now = time.monotonic()
                    if (
                        not timed_out
                        and interrupted is None
                        and now - client_start >= timeout_sec
                    ):
                        timed_out = True
                        stop_deadline = now + 10
                        stop_client()
                    if stop_deadline is not None and now >= stop_deadline:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        stop_deadline = None
                    if process.poll() is not None:
                        if drain_deadline is None:
                            drain_deadline = now + 1
                        elif now >= drain_deadline:
                            # Do not let an inherited stdout descriptor hold the
                            # campaign forever after the client has exited.
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            break
                    for key, _ in selector.select(timeout=0.1):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if chunk:
                            log.write(chunk)
                            try:
                                print(
                                    chunk.decode(errors="replace"), end="", flush=True
                                )
                            except (BrokenPipeError, OSError):
                                pass
                        else:
                            selector.unregister(key.fileobj)
            code = process.wait()
            if code < 0:
                code = 128 - code
    except (OSError, ValueError, KeyError, TypeError) as exc:
        status["error"] = str(exc)
    finally:
        if process is not None:
            if process.poll() is None:
                stop_client()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            # A successful client can leave input workers behind after closing
            # stdout. The private group belongs entirely to this one trial.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.stdout.close()
        for number, handler in handlers.items():
            signal.signal(number, handler)
        if interrupted is not None:
            state, code = "interrupted", 128 + interrupted
        elif timed_out:
            state, code = "timeout", 124
        else:
            state = "completed" if code == 0 else "failed"
        status.update(
            state=state,
            exit_code=code,
            finished_at=benchmark_launcher.timestamp(),
            wall_seconds=time.monotonic() - start,
            client_wall_seconds=time.monotonic() - client_start
            if process is not None
            else None,
        )
        if process is None:
            status["preflight_seconds"] = status["wall_seconds"]
            status.pop("client_started_at", None)
        benchmark_launcher.write_json(status_path, status)
    result = measurements.collect_run(output, launch, status)
    status["measurement_status"] = result["measurement_status"]
    benchmark_launcher.write_json(status_path, status)
    print(
        f"[result] {output}: {status['state']}; measurement={status['measurement_status']}",
        flush=True,
    )
    return status


def execution_exit_code(status):
    return status["exit_code"] or (0 if status["measurement_status"] == "valid" else 2)


def detach_driver(args, argv, script, name):
    """Keep launcher metadata beside a benchmark's required-new directory."""
    return benchmark_launcher.launch(
        [
            sys.executable,
            "-u",
            str(script),
            *argv,
            "--foreground",
            "--output-dir",
            str(args.output_dir),
        ],
        Path(str(args.output_dir) + ".launcher"),
        name=name,
        session=args.tmux_session,
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    if args.list_configs:
        print(PROFILES.read_text(), end="")
        return 0
    if args.check_dependencies:
        try:
            check_dependencies()
        except ValueError as exc:
            p.error(str(exc))
        print(f"Non-GNN dependencies import successfully in {sys.executable}")
        return 0
    if args.execute_config is not None:
        try:
            if args.prepare_only:
                raise ValueError(
                    "--execute-config cannot be combined with --prepare-only"
                )
            if args.dry_run:
                config, launch = prepared_config(args.execute_config, args.backend)
                print(shlex.join(launch["command"]))
                return 0
            if args.detach:
                config, _ = prepared_config(args.execute_config, args.backend)
                check_runtime(args.backend)
                check_dependencies()
                args.output_dir = config.parent
                return detach_driver(args, argv, Path(__file__).resolve(), "non-gnn")
            return execution_exit_code(
                execute_prepared(
                    args.execute_config,
                    backend=args.backend,
                    preflight_error=args.preflight_error,
                    preflight_exit_code=args.preflight_exit_code,
                )
            )
        except (ValueError, OSError) as exc:
            p.error(str(exc))
    if args.profile is None or args.data_dir is None:
        p.error("--profile and --data-dir are required")
    args.data_dir = args.data_dir.resolve()
    args.vocab_file = args.vocab_file.resolve()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.output_dir = (
        args.output_dir
        or ROOT
        / "model_dirs/non_gnn"
        / f"{args.profile}_{args.backend.lower()}_{stamp}_{os.getpid()}"
    ).resolve()
    if args.study_dir is not None:
        args.study_dir = args.study_dir.resolve()
    try:
        records.validate_output_dir(args.output_dir)
        params, metadata = build_config(args)
        config = args.output_dir / "params.yaml"
        command = [
            sys.executable,
            "-m",
            "cerebras.modelzoo.cli.main",
            "fit",
            str(config),
            "--target_device",
            args.backend,
            "--model_dir",
            str(args.output_dir),
        ]
        if args.backend == "GPU" and args.gpu_implementation == "native":
            command = [
                sys.executable,
                str(ROOT / "benchmark_scripts/non_gnn/gpu/train.py"),
                "--config",
                str(config),
            ]
        if args.dry_run:
            print(yaml.safe_dump(params, sort_keys=False), end="")
            print(
                f"# effective batch={metadata['effective_batch_size']}; optimizer steps={args.max_steps}"
            )
            print(f"# {shlex.join(command)}")
            return 0
        if args.backend == "GPU" and not args.prepare_only:
            raise ValueError("Use the Pegasus submit script for GPU execution")
        if args.output_dir.exists():
            raise ValueError(
                f"Output directory already exists: {args.output_dir}; choose a new directory"
            )
        check_runtime(args.backend)
        check_dependencies()
        check_data(args, metadata)
        from cerebras.modelzoo.trainer.validate import validate_trainer_params

        validate_trainer_params(copy.deepcopy(params))
        if args.detach:
            return detach_driver(args, argv, Path(__file__).resolve(), "non-gnn")
        metadata.update(records.source_identity())
        metadata["dataset_identity"] = records.dataset_identity(
            args.data_dir, args.vocab_file if args.profile.startswith("bert_") else None
        )
        metadata["created_at"] = benchmark_launcher.timestamp()
        metadata["command"] = command
        metadata["arguments"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in ("mount_dir", "detach", "tmux_session")
        }
        metadata["arguments"]["mount_dir"] = [str(path) for path in args.mount_dir]
        text = yaml.safe_dump(params, sort_keys=False)
        metadata["params_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        args.output_dir.mkdir(parents=True, exist_ok=False)
        config.write_text(text)
        benchmark_launcher.write_json(args.output_dir / "launch.json", metadata)
        print(f"Prepared {config}", flush=True)
        print(shlex.join(command), flush=True)
        if args.prepare_only:
            return 0
        return execution_exit_code(execute_prepared(config, backend=args.backend))
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
