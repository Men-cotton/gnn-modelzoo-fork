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
import shlex
import shutil
import subprocess
import sys

import yaml

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
        help="Native GPU timing warmup within --max-steps",
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
    if (
        args.backend == "GPU"
        and args.gpu_implementation == "native"
        and args.warmup_steps >= args.max_steps
    ):
        raise ValueError("--warmup-steps must be smaller than --max-steps")
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
        "data_dir": str(args.data_dir),
        "llama_data_format": (
            args.llama_data_format if init["model"]["name"] == "llama" else None
        ),
    }
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


def main(argv=None):
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
    try:
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
        metadata["git_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        metadata["git_status"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--short"], text=True
        ).strip()
        metadata["created_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        metadata["command"] = command
        metadata["arguments"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "mount_dir"
        }
        metadata["arguments"]["mount_dir"] = [str(path) for path in args.mount_dir]
        text = yaml.safe_dump(params, sort_keys=False)
        metadata["params_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        args.output_dir.mkdir(parents=True, exist_ok=False)
        config.write_text(text)
        (args.output_dir / "launch.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(f"Prepared {config}", flush=True)
        print(shlex.join(command), flush=True)
        if args.prepare_only:
            return 0
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        with (args.output_dir / "console.log").open("w") as log:
            with subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return proc.wait()
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
