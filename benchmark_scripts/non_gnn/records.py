"""Source, input and execution provenance for one non-GNN trial."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from cerebras.modelzoo.tools.benchmark_labels import label_value, study_label

ROOT = Path(__file__).resolve().parents[2]


def validate_output_dir(path):
    """Keep generated configurations outside the source identity's scope."""
    path = Path(path).resolve()
    for source in (ROOT / "src", ROOT / "benchmark_scripts"):
        if path == source or source in path.parents:
            raise ValueError(
                "Benchmark output must be outside src/ and benchmark_scripts/; "
                "use model_dirs/ or a separate data directory"
            )


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def source_identity():
    """Hash executable/configuration sources, including uncommitted content."""
    suffixes = {".py", ".sh", ".pbs", ".yaml", ".yml", ".toml"}
    paths = [
        path
        for directory in (ROOT / "src", ROOT / "benchmark_scripts")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in suffixes
    ]
    paths += [
        ROOT / name
        for name in ("pyproject.toml", "uv.lock", "setup.py", "common.sh")
        if (ROOT / name).is_file()
    ]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}
    return {
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--short"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "source_hash_scope": "src and benchmark_scripts Python/shell/PBS/YAML/TOML; root pyproject.toml, uv.lock, setup.py and common.sh",
    }


def environment():
    """Inspect the current execution host without serializing credentials."""
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "packages": sorted(
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        ),
        "runtime_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PBS_JOBID",
            )
        },
    }
    try:
        result["meminfo"] = Path("/proc/meminfo").read_text()
    except OSError:
        result["meminfo"] = None
    return result


def dataset_identity(data_dir, vocab_file=None):
    """Record data content hashes and any preparation manifest independently."""
    data_dir = Path(data_dir).resolve()
    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and (path.suffix in {".csv", ".h5", ".hdf5"} or path.name == "meta.dat")
    )
    files = {
        str(path.relative_to(data_dir)): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    }
    result = {
        "data_dir": str(data_dir),
        "files": files,
        "content_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "manifest": None,
        "vocabulary": None,
    }
    for manifest in (data_dir / "manifest.json", data_dir.parent / "manifest.json"):
        if manifest.is_file():
            value = json.loads(manifest.read_text())
            result["manifest"] = {
                "path": str(manifest),
                "sha256": sha256(manifest),
                "request": value.get("request"),
                "stats": value.get("stats"),
            }
            break
    if vocab_file is not None:
        path = Path(vocab_file).resolve()
        result["vocabulary"] = {"path": str(path), "sha256": sha256(path)}
    return result


def job_labels(args, metadata):
    """Describe the prepared trial using the same SDK-safe labels as GNN jobs."""
    output = Path(args.output_dir).resolve()
    study = Path(getattr(args, "study_dir", None) or output).resolve()
    profile = (
        args.profile.replace("llama3p2", "llama3.2")
        .replace("_msl", "-s")
        .replace("_", "-")
    )
    run = (
        f"{profile}-b{metadata['effective_batch_size']}-w{args.num_workers}"
        f"-r{getattr(args, 'repeat', 1)}"
    )
    return [
        f"run={label_value(run, max_length=60)}",
        f"study={study_label(study, compact=True)}",
    ]
