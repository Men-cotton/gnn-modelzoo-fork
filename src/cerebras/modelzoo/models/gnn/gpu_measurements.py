"""User-accessible metadata and optional fixed-shape input measurements."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

import torch


def logical_tensor_bytes(value):
    """Logical payload, including padding; not bus traffic or resident bytes."""
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(logical_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(logical_tensor_bytes(item) for item in value)
    return 0


def save_provenance(output_dir, *, optimizer, device):
    """Preserve source identity and effective settings before any timed work."""
    root = Path(__file__).resolve().parents[5]
    result = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "executable": sys.executable,
        "argv": sys.argv,
        "source_root": str(root),
        "packages": sorted(
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
        ),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PBS_JOBID",
                "NO_COMPILE",
            )
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "torch_threads": torch.get_num_threads(),
        "optimizer_defaults": optimizer.defaults,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["cuda_device"] = dict(
            name=properties.name,
            total_memory=properties.total_memory,
            capability=[properties.major, properties.minor],
        )
    try:

        def git(*args):
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.PIPE
            )

        result["git_commit"] = git("rev-parse", "HEAD").strip()
        result["git_status"] = git("status", "--short")
        diff = git(
            "diff",
            "--no-ext-diff",
            "HEAD",
            "--",
            "src",
            "benchmark_scripts",
            "pyproject.toml",
            "uv.lock",
        )
        (output_dir / "source_changes.patch").write_text(diff)
        result["source_changes_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
    except (OSError, subprocess.CalledProcessError) as exc:
        result["git_error"] = str(exc)
    # Include untracked runtime files: a tracked diff alone cannot identify them.
    result["gnn_source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.rglob("*.py"))
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
