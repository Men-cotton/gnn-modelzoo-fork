"""Stable CSX job metadata for worker studies and their individual trials."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

FIELDS = ("model", "dataset", "cache", "mode", "workers", "repeat", "trial", "study")
MANAGED_KEYS = frozenset(f"gnn-{field}" for field in FIELDS)


def label_value(value: object) -> str:
    """Meet the SDK's stricter, nonempty Kubernetes label-value subset."""
    original = str(value)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", original).strip("-_.") or "value"
    if cleaned != original or len(cleaned) > 63:
        suffix = hashlib.sha256(original.encode()).hexdigest()[:10]
        cleaned = f"{cleaned[:52].rstrip('-_.')}-{suffix}"
    return cleaned


def study_label(study_dir: Path) -> str:
    """Distinguish identically named studies in different output directories."""
    path = Path(study_dir).resolve()
    suffix = hashlib.sha256(str(path).encode()).hexdigest()[:10]
    return label_value(f"{path.name or 'study'}-{suffix}")


def apply_job_labels(
    config: dict,
    *,
    mode: str,
    repeat: int,
    trial_dir: Path,
    study_dir: Path,
) -> None:
    """Update our eight keys in place while preserving all other user labels.

    The SDK accepts a list of ``key=value`` strings, with each token 1–63
    characters, alphanumeric ends and only ``[-A-Za-z0-9_.]`` inside. In
    particular, it does not accept Kubernetes' optional slash-prefixed keys.
    Existing user labels are left unchanged for the SDK to validate normally.
    """
    trainer = config["trainer"]
    init = trainer["init"]
    loader = trainer["fit"]["train_dataloader"]
    model = init["model"]
    dataset = loader.get("dataset", "unknown")
    profile = loader.get("dataset_profiles", {}).get(dataset, {})
    fraction = loader.get("cache_fraction")
    if fraction is None:
        cache = "none"
    elif fraction == 0:
        cache = "zero"
    elif fraction == 1:
        cache = "full"
    else:
        cache = f"partial-{fraction}"
    trial_path, study_path = Path(trial_dir).resolve(), Path(study_dir).resolve()
    try:
        trial = str(trial_path.relative_to(study_path))
    except ValueError:
        trial = study_label(trial_path)
    values = dict(
        model=model.get("architecture", {}).get("type", model.get("name", "gnn")),
        dataset=profile.get("dataset_name", loader.get("dataset_name", dataset)),
        cache=cache,
        mode=mode,
        workers=loader["num_workers"],
        repeat=repeat,
        trial=trial,
        study=study_label(study_path),
    )
    cluster = init["backend"].setdefault("cluster_config", {})
    retained = [
        label
        for label in cluster.get("job_labels") or []
        if label.split("=", 1)[0] not in MANAGED_KEYS
    ]
    cluster["job_labels"] = retained + [
        f"gnn-{field}={label_value(values[field])}" for field in FIELDS
    ]
