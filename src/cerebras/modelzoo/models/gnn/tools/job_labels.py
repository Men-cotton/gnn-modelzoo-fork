"""Stable CSX job metadata for worker studies and their individual trials."""

from __future__ import annotations

from pathlib import Path
import re

from cerebras.modelzoo.tools.benchmark_labels import label_value, study_label

FIELDS = ("model", "dataset", "cache", "mode", "workers", "repeat", "trial", "study")
MANAGED_KEYS = frozenset({"run", "study", *(f"gnn-{field}" for field in FIELDS)})
MODE_NAMES = {"sensitivity": "sens", "diagnostic": "diag", "autotune": "tune"}


def trial_stage(trial: Path, mode: str, loader: dict, repeat: int) -> str:
    """Describe known trial roles without repeating their encoded input knobs.

    Keep unrecognized paths intact for label_value's collision-safe fallback.
    In particular, dropping parents would merge repeated reference trials in
    different worker comparisons, or baseline and selected learning runs.
    """
    parts = trial.parts
    name = MODE_NAMES.get(mode, mode)
    candidate = re.fullmatch(
        r"(sensitivity|workers|loader|confirm)_w(\d+)_p(\d+)_s([01])_r(\d+)",
        trial.name,
    )
    if candidate:
        phase = candidate[1]
        expected = (
            f"{phase}_w{loader['num_workers']:02d}"
            f"_p{loader.get('prefetch_factor') or 0}"
            f"_s{int(bool(loader.get('persistent_workers', False)))}_r{repeat}"
        )
        if trial.name == expected and mode in {"sensitivity", "autotune"}:
            phase = MODE_NAMES.get(phase, phase)
            if len(parts) == 1:
                return phase
            if len(parts) == 2 and parts[0] == "tuning" and mode == "autotune":
                return f"tune-{phase}"
            comparison = re.fullmatch(r"workers_w(\d+)", parts[0])
            if (
                len(parts) == 2
                and comparison
                and parts[0] == f"workers_w{int(comparison[1]):02d}"
                and mode == "sensitivity"
            ):
                return f"vs{int(comparison[1])}-{phase}"
    if mode == "learning":
        if parts in {("baseline",), ("selected",)}:
            return f"learning-{trial.name}"
        if parts == ("handoff", "selected_learning"):
            return "handoff-learning"
    if mode == "selected" and parts == ("handoff", "selected_csx"):
        return "handoff-selected"
    if mode == "diagnostic":
        if parts == (f"diagnostic_w{loader['num_workers']:02d}",):
            return "diag"
        if parts == ("handoff", "diagnostic"):
            return "handoff-diag"
    if mode == "intervention" and len(parts) == 1:
        controls = {
            "prefetch1": "control-prefetch",
            "persistent_off": "control-persistent",
            "feature_cache": "control-cache",
            "static_batch": "control-static",
        }
        for control, description in controls.items():
            if trial.name == f"control_{control}_r{repeat}":
                return description
    return f"{name}-trial-{trial}"


def apply_job_labels(
    config: dict,
    *,
    mode: str,
    repeat: int,
    trial_dir: Path,
    study_dir: Path,
) -> None:
    """Write a readable run description and compact study identifier.

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
    trial_path, study_path = Path(trial_dir).resolve(), Path(study_dir).resolve()
    try:
        trial = trial_path.relative_to(study_path)
    except ValueError:
        trial = trial_path
    model_name = model.get("architecture", {}).get("type", model.get("name", "model"))
    dataset_name = profile.get("dataset_name", loader.get("dataset_name", dataset))
    parts = [
        {"graphsage": "sage"}.get(model_name, model_name),
        {"ogbn-arxiv": "arxiv", "ogbn-products": "products"}.get(
            dataset_name, dataset_name
        ),
        trial_stage(trial, mode, loader, repeat),
        f"w{loader['num_workers']}",
        f"pf{loader.get('prefetch_factor') or 0}",
        "persist" if loader.get("persistent_workers", False) else "nopersist",
    ]
    fraction = loader.get("cache_fraction")
    if fraction is not None:
        parts.append(f"cache{int(fraction) if fraction in (0, 1) else fraction}")
    if loader.get("static_batch_cache_size", 0):
        parts.append(f"static{loader['static_batch_cache_size']}")
    parts.append(f"r{repeat}")
    cluster = init["backend"].setdefault("cluster_config", {})
    retained = [
        label
        for label in cluster.get("job_labels") or []
        if label.split("=", 1)[0] not in MANAGED_KEYS
    ]
    cluster["job_labels"] = retained + [
        f"run={label_value('-'.join(parts), max_length=60)}",
        f"study={study_label(study_path, compact=True)}",
    ]
