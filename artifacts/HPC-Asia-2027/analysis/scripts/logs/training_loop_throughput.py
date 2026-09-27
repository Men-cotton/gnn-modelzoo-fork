"""Compare independently repeated training windows using verified seed counts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys

import yaml


def active_path(path: Path) -> Path:
    if "invalid" in path.parts:
        raise ValueError("Input is outside the active measurement set")
    path = path.resolve()
    if "invalid" in path.parts:
        raise ValueError("Input is outside the active measurement set")
    return path


def read_json(path: Path):
    return json.loads(active_path(path).read_text())


def config_identity(config):
    """Compare learning semantics while permitting platform-specific input tuning."""
    init = config["trainer"]["init"]
    loader = config["trainer"]["fit"]["train_dataloader"]
    model = deepcopy(init["model"])
    model.get("task", {}).pop("compute_eval_metrics", None)
    profiles = loader.get("dataset_profiles") or {}
    dataset = loader.get("dataset")
    profile = profiles.get(str(dataset).lower()) or profiles.get(dataset) or {}
    loader = {**profile, **loader}
    optimizer = init["optimizer"]
    if set(optimizer) != {"AdamW"} or any(
        key not in optimizer["AdamW"] for key in ("eps", "betas")
    ):
        raise ValueError("Matched training requires explicit AdamW eps and betas")
    return dict(
        dataset=loader.get("dataset_name") or dataset,
        model=model,
        optimizer=optimizer,
        precision=init.get("precision"),
        seed=init.get("seed"),
        batch_size=loader["batch_size"],
        fanouts=loader["fanouts"],
        sampler_seed=loader.get("sampler_seed", 0),
        drop_last=loader.get("drop_last_batch", False),
        shuffle=loader.get("shuffle", False),
    )


def check_native_runtime(log: Path, config: dict) -> dict:
    records = []
    for line in log.read_text().splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            if isinstance(row, dict) and row.get("event") == "run":
                records.append(row)
    if len(records) != 1 or not str(records[0].get("device", "")).startswith("cuda"):
        raise ValueError("GPU comparison requires one native CUDA execution")
    run = records[0]
    init = config["trainer"]["init"]
    precision = init.get("precision")
    if not isinstance(precision, dict):
        raise ValueError("GPU comparison requires explicit precision configuration")
    dtype = (
        "torch.float32"
        if not precision.get("enabled", True)
        else {"float16": "torch.float16", "bfloat16": "torch.bfloat16"}.get(
            precision.get("fp16_type", "float16")
        )
    )
    if run.get("precision") != dtype or run.get("compile") != bool(
        init.get("benchmark", {}).get("compile", False)
    ):
        raise ValueError(
            "Native precision or compilation differs from its recorded configuration"
        )
    return {
        key: run.get(key)
        for key in (
            "device_name",
            "precision",
            "compile",
            "measure_neighbor_padding",
            "measure_input",
        )
    }


def summarize_study(directory: Path, parsers: dict) -> dict:
    directory = active_path(directory)
    state = read_json(directory / "study.json")
    settings = state["settings"]
    if settings["mode"] != "sensitivity" or len(settings["workers"]) != 1:
        raise ValueError(
            "Final comparison requires one selected setting and fresh sensitivity repetitions"
        )
    backend = settings["backend"]
    if backend not in parsers:
        raise ValueError("Unsupported comparison backend")
    repeats = settings["repeats"]
    trials = state["trials"]
    if (
        repeats < 3
        or len(trials) != repeats
        or sorted(t["repeat"] for t in trials) != list(range(1, repeats + 1))
    ):
        raise ValueError("Require every planned independent repetition, at least three")
    if state["status"] != "completed" or any(
        t["status"] not in {"completed", "unstable"} for t in trials
    ):
        raise ValueError(
            "Every comparison repetition must have a completed finite measurement"
        )
    if len({trial["trial_id"] for trial in trials}) != repeats:
        raise ValueError(
            "Each independent repetition requires a distinct trial directory"
        )
    start = settings["warmup_steps"]
    end = start + settings["measure_steps"]
    measurements, identities, input_ids, loader_settings, native_settings = (
        [],
        [],
        [],
        [],
        [],
    )
    for trial in trials:
        if (
            not isinstance(trial["trial_id"], str)
            or Path(trial["trial_id"]).name != trial["trial_id"]
            or trial["trial_id"] in (".", "..")
        ):
            raise ValueError("Trial directories must be direct children of the study")
        folder = active_path(directory / trial["trial_id"])
        if folder.parent != directory:
            raise ValueError("Trial directory resolves outside its study")
        config_path, log_path = (
            active_path(folder / name) for name in ("params.yaml", "train.log")
        )
        if config_path.parent != folder or log_path.parent != folder:
            raise ValueError("Trial files resolve outside their trial directory")
        config = yaml.safe_load(config_path.read_text())
        # Match gnn.tools.autotune.digest's JSON serialization exactly.
        encoded = json.dumps(config, sort_keys=True, allow_nan=False).encode()
        if hashlib.sha256(encoded).hexdigest() != trial["config_sha256"]:
            raise ValueError("Trial configuration differs from recorded execution")
        identities.append(config_identity(config))
        loader = config["trainer"]["fit"]["train_dataloader"]
        knobs = {
            name: loader.get(name)
            for name in (
                "num_workers",
                "prefetch_factor",
                "persistent_workers",
                "pin_memory",
                "cache_fraction",
                "worker_diagnostics",
            )
        }
        if isinstance(knobs["worker_diagnostics"], dict):
            knobs["worker_diagnostics"] = dict(knobs["worker_diagnostics"])
            knobs["worker_diagnostics"].pop("output_dir", None)
        if knobs["num_workers"] != settings["workers"][0] or any(
            loader.get(key) != value for key, value in trial["knobs"].items()
        ):
            raise ValueError(
                "Trial input configuration differs from its selected settings"
            )
        loader_settings.append(knobs)
        result = parsers[backend].summarize(
            log_path, start, end, tolerance=settings["stability_tolerance_percent"]
        )
        if backend == "fixed_shape":
            native_settings.append(check_native_runtime(log_path, config))
        if result != trial.get("measurement"):
            raise ValueError(
                "Recomputed measurement differs from its recorded numerator, window or accounting metadata"
            )
        if result.get("skipped_optimizer_steps", 0):
            raise ValueError("Comparison contains skipped optimizer updates")
        if result.get("metric") != "seed_nodes_per_second":
            raise ValueError("Comparison requires seed_nodes_per_second")
        rate = result.get("seed_nodes_per_second", result.get("throughput"))
        if rate is None or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Seed throughput must be finite and positive")
        contract = result.get("input_contract") or result.get(
            "numerator_provenance", {}
        )
        if isinstance(contract, dict) and "input_contract" in contract:
            contract = contract["input_contract"]
        if (
            isinstance(contract, dict)
            and "ordered_targets_and_labels_sha256" in contract
        ):
            for field, condition in (
                ("dataset_name", "dataset"),
                ("batch_size", "batch_size"),
                ("sampler_seed", "sampler_seed"),
                ("drop_last", "drop_last"),
                ("shuffle", "shuffle"),
            ):
                if contract.get(field) != identities[-1][condition]:
                    raise ValueError(
                        f"Runtime input contract differs from configuration: {field}"
                    )
            input_ids.append(contract["ordered_targets_and_labels_sha256"])
        measurements.append(
            dict(
                repeat=trial["repeat"],
                throughput=rate,
                measurement=result,
                unstable=not (result.get("half_window_check") or {}).get(
                    "within_tolerance", False
                ),
            )
        )
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Learning conditions changed between repetitions")
    if any(knobs != loader_settings[0] for knobs in loader_settings[1:]):
        raise ValueError("Input settings changed between repetitions")
    if any(runtime != native_settings[0] for runtime in native_settings[1:]):
        raise ValueError("Native GPU execution settings changed between repetitions")
    if backend in {"csx", "fixed_shape"} and (
        len(input_ids) != repeats or len(set(input_ids)) != 1
    ):
        raise ValueError("Input identity must agree in every fixed-shape repetition")
    rates = [row["throughput"] for row in measurements]
    source = state["environment"].get("source_sha256")
    if not source:
        raise ValueError("Missing source identity")
    return dict(
        study=str(directory),
        backend=backend,
        metric="seed_nodes_per_second",
        start_step=start,
        end_step=end,
        n=len(rates),
        mean=statistics.mean(rates),
        sample_stddev=statistics.stdev(rates),
        minimum=min(rates),
        maximum=max(rates),
        source_sha256=source,
        input_sha256=input_ids[0] if input_ids else None,
        input_identity_scope="ordered training target IDs and labels; feature and edge contents require separate verification"
        if input_ids
        else "sampling representation specific to PyG",
        learning_conditions=identities[0],
        settings=settings,
        runs=measurements,
        environment=state["environment"],
        native_runtime=native_settings[0] if native_settings else None,
    )


def compare(studies: list[dict]) -> dict:
    if not studies:
        raise ValueError("Supply at least one study")
    if len({study["study"] for study in studies}) != len(studies):
        raise ValueError("Supply each independent study only once")
    baseline = studies[0]
    for study in studies[1:]:
        for field in (
            "metric",
            "start_step",
            "end_step",
            "source_sha256",
            "learning_conditions",
        ):
            if study[field] != baseline[field]:
                raise ValueError(f"Comparison conditions differ: {field}")
    fixed = [study for study in studies if study["backend"] in {"csx", "fixed_shape"}]
    if len({study["input_sha256"] for study in fixed}) > 1:
        raise ValueError("Fixed-shape input targets/labels differ across systems")
    csx = [study for study in studies if study["backend"] == "csx"]
    ratios = []
    if len(csx) > 1:
        raise ValueError("Compare one CSX condition at a time")
    if csx:
        for study in studies:
            if study["backend"] != "csx":
                ratios.append(
                    dict(
                        gpu_backend=study["backend"],
                        csx_mean_over_gpu_mean=csx[0]["mean"] / study["mean"],
                        comparison_scope="shared_fixed_shape_implementation"
                        if study["backend"] == "fixed_shape"
                        else "separate_pyg_implementation",
                    )
                )
    return dict(
        metric="seed_nodes_per_second",
        studies=studies,
        ratios=ratios,
        aggregation="Equal-weight independent runs; sample standard deviation uses ddof=1; unstable complete runs retained and flagged",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path, nargs="+")
    parser.add_argument(
        "--modelzoo-root",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "gnn-modelzoo",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.modelzoo_root.resolve() / "src"))
    parsers = {
        backend: importlib.import_module("cerebras.modelzoo.models.gnn.tools." + name)
        for backend, name in (
            ("csx", "measure_window"),
            ("fixed_shape", "measure_fixed_shape"),
            ("pyg", "measure_pyg"),
        )
    }
    try:
        result = compare([summarize_study(path, parsers) for path in args.studies])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
