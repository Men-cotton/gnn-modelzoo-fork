"""Measure one completed CSX training run between exact logged step endpoints.

Invoke with uv run --no-sync tools/measure_window.py LOG.
The output reports seeds, supervised targets and nominal slots over one window.
No compilation or pre-training durations are collected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Sequence

PROGRESS = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*"
    r"\| Train Device=CSX, Step=(\d+), Loss=([^,]+),"
)


@dataclass(frozen=True)
class Point:
    step: int
    timestamp: datetime


def read_points(path: Path) -> list[Point]:
    """Read one log; reject mixed runs, nonfinite loss, and incomplete runs."""
    if "invalid" in path.resolve().parts:
        raise ValueError("Input is outside the active measurement set")
    points: list[Point] = []
    completions = 0
    for line in path.read_text().splitlines():
        if "Training completed successfully!" in line:
            completions += 1
        match = PROGRESS.search(line)
        if not match:
            continue
        if not math.isfinite(float(match[3])):
            raise ValueError(f"Nonfinite loss at step {match[2]}")
        point = Point(
            int(match[2]), datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f")
        )
        if points and (
            point.step <= points[-1].step or point.timestamp <= points[-1].timestamp
        ):
            raise ValueError(
                "Steps/timestamps must increase strictly; use one fresh log per trial"
            )
        points.append(point)
    if completions != 1:
        raise ValueError("Expected exactly one successfully completed training run")
    if len(points) < 2:
        raise ValueError("Not enough CSX progress records")
    return points


def measure(points: Sequence[Point], start: int, end: int, batch_size: int) -> dict:
    """Use t(end)-t(start): completed steps start+1 through end are counted."""
    if start < 0 or end <= start or batch_size <= 0:
        raise ValueError("Require 0 <= start < end and a positive batch size")
    by_step = {point.step: point for point in points}
    if start not in by_step or end not in by_step:
        raise ValueError(
            f"Exact step {start} and {end} timestamps are required; no interpolation is used"
        )
    seconds = (by_step[end].timestamp - by_step[start].timestamp).total_seconds()
    if seconds <= 0:
        raise ValueError("Nonpositive measurement duration")
    return {
        "start_step": start,
        "end_step": end,
        "measured_steps": end - start,
        "nominal_batch_size": batch_size,
        "training_window_seconds": seconds,
        "nominal_slots_per_second": batch_size * (end - start) / seconds,
    }


def _check_training_only_window(path: Path, start: int, end: int) -> None:
    """Reject logged evaluation/checkpoint work between the selected endpoints."""
    activity = re.compile(
        r"\b(?:Eval(?:uation)?|Validat(?:e|ion))\b"
        r"|\b(?:Saving|Saved) (?:a )?checkpoint\b"
        r"|\bCheckpoint (?:saved|saving)\b",
        re.IGNORECASE,
    )
    inside = False
    for line in path.read_text().splitlines():
        match = PROGRESS.search(line)
        if match:
            step = int(match[2])
            if step == end:
                break
            if step == start:
                inside = True
        if inside and activity.search(line):
            raise ValueError(
                "Evaluation or checkpoint activity overlaps the training window: "
                + line.strip()
            )


def read_input_contract(path: Path) -> dict:
    """Recover the sampler's actual batch schedule, without guessing OGB sizes."""
    if "invalid" in path.resolve().parts:
        raise ValueError("Input is outside the active measurement set")
    contracts = []
    for line in path.read_text().splitlines():
        if "GNN_INPUT_CONTRACT " not in line:
            continue
        contract = json.loads(line.split("GNN_INPUT_CONTRACT ", 1)[1])
        if not isinstance(contract, dict):
            raise ValueError("Input contract must be an object")
        if contract.get("split") == "train" and contract not in contracts:
            contracts.append(contract)
    if not contracts:
        raise ValueError("Missing GNN_INPUT_CONTRACT for exact seed accounting")
    if len(contracts) != 1:
        raise ValueError("Different train input contracts were logged in one run")
    return validate_input_contract(contracts[0])


def validate_input_contract(contract: dict) -> dict:
    """Validate the shared emitted-payload accounting schema."""
    if not isinstance(contract, dict):
        raise ValueError("Input contract must be an object")
    if (
        type(contract.get("version")) is not int
        or contract["version"] not in (1, 2)
        or type(contract.get("num_streamers")) is not int
        or contract["num_streamers"] != 1
        or type(contract.get("batch_index_origin")) is not int
        or contract["batch_index_origin"] != 0
        or contract.get("traversal") != "continuous_sequential_batches"
        or contract.get("event") != "gnn_input_contract"
        or contract.get("traversal_scope") != "single_data_executor"
        or contract.get("restartable") is not False
        or not isinstance(contract.get("ordered_targets_and_labels_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", contract["ordered_targets_and_labels_sha256"]
        )
    ):
        raise ValueError("Unsupported input contract traversal or streamer count")
    if contract["version"] == 2:
        if (
            contract.get("target_order") not in ("fixed", "reshuffle_each_epoch")
            or contract.get("ordered_targets_and_labels_epoch") != 0
            or contract.get("supervised_targets_by_batch_scope")
            not in ("all_epochs", "first_epoch")
        ):
            raise ValueError("Unsupported input contract target order or count scope")
    size = contract.get("batch_size")
    seeds = contract.get("seed_nodes_by_batch")
    supervised = contract.get("supervised_targets_by_batch")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or not isinstance(seeds, list)
        or not seeds
        or not isinstance(supervised, list)
        or len(seeds) != len(supervised)
    ):
        raise ValueError("Invalid input contract batch sizes/counts")
    for seed, target in zip(seeds, supervised):
        if (
            any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (seed, target)
            )
            or not 0 <= target <= seed <= size
        ):
            raise ValueError(
                "Input contract requires 0 <= supervised targets <= seeds <= slots"
            )
    if not any(seeds):
        raise ValueError("Input contract has no real seeds")
    return contract


def _check_fresh_continuous_run(path: Path, end: int) -> None:
    """A new executor after evaluation restarts a nonrestartable input loader."""
    activity = re.compile(
        r"\b(?:Eval(?:uation)?|Validat(?:e|ion))\b"
        r"|\b(?:Loading|Loaded|Restoring|Restored) (?:a |the |weights from )?checkpoint\b",
        re.IGNORECASE,
    )
    starts = []
    schedule = re.compile(
        r"Starting train loop (\d+) of (\d+), from global step (\d+) to (\d+) \((\d+) steps?\)"
    )
    for line in path.read_text().splitlines():
        match = PROGRESS.search(line)
        if match and int(match[2]) == end:
            break
        if activity.search(line):
            raise ValueError(
                "Exact schedule accounting requires no evaluation/executor restart or checkpoint restore before the measurement endpoint"
            )
        loop = schedule.search(line)
        if loop:
            starts.append(tuple(map(int, loop.groups())))
    if (
        len(starts) != 1
        or starts[0][:3] != (1, 1, 1)
        or starts[0][3] < end
        or starts[0][3] != starts[0][4]
    ):
        raise ValueError(
            "Exact schedule accounting requires one logged train loop from global step 1 through the measurement endpoint"
        )


def count_scheduled_targets(values: list[int], start: int, end: int) -> int:
    """Count targets in (start, end] using a periodic batch schedule."""

    def count(steps):
        epochs, remainder = divmod(steps, len(values))
        return epochs * sum(values) + sum(values[:remainder])

    return count(end) - count(start)


def account_window(result: dict, contract: dict) -> None:
    if (
        contract.get("supervised_targets_by_batch_scope", "all_epochs")
        != "all_epochs"
    ):
        raise ValueError(
            "Input contract cannot establish repeated supervised target counts; "
            "reshuffled ignored labels require consumed-count measurements"
        )
    start, end = result["start_step"], result["end_step"]
    for field, schedule in (
        ("seed_nodes", "seed_nodes_by_batch"),
        ("supervised_targets", "supervised_targets_by_batch"),
    ):
        result[field] = count_scheduled_targets(contract[schedule], start, end)
        result[field + "_per_second"] = (
            result[field] / result["training_window_seconds"]
        )
    result["nominal_slots"] = result["nominal_batch_size"] * result["measured_steps"]
    result["padded_slots"] = result["nominal_slots"] - result["seed_nodes"]
    result["numerator_provenance"] = {
        "kind": "derived_from_runtime_sampler_batch_schedule",
        "input_contract": contract,
        "first_batch_global_step": 1,
        "assumptions": "Fresh single-streamer run with continuous traversal from batch zero; no checkpoint resume, epoch truncation or skipped input batches. Batch counts come from actual ordered split IDs and labels. This is schedule accounting, not a device performance counter.",
    }


def _summarize(path, start, end, batch_size, tolerance, *, probe):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Stability tolerance must be finite and positive")
    points = read_points(path)
    contract = read_input_contract(path)
    if batch_size is not None and batch_size != contract["batch_size"]:
        raise ValueError("Declared batch size differs from runtime input contract")
    batch_size = contract["batch_size"]
    static = contract.get("static_batch_cache_size", 0)
    if static and not probe:
        raise ValueError("Static replay cannot establish ordinary training throughput")
    if probe and not static:
        raise ValueError("A nominal replay probe requires static_batch_cache_size > 0")
    _check_training_only_window(path, start, end)
    _check_fresh_continuous_run(path, end)

    def interval(a, b):
        result = measure(points, a, b, batch_size)
        if probe:
            result["probe_nominal_slots_per_second"] = result.pop(
                "nominal_slots_per_second"
            )
            result["metric"] = "probe_nominal_slots_per_second"
            result["nominal_slots"] = batch_size * (b - a)
        else:
            account_window(result, contract)
            result["metric"] = "seed_nodes_per_second"
        result["throughput"] = result[result["metric"]]
        return result

    result = interval(start, end)
    result["input_contract"] = contract
    result["definition"] = (
        "Static replay mechanism probe; nominal batch slots/s, not ordinary training throughput"
        if probe
        else "Runtime-schedule valid seed nodes/s, supervised targets/s and nominal slots/s over identical exact endpoints"
    )
    midpoint = (start + end) // 2
    if (end - start) % 2 == 0 and any(p.step == midpoint for p in points):
        first, second = interval(start, midpoint), interval(midpoint, end)
        metric = result["metric"]
        a, b = first[metric], second[metric]
        difference = abs(a - b) / ((a + b) / 2) * 100
        result["half_window_check"] = {
            "first_half": first,
            "second_half": second,
            "symmetric_difference_percent": difference,
            "tolerance_percent": tolerance,
            "within_tolerance": difference <= tolerance,
            "interpretation": "screening heuristic, not proof of stationarity",
            "metric": metric,
        }
    else:
        result["half_window_check"] = None
        result["half_window_note"] = (
            "Exact midpoint timestamp unavailable; use log_steps=10"
        )
    return result


def summarize(
    path: Path,
    start: int = 40,
    end: int = 240,
    batch_size: int | None = None,
    tolerance: float = 2.0,
) -> dict:
    """Measure a fresh single-executor run with its runtime input contract."""
    return _summarize(path, start, end, batch_size, tolerance, probe=False)


def summarize_probe(
    path: Path,
    start: int = 40,
    end: int = 240,
    batch_size: int | None = None,
    tolerance: float = 2.0,
) -> dict:
    """Measure explicitly configured static replay as a separate probe metric."""
    return _summarize(path, start, end, batch_size, tolerance, probe=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--start-step", type=int, default=40)
    parser.add_argument("--end-step", type=int, default=240)
    parser.add_argument(
        "--batch-size", type=int, help="Optional cross-check against runtime batch size"
    )
    parser.add_argument("--stability-tolerance-percent", type=float, default=2.0)
    parser.add_argument(
        "--static-replay-probe",
        action="store_true",
        help="Emit the separate probe metric for a static replay run",
    )
    args = parser.parse_args()
    try:
        if (
            not math.isfinite(args.stability_tolerance_percent)
            or args.stability_tolerance_percent <= 0
        ):
            raise ValueError("Stability tolerance must be finite and positive")
        measurement = summarize_probe if args.static_replay_probe else summarize
        result = measurement(
            args.log,
            args.start_step,
            args.end_step,
            args.batch_size,
            args.stability_tolerance_percent,
        )
        print(json.dumps(result, indent=2))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
