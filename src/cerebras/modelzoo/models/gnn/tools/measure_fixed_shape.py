"""Measure exact synchronized endpoints in one completed native GPU run.

Accepts metrics.jsonl or the native runner's captured stdout. Endpoint elapsed
time includes inter-window logging, matching the CSX log-endpoint boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from .measure_window import count_scheduled_targets, validate_input_contract
except ImportError:
    from measure_window import count_scheduled_targets, validate_input_contract


def summarize(
    path: Path, start: int = 40, end: int = 240, tolerance: float = 2.0
) -> dict:
    if "invalid" in path.resolve().parts:
        raise ValueError("Input is outside the active measurement set")
    if start < 0 or end <= start or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Require 0 <= start < end and finite positive tolerance")
    records = []
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get("event") in (
            "run",
            "train",
            "eval",
            "checkpoint",
            "summary",
        ):
            records.append(row)
    runs = [row for row in records if row["event"] == "run"]
    summaries = [row for row in records if row["event"] == "summary"]
    if (
        len(runs) != 1
        or runs[0].get("backend") != "fixed_shape_gpu"
        or len(summaries) != 1
        or not summaries[0].get("completed")
    ):
        raise ValueError(
            "Expected one completed fixed_shape_gpu run from the current metric schema"
        )
    if records[0]["event"] != "run" or records[-1]["event"] != "summary":
        raise ValueError("Run metadata and completion must bracket all records")
    contract = validate_input_contract(runs[0].get("input_contract"))
    if contract.get("split") != "train" or contract.get("static_batch_cache_size", 0):
        raise ValueError(
            "Native training measurement requires a train input contract without static replay"
        )
    windows = [row for row in records if row["event"] == "train"]
    counts = (
        "steps",
        "seed_nodes",
        "supervised_targets",
        "nominal_slots",
        "optimizer_steps",
        "skipped_optimizer_steps",
    )

    def validate_counts(row):
        for field in counts:
            if (
                not isinstance(row.get(field), int)
                or isinstance(row[field], bool)
                or row[field] < 0
            ):
                raise ValueError("Training counters must be nonnegative integers")
        if (
            not 0
            <= row["supervised_targets"]
            <= row["seed_nodes"]
            <= row["nominal_slots"]
        ):
            raise ValueError("Require supervised targets <= seeds <= nominal slots")
        if row["optimizer_steps"] + row["skipped_optimizer_steps"] != row["steps"]:
            raise ValueError("Optimizer and skipped updates must sum to training steps")
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("Window duration must be finite and positive")

    previous_step = 0
    previous_time = -math.inf
    for row in windows:
        validate_counts(row)
        if not math.isfinite(float(row["loss"])):
            raise ValueError("Nonfinite training loss")
        boundary = float(row["boundary_monotonic_seconds"])
        if (
            row["start_step"] != previous_step
            or row["step"] <= previous_step
            or not math.isfinite(boundary)
            or boundary <= previous_time
        ):
            raise ValueError(
                "Training windows must be contiguous with increasing steps and timestamps"
            )
        if row["steps"] != row["step"] - row["start_step"]:
            raise ValueError("Window step count does not match its boundaries")
        expected = dict(
            seed_nodes=count_scheduled_targets(
                contract["seed_nodes_by_batch"], row["start_step"], row["step"]
            ),
            nominal_slots=contract["batch_size"] * row["steps"],
        )
        # Reshuffling ignored labels changes supervision per batch. In that
        # case use the consumed counters, whose bounds and totals we validate.
        if (
            contract.get("supervised_targets_by_batch_scope", "all_epochs")
            == "all_epochs"
        ):
            expected["supervised_targets"] = count_scheduled_targets(
                contract["supervised_targets_by_batch"], row["start_step"], row["step"]
            )
        for field, count in expected.items():
            if row[field] != count:
                raise ValueError(
                    "Observed batch counters disagree with runtime input contract"
                )
        previous_step, previous_time = row["step"], boundary
    if previous_step != summaries[0]["end_step"]:
        raise ValueError("Completion does not match final training window")
    summary = summaries[0]
    validate_counts(summary)
    if summary["steps"] != summary["end_step"] - summary["start_step"]:
        raise ValueError("Summary step count does not match its boundaries")
    measured = [row for row in windows if row["step"] > summary["start_step"]]
    if not measured or measured[0]["start_step"] != summary["start_step"]:
        raise ValueError("Summary must start at an exact window boundary")
    for field in counts:
        if summary[field] != sum(row[field] for row in measured):
            raise ValueError("Summary counters disagree with training windows")
    if not math.isclose(
        summary["seconds"],
        sum(row["seconds"] for row in measured),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("Summary duration disagrees with training windows")
    by_step = {row["step"]: row for row in windows}
    if start not in by_step or end not in by_step:
        raise ValueError("Exact start and end train timestamps are required")
    inside = False
    for row in records:
        if row["event"] == "train":
            if row["step"] == end:
                break
            if row["step"] == start:
                inside = True
        if inside and row["event"] in ("eval", "checkpoint"):
            raise ValueError(
                "Evaluation or checkpoint activity overlaps the training window"
            )

    def interval(a, b):
        selected = [row for row in windows if a < row["step"] <= b]
        seconds = (
            by_step[b]["boundary_monotonic_seconds"]
            - by_step[a]["boundary_monotonic_seconds"]
        )
        result = dict(
            start_step=a,
            end_step=b,
            measured_steps=b - a,
            training_window_seconds=seconds,
        )
        for count in (
            "seed_nodes",
            "supervised_targets",
            "nominal_slots",
            "optimizer_steps",
            "skipped_optimizer_steps",
        ):
            result[count] = sum(row[count] for row in selected)
        for count in ("seed_nodes", "supervised_targets", "nominal_slots"):
            result[count + "_per_second"] = result[count] / seconds
        result["padded_slots"] = result["nominal_slots"] - result["seed_nodes"]
        result["throughput"] = result["seed_nodes_per_second"]
        result["metric"] = "seed_nodes_per_second"
        result["summed_training_windows_seconds"] = sum(
            row["seconds"] for row in selected
        )
        return result

    result = interval(start, end)
    result["definition"] = (
        "Consumed target_mask seeds/s between synchronized endpoints; includes logging, excludes warmup and rejects overlapping evaluation/checkpoint activity. Supervised targets also exclude label -100."
    )
    result["numerator_provenance"] = "counted from consumed CPU payload masks"
    result["input_contract"] = contract
    midpoint = (start + end) // 2
    if (end - start) % 2 == 0 and midpoint in by_step:
        first, second = interval(start, midpoint), interval(midpoint, end)
        a, b = first["throughput"], second["throughput"]
        difference = abs(a - b) / ((a + b) / 2) * 100 if a + b else 0
        result["half_window_check"] = dict(
            first_half=first,
            second_half=second,
            symmetric_difference_percent=difference,
            tolerance_percent=tolerance,
            within_tolerance=difference <= tolerance,
            interpretation="screening heuristic, not proof of stationarity",
        )
    else:
        result["half_window_check"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--start-step", type=int, default=40)
    parser.add_argument("--end-step", type=int, default=240)
    parser.add_argument("--stability-tolerance-percent", type=float, default=2.0)
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                summarize(
                    args.log,
                    args.start_step,
                    args.end_step,
                    args.stability_tolerance_percent,
                ),
                indent=2,
            )
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
