"""Measure PyG CUDA windows from cumulative seed counts and synchronized wall time."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

PREFIX = "[Autotune] "
COMPLETE = re.compile(r"^Training Completed\. Total Steps: (\d+)")


@dataclass(frozen=True)
class Point:
    step: int
    seconds: float
    seeds: int
    optimizer_steps: int | None = None
    skipped_optimizer_steps: int | None = None


def read_points(path: Path) -> list[Point]:
    if "invalid" in path.resolve().parts:
        raise ValueError("Input is outside the active measurement set")
    points = []
    completed = []
    for line in path.read_text().splitlines():
        match = COMPLETE.match(line)
        if match:
            completed.append(int(match[1]))
        if "[Eval]" in line:
            raise ValueError("Evaluation is present in the measurement log")
        if not line.startswith(PREFIX):
            continue
        try:
            row = json.loads(line[len(PREFIX) :])
            if row.get("version") not in (1, 2):
                raise ValueError("Unsupported PyG measurement version")
            step, seeds = row["step"], row["seed_nodes"]
            if (
                type(step) is not int
                or type(seeds) is not int
                or seeds <= 0
                or step <= 0
            ):
                raise ValueError(
                    "Step and cumulative seed count must be positive integers"
                )
            updates = skipped = None
            if row["version"] == 2:
                updates, skipped = (
                    row["optimizer_steps"],
                    row["skipped_optimizer_steps"],
                )
                scale = row["loss_scale"]
                if (
                    type(updates) is not int
                    or type(skipped) is not int
                    or min(updates, skipped) < 0
                    or updates + skipped != step
                    or not math.isfinite(scale)
                    or scale <= 0
                ):
                    raise ValueError("Invalid PyG optimizer update accounting")
            point = Point(step, float(row["wall_seconds"]), seeds, updates, skipped)
            finite = row["all_losses_finite"] is True and math.isfinite(
                float(row["loss"])
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid PyG measurement record: {exc}") from exc
        if not finite or not math.isfinite(point.seconds) or point.seconds <= 0:
            raise ValueError("Nonfinite loss or invalid wall time")
        if points and (
            point.step <= points[-1].step
            or point.seconds <= points[-1].seconds
            or point.seeds <= points[-1].seeds
        ):
            raise ValueError(
                "Steps, wall times and seed counts must increase; use one fresh log"
            )
        if points:
            previous = points[-1]
            if (previous.optimizer_steps is None) != (point.optimizer_steps is None):
                raise ValueError("Mixed PyG measurement versions")
            if point.optimizer_steps is not None and (
                point.optimizer_steps < previous.optimizer_steps
                or point.skipped_optimizer_steps < previous.skipped_optimizer_steps
            ):
                raise ValueError("Optimizer counters must not decrease")
        points.append(point)
    if len(points) < 2 or completed != [points[-1].step]:
        raise ValueError("Expected one completed run with a final measurement record")
    return points


def measure(points: list[Point], start: int, end: int) -> dict:
    by_step = {p.step: p for p in points}
    if start < 0 or end <= start or start not in by_step or end not in by_step:
        raise ValueError(f"Exact step {start} and {end} timestamps required")
    a, b = by_step[start], by_step[end]
    seconds = b.seconds - a.seconds
    if seconds <= 0:
        raise ValueError("Nonpositive measurement duration")
    count = b.seeds - a.seeds
    metric = "seed_nodes_per_second"
    result = {
        "start_step": start,
        "end_step": end,
        "measured_steps": end - start,
        "training_window_seconds": seconds,
        "count": count,
        "seed_nodes": count,
        "seed_nodes_per_second": count / seconds,
        "metric": metric,
        "throughput": count / seconds,
    }
    result["optimizer_update_check"] = "unavailable_legacy_log"
    if a.optimizer_steps is not None:
        result.update(
            optimizer_update_check="recorded",
            optimizer_steps=b.optimizer_steps - a.optimizer_steps,
            skipped_optimizer_steps=b.skipped_optimizer_steps
            - a.skipped_optimizer_steps,
        )
    return result


def summarize(path: Path, start: int, end: int, tolerance: float = 2.0) -> dict:
    points = read_points(path)
    result = measure(points, start, end)
    midpoint = (start + end) // 2
    first = measure(points, start, midpoint)
    second = measure(points, midpoint, end)
    a, b = first["throughput"], second["throughput"]
    difference = abs(a - b) / ((a + b) / 2) * 100
    result["half_window_check"] = {
        "first_half": first,
        "second_half": second,
        "symmetric_difference_percent": difference,
        "tolerance_percent": tolerance,
        "within_tolerance": difference <= tolerance,
        "interpretation": "screening heuristic, not proof of stationarity",
    }
    result["definition"] = (
        "actual trained seed nodes / synchronized training-window wall time"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--start-step", type=int, default=40)
    parser.add_argument("--end-step", type=int, default=440)
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.log, args.start_step, args.end_step), indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
