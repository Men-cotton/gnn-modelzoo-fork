"""Measure one completed CSX training run between exact logged step endpoints.

Invoke with uv run --no-sync tools/measure_window.py LOG.
The output follows the IA3-2026 nominal batch-slot throughput definition.
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


def summarize(
    path: Path,
    start: int = 40,
    end: int = 240,
    batch_size: int = 4096,
    tolerance: float = 2.0,
) -> dict:
    """Measure the full window and compare its two halves."""
    points = read_points(path)
    result = measure(points, start, end, batch_size)
    result["definition"] = "nominal seed-node slots/s; includes padded slots"
    midpoint = (start + end) // 2
    if (end - start) % 2 == 0 and any(p.step == midpoint for p in points):
        first = measure(points, start, midpoint, batch_size)
        second = measure(points, midpoint, end, batch_size)
        a, b = first["nominal_slots_per_second"], second["nominal_slots_per_second"]
        difference = abs(a - b) / ((a + b) / 2) * 100
        result["half_window_check"] = {
            "first_half": first,
            "second_half": second,
            "symmetric_difference_percent": difference,
            "tolerance_percent": tolerance,
            "within_tolerance": difference <= tolerance,
            "interpretation": "screening heuristic, not proof of stationarity",
        }
    else:
        result["half_window_check"] = None
        result["half_window_note"] = (
            "Exact midpoint timestamp unavailable; use log_steps=10 for new trials"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--start-step", type=int, default=40)
    parser.add_argument("--end-step", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--stability-tolerance-percent", type=float, default=2.0)
    args = parser.parse_args()
    try:
        if (
            not math.isfinite(args.stability_tolerance_percent)
            or args.stability_tolerance_percent <= 0
        ):
            raise ValueError("Stability tolerance must be finite and positive")
        result = summarize(
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
