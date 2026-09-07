"""Recompute historical GPU windows without copying or editing the source logs."""

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .measure_pyg import measure, read_points
else:
    from measure_pyg import measure, read_points


def analyze(artifacts: Path) -> dict:
    runs = []
    for date, infix in (("2026-01-30", ""), ("2026-06-22", "_graphsage")):
        for dataset in ("arxiv", "products"):
            for cache in ("not", "cache"):
                relative = f"raw_logs/{date}/{dataset}{infix}_1gpu_{cache}.log"
                path = artifacts / relative
                contents = path.read_text()
                points = read_points(path, legacy=True)
                last = points[-1].step
                reference = measure(points, 40, last, nominal_batch_size=4096)
                windows = []
                for start, length in (
                    (40, 200),
                    (40, 400),
                    (80, 400),
                    (40, 800),
                    (80, 800),
                ):
                    row = {"start_step": start, "end_step": start + length}
                    try:
                        full = measure(
                            points, start, start + length, nominal_batch_size=4096
                        )
                        row.update(full)
                        row["difference_from_reference_percent"] = (
                            full["throughput"] / reference["throughput"] - 1
                        ) * 100
                        try:
                            a = measure(
                                points,
                                start,
                                start + length // 2,
                                nominal_batch_size=4096,
                            )["throughput"]
                            b = measure(
                                points,
                                start + length // 2,
                                start + length,
                                nominal_batch_size=4096,
                            )["throughput"]
                            row["half_difference_percent"] = abs(a - b) / (a + b) * 200
                        except ValueError as exc:
                            row["half_difference_percent"] = None
                            row["half_note"] = str(exc)
                    except ValueError as exc:
                        row["unavailable"] = str(exc)
                    windows.append(row)
                runs.append(
                    {
                        "source": relative,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "dataset": dataset,
                        "cache": cache,
                        "first_step": points[0].step,
                        "last_step": last,
                        "reference": reference,
                        "windows": windows,
                        "cache_markers": [
                            s
                            for s in contents.splitlines()
                            if s.startswith("[GraphCache]")
                        ],
                        "provenance_lines": [
                            s
                            for s in contents.splitlines()
                            if any(
                                k in s
                                for k in (
                                    "git_commit=",
                                    "cuda_device_0=",
                                    "requested_no_compile=",
                                    "'num_workers':",
                                    "'batch_size':",
                                )
                            )
                        ],
                    }
                )
    return {
        "definition": "Historical nominal batch slots/s, not actual seed nodes/s",
        "selection": "Eight completed GraphSAGE single-GPU throughput logs; evaluation, GCN and invalid/ excluded",
        "reference": "step 40 to final logged step; overlapping intervals, not independent repetitions",
        "recommendation": {
            "pyg_warmup_steps": 40,
            "pyg_measure_steps": 400,
            "pyg_confirm_steps": 800,
            "repeats": 3,
        },
        "limitations": [
            "No per-step seed counts in these logs; tail batches are not corrected",
            "No minimum warm-up inferred from sparse 20/40-step logs",
            "January cached arxiv fails the 2 percent half-window screen at 40->440",
            "Arxiv ends at 500 steps; 800 measured steps is an unvalidated longer confirmation proposal",
            "New workers, profiler settings, software or GPU hardware require new measurements",
        ],
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.artifacts)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for run in result["runs"]:
        row = run["windows"][1]
        print(
            f"{run['source']}: 40->440 {row['training_window_seconds']:.3f}s, reference {row['difference_from_reference_percent']:+.3f}%, halves {row['half_difference_percent']:.3f}%"
        )


if __name__ == "__main__":
    main()
