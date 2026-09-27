"""Plot recorded validation curves for researcher review."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_curve(path):
    if "invalid" in path.resolve().parts:
        raise ValueError("Input is outside the active measurement set")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if sum(row.get("event") == "run" for row in records) != 1:
            raise ValueError("Expected exactly one native training run")
        if not records[-1].get("completed"):
            raise ValueError("Native training did not complete")
        rows = [row for row in records if row.get("event") == "eval"]
    else:
        result = json.loads(path.read_text())
        if result.get("review_status") != "pending_human_review":
            raise ValueError("Expected learning_curves.json with researcher review status")
        rows = result["evaluations"]
    if not rows:
        raise ValueError("No validation evaluations")
    previous = -1
    for row in rows:
        if row["step"] <= previous or not math.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1:
            raise ValueError("Invalid validation step or accuracy")
        previous = row["step"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curves", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [path.parent.name for path in args.curves]
    if len(labels) != len(args.curves):
        parser.error("Provide one label per curve")
    figure, axis = plt.subplots(figsize=(6, 4))
    try:
        for path, label in zip(args.curves, labels):
            rows = read_curve(path)
            axis.plot([row["step"] for row in rows], [row["accuracy"] for row in rows], marker="o", label=label)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    axis.set(xlabel="Training step", ylabel="Validation accuracy", ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
