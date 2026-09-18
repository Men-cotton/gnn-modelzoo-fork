#!/usr/bin/env python3
"""Recover bounded training measurements without submitting or modifying jobs.

The measured unit is one optimizer update. Rates include the training input
path and exclude pre-loop setup and the declared warmup. Native GPU first-use
compilation is excluded only insofar as warmup absorbs it. These rates are not
hardware utilization.
"""

import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


PROGRESS = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3,6}).*?"
    r"\bTrain Device=(?P<device>CSX|GPU|cuda(?::\d+)?), Step=(?P<step>\d+), "
    r"Loss=(?P<loss>[^,\s]+)"
)
ACTIVITY = re.compile(
    r"\b(?:Eval(?:uation)?|Validat(?:e|ion))\b"
    r"|\b(?:Saving|Saved) (?:a )?checkpoint\b"
    r"|\bCheckpoint (?:saved|saving)\b",
    re.I,
)
RESTORE = re.compile(
    r"\b(?:Loading|Loaded|Restoring|Restored) (?:a |the |weights from )?checkpoint\b",
    re.I,
)
RATES = (
    "updates_per_second",
    "sequences_per_second",
    "nominal_tokens_per_second",
    "valid_tokens_per_second",
    "loss_target_tokens_per_second",
)
STEP_FIELDS = (
    "step",
    "timestamp",
    "elapsed_seconds",
    "loss",
    "samples",
    "valid_tokens",
    "loss_target_tokens",
    "update_valid_tokens",
    "update_loss_target_tokens",
    "nsp_examples",
    "update_nsp_examples",
    "warmup",
)
RUN_FIELDS = (
    "name",
    "profile",
    "backend",
    "gpu_implementation",
    "repeat",
    "state",
    "exit_code",
    "measurement_status",
    "unstable",
    "start_step",
    "end_step",
    "measured_optimizer_steps",
    "training_window_seconds",
    "sequences",
    "nominal_tokens",
    *RATES,
    "client_wall_seconds",
    "job_ids",
    "result",
    "reason",
)


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _positive_integer(value):
    return type(value) is int and value > 0


def _loss(records):
    losses = [row.get("loss") for row in records]
    finite = [value for value in losses if _number(value)]
    return {
        "recorded_steps": len(records),
        "finite_steps": len(finite),
        "nonfinite_or_invalid_steps": len(losses) - len(finite),
        "first": losses[0] if losses else None,
        "last": losses[-1] if losses else None,
        "minimum_finite": min(finite) if finite else None,
        "maximum_finite": max(finite) if finite else None,
        "interpretation": "Observed training loss; no convergence or accuracy threshold.",
    }


def _definitions(launch):
    bert = str(launch.get("profile", "")).startswith("bert_")
    return {
        "sequences": "Fixed-length input sequences consumed by optimizer updates.",
        "nominal_tokens": "Sequences multiplied by configured sequence length, including padding.",
        "valid_tokens": "nonpadding_input_positions"
        if bert
        else "loss_target_positions",
        "loss_target_tokens": "masked_language_model_positions"
        if bert
        else "causal_language_model_positions",
    }


def _validate_records(records, launch, errors):
    end = launch.get("max_optimizer_steps")
    warmup = launch.get("warmup_steps")
    if not _positive_integer(end) or type(warmup) is not int or not 0 <= warmup < end:
        errors.append("Require 0 <= warmup_steps < max_optimizer_steps.")
        return False
    if not _positive_integer(
        launch.get("effective_batch_size")
    ) or not _positive_integer(launch.get("sequence_length")):
        errors.append(
            "Positive integer effective batch size and sequence length are required."
        )
        return False
    tolerance = launch.get("stability_tolerance_percent", 2.0)
    if not _number(tolerance) or tolerance <= 0:
        errors.append("Stability tolerance must be finite and positive.")
        return False
    steps = [row.get("step") for row in records]
    if steps != list(range(1, end + 1)):
        errors.append(
            "Expected one fresh run with every optimizer update from 1 through max_optimizer_steps; missing, repeated or restarted steps are invalid."
        )
    times = [row.get("elapsed_seconds") for row in records]
    if any(not _number(value) for value in times) or any(
        b <= a for a, b in zip(times, times[1:])
    ):
        errors.append("Training record times must be finite and strictly increasing.")
    elif times and times[0] < 0:
        errors.append("Training record elapsed time cannot be negative.")
    if any(not _number(row.get("loss")) for row in records):
        errors.append("Training loss contains a nonfinite or invalid value.")
    return not errors


def _rates(launch, seconds, start, end):
    updates = end - start
    sequences = updates * launch["effective_batch_size"]
    tokens = sequences * launch["sequence_length"]
    return {
        "start_step": start,
        "end_step": end,
        "measured_optimizer_steps": updates,
        "effective_batch_size": launch["effective_batch_size"],
        "sequence_length": launch["sequence_length"],
        "training_window_seconds": seconds,
        "sequences": sequences,
        "nominal_tokens": tokens,
        "updates_per_second": updates / seconds,
        "sequences_per_second": sequences / seconds,
        "nominal_tokens_per_second": tokens / seconds,
        "valid_tokens": None,
        "valid_tokens_per_second": None,
        "loss_target_tokens": None,
        "loss_target_tokens_per_second": None,
        "interval": "Completed optimizer updates (start_step, end_step].",
        "count_provenance": "Configured fixed effective batch and sequence length per completed optimizer update.",
    }


def _stability(measurement, by_step, start_time, end_time, tolerance):
    start, end = measurement["start_step"], measurement["end_step"]
    mid = (start + end) // 2
    measurement["half_window_check"] = None
    if not start < mid < end or mid not in by_step:
        return
    first_seconds = by_step[mid]["elapsed_seconds"] - start_time
    second_seconds = end_time - by_step[mid]["elapsed_seconds"]
    if first_seconds <= 0 or second_seconds <= 0:
        return
    first = (mid - start) / first_seconds
    second = (end - mid) / second_seconds
    difference = abs(first - second) / ((first + second) / 2) * 100
    measurement["half_window_check"] = {
        "midpoint_step": mid,
        "first_half_updates_per_second": first,
        "second_half_updates_per_second": second,
        "symmetric_difference_percent": difference,
        "tolerance_percent": tolerance,
        "within_tolerance": difference <= tolerance,
        "interpretation": "Screening heuristic, not proof of stationarity; unstable observations are retained.",
    }


def _modelzoo(text, launch):
    records, errors, activity = [], [], []
    completions = 0
    origin = None
    current_step = 0
    for line in text.splitlines():
        completions += "Training completed successfully!" in line
        match = PROGRESS.search(line)
        if match:
            try:
                timestamp = datetime.strptime(
                    match["timestamp"], "%Y-%m-%d %H:%M:%S,%f"
                )
                origin = timestamp if origin is None else origin
                loss = float(match["loss"])
            except ValueError:
                errors.append("Malformed training timestamp or loss: " + line)
                continue
            current_step = int(match["step"])
            records.append(
                {
                    "step": current_step,
                    "timestamp": match["timestamp"],
                    "elapsed_seconds": (timestamp - origin).total_seconds(),
                    "loss": loss,
                    "device": match["device"],
                }
            )
        if ACTIVITY.search(line):
            activity.append((current_step, line))
        if RESTORE.search(line):
            errors.append(
                "Checkpoint restore prevents fresh-run measurement: " + line.strip()
            )
    if not records:
        return records, None, [*errors, "No training progress records."]
    if completions != 1:
        errors.append("Expected exactly one successful training completion marker.")
    if any(
        ("GPU" if row["device"].startswith("cuda") else row["device"])
        != launch.get("backend")
        for row in records
    ):
        errors.append("Logged device differs from the prepared backend.")
    loops = re.findall(
        r"Starting train loop (\d+) of (\d+), from global step (\d+) to (\d+) \((\d+) steps?\)",
        text,
    )
    if loops and (len(loops) != 1 or loops[0][:3] != ("1", "1", "1")):
        errors.append("Expected one fresh continuous Model Zoo training executor.")
    if not _validate_records(records, launch, errors):
        return records, None, errors
    start, end = launch["warmup_steps"], launch["max_optimizer_steps"]
    by_step = {row["step"]: row for row in records}
    if start not in by_step:
        errors.append(
            f"Exact step {start} timestamp is unavailable; no interpolation or fabricated step-zero time is used."
        )
    for step, line in activity:
        if start <= step < end:
            errors.append(
                "Evaluation or checkpoint work overlaps the measurement window: "
                + line.strip()
            )
    if errors:
        return records, None, errors
    first, last = by_step[start], by_step[end]
    seconds = last["elapsed_seconds"] - first["elapsed_seconds"]
    measurement = _rates(launch, seconds, start, end)
    measurement.update(
        clock="Model Zoo client progress log timestamps; training-loop endpoint difference.",
        start_timestamp=first["timestamp"],
        end_timestamp=last["timestamp"],
        measurement_start_elapsed_seconds=first["elapsed_seconds"],
        measurement_end_elapsed_seconds=last["elapsed_seconds"],
    )
    _stability(
        measurement,
        by_step,
        first["elapsed_seconds"],
        last["elapsed_seconds"],
        launch.get("stability_tolerance_percent", 2.0),
    )
    return records, measurement, errors


def _native(text, launch):
    records, summaries, errors = [], [], []
    for index, line in enumerate(text.splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected a JSON object")
        except ValueError:
            errors.append(f"Malformed metrics.jsonl record at line {index}.")
            continue
        if row.get("event") == "train":
            if summaries:
                errors.append("Training record appears after final summary.")
            records.append(row)
        elif row.get("event") == "summary":
            summaries.append(row)
    if not records:
        return records, None, [*errors, "No native GPU training records."]
    if len(summaries) != 1:
        errors.append("Expected exactly one final native GPU measurement summary.")
    if not _validate_records(records, launch, errors):
        return records, None, errors
    summary = summaries[0]
    start, end = launch["warmup_steps"], launch["max_optimizer_steps"]
    by_step = {row["step"]: row for row in records}
    seconds = summary.get("measurement_window_seconds", summary.get("elapsed_seconds"))
    if not _number(seconds) or seconds <= 0:
        return (
            records,
            None,
            ["Native GPU summary requires a positive finite measurement duration."],
        )
    if "measurement_window_seconds" in summary and (
        not _number(summary.get("elapsed_seconds"))
        or not math.isclose(
            summary["elapsed_seconds"], seconds, rel_tol=1e-8, abs_tol=1e-7
        )
    ):
        errors.append(
            "Native GPU elapsed_seconds alias disagrees with the measurement window."
        )
    if (
        summary.get("warmup_steps") != start
        or summary.get("measured_optimizer_steps") != end - start
    ):
        errors.append("Native GPU summary window differs from the prepared window.")
    for field, expected in (("window_start_step", start), ("window_end_step", end)):
        if field in summary and summary[field] != expected:
            errors.append(f"Native GPU {field} differs from the prepared window.")
    end_time = by_step[end]["elapsed_seconds"]
    start_time = summary.get("measurement_start_elapsed_seconds", end_time - seconds)
    if not _number(start_time):
        return records, None, [*errors, "Native GPU measurement start time is invalid."]
    timestamp_start = by_step[start]["elapsed_seconds"] if start else 0.0
    if (
        start_time < timestamp_start - 1e-7
        or start_time >= by_step[start + 1]["elapsed_seconds"]
    ):
        errors.append(
            "Native GPU measured clock must begin after warmup and before the next completed update."
        )
    if not math.isclose(end_time - start_time, seconds, rel_tol=1e-8, abs_tol=1e-7):
        errors.append(
            "Native GPU explicit clock endpoints disagree with the summary duration."
        )
    if "measurement_end_elapsed_seconds" in summary:
        declared_end = summary["measurement_end_elapsed_seconds"]
        if not _number(declared_end) or not math.isclose(
            declared_end, end_time, rel_tol=1e-8, abs_tol=1e-7
        ):
            errors.append(
                "Native GPU measurement end differs from the last completed update."
            )
    measurement = _rates(launch, seconds, start, end)
    measurement.update(
        clock="Native GPU synchronized summary clock; includes data loading, transfer, forward/backward and optimizer.",
        measurement_start_elapsed_seconds=start_time,
        measurement_end_elapsed_seconds=end_time,
        progress_endpoint_seconds=end_time - timestamp_start,
        start_clock_provenance="explicit_summary"
        if "measurement_start_elapsed_seconds" in summary
        else "legacy_summary_duration_and_final_progress_record",
        peak_allocated_bytes=summary.get("peak_allocated_bytes"),
        peak_reserved_bytes=summary.get("peak_reserved_bytes"),
        peak_memory_scope=summary.get("peak_memory_scope", "measurement_window"),
    )
    for row in records:
        if row.get("samples") != row["step"] * launch["effective_batch_size"]:
            errors.append(
                "Native GPU consumed sequence counts disagree with effective batch size."
            )
            break
    if summary.get("samples") != measurement["sequences"]:
        errors.append(
            "Native GPU summary sequence count differs from the selected updates."
        )
    for field in ("valid_tokens", "loss_target_tokens"):
        values = [row.get(field) for row in records]
        if all(value is None for value in values):
            continue
        if any(type(value) is not int or value < 0 for value in values) or any(
            b < a for a, b in zip(values, values[1:])
        ):
            errors.append(
                f"Native GPU {field} must contain nonnegative cumulative integer counts."
            )
            continue
        count = values[end - 1] - (values[start - 1] if start else 0)
        if count > measurement["nominal_tokens"]:
            errors.append(f"Native GPU {field} exceeds nominal token positions.")
        measurement[field] = count
        measurement[field + "_per_second"] = count / seconds
        declared = summary.get("measured_" + field)
        if declared is not None and declared != count:
            errors.append(
                f"Native GPU measured_{field} differs from consumed-batch counts."
            )
    if "measured_nsp_examples" in summary:
        count = summary["measured_nsp_examples"]
        if count != measurement["sequences"]:
            errors.append(
                "Native GPU NSP count must equal the measured sequence count."
            )
        measurement["nsp_examples"] = count
    for field in (
        "sequences_per_second",
        "nominal_tokens_per_second",
        "valid_tokens_per_second",
        "loss_target_tokens_per_second",
    ):
        recorded = summary.get(
            "samples_per_second" if field == "sequences_per_second" else field
        )
        expected = measurement[field]
        if recorded is not None and (
            not _number(recorded)
            or expected is None
            or not math.isclose(recorded, expected, rel_tol=1e-8, abs_tol=1e-7)
        ):
            errors.append(
                f"Native GPU summary {field} disagrees with count/time recomputation."
            )
    _stability(
        measurement,
        by_step,
        start_time,
        end_time,
        launch.get("stability_tolerance_percent", 2.0),
    )
    return records, measurement if not errors else None, errors


def _artifacts(output):
    result = []
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output)
        if path.is_symlink():
            result.append(
                {
                    "path": str(relative),
                    "type": "symlink",
                    "target": str(path.readlink()),
                    "target_exists": path.exists(),
                    "size_bytes": path.lstat().st_size,
                }
            )
            continue
        if not path.is_file() or path.name in {"result.json", "step_metrics.csv"}:
            continue
        kind = "sdk_artifact" if "cerebras_logs" in relative.parts else "run_artifact"
        if path.name in {"console.log", "client.log", "metrics.jsonl"}:
            kind = "raw_measurement_log"
        result.append(
            {"path": str(relative), "type": kind, "size_bytes": path.stat().st_size}
        )
    return result


def collect_run(output: Path, launch: dict, status: dict) -> dict:
    """Write derived records; preserve failed attempts and every raw artifact."""
    output = Path(output)
    console = output / "console.log"
    text = console.read_text(errors="replace") if console.is_file() else ""
    native = (
        launch.get("backend") == "GPU" and launch.get("gpu_implementation") == "native"
    )
    metrics = output / "metrics.jsonl"
    if native:
        records, measurement, errors = _native(
            metrics.read_text(errors="replace") if metrics.is_file() else "", launch
        )
    else:
        records, measurement, errors = _modelzoo(text, launch)
    if launch.get("params_sha256"):
        params = output / "params.yaml"
        if (
            not params.is_file()
            or hashlib.sha256(params.read_bytes()).hexdigest()
            != launch["params_sha256"]
        ):
            errors.append(
                "Prepared params.yaml is missing or differs from its recorded SHA256."
            )
    if status.get("state") != "completed" or status.get("exit_code") != 0:
        errors.append(
            "Client did not report completed with exit_code 0; partial observations are excluded from valid-run statistics."
        )
    measurement_status = (
        "valid"
        if measurement is not None and not errors
        else "invalid"
        if records
        else "unavailable"
    )
    result = {
        "schema_version": 1,
        "output": str(output.resolve()),
        "launch": launch,
        "status": status,
        "measurement_status": measurement_status,
        "invalid_reasons": list(dict.fromkeys(errors)),
        "measurement": measurement,
        "loss": _loss(records),
        "step_records": records,
        "token_definitions": _definitions(launch),
        "job_ids": sorted(
            set(re.findall(r"\bwsjob-[A-Za-z0-9-]+", text))
            | set(status.get("job_ids", []))
        ),
        "artifacts": _artifacts(output),
        "scope": "Training-loop measurements. Compiler estimates and SDK reports are retained as artifacts, not inferred physical utilization.",
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "result.json", result)
    with (output / "step_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_safe(records))
    refresh_campaign(launch)
    return _safe(result)


def refresh_campaign(launch: dict) -> None:
    """Refresh the owning campaign after a client status or result changes."""
    if launch.get("study_dir"):
        study = Path(launch["study_dir"])
        campaign = _read_json(study / "campaign.json")
        if isinstance(campaign.get("jobs"), list):
            summarize_campaign(study, campaign["jobs"])


def _condition(launch, job):
    fields = (
        "profile",
        "backend",
        "gpu_implementation",
        "sequence_length",
        "effective_batch_size",
        "loader_batch_size",
        "grad_accum_steps",
        "precision",
        "warmup_steps",
        "max_optimizer_steps",
        "compile",
        "gradient_checkpointing",
        "source_sha256",
        "git_commit",
        "dataset_identity",
    )
    result = {field: launch.get(field) for field in fields}
    result["profile"] = launch.get("profile", job.get("profile"))
    arguments = launch.get("arguments", {})
    for field in (
        "seed",
        "num_workers",
        "csx_micro_batch_size",
        "stability_tolerance_percent",
    ):
        result[field] = launch.get(field, arguments.get(field))
    return result


def summarize_campaign(output: Path, jobs: list[dict]) -> dict:
    """Summarize only independently valid runs, retaining missing/failed rows."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # A submitted GPU campaign completes in separate scheduler processes. Read
    # all run results only after acquiring the lock, then publish both views.
    with (output / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # A waiting completion callback may carry an older submission snapshot.
        # Failed submissions have no client result that could correct that old
        # state, so reload the parent's journal under the same lock as results.
        campaign = _read_json(output / "campaign.json")
        if isinstance(campaign.get("jobs"), list):
            jobs = campaign["jobs"]
        return _summarize_campaign(output, jobs)


def _summarize_campaign(output, jobs):
    groups, rows = {}, []
    for job in jobs:
        config = Path(job["config"])
        if not config.is_absolute():
            config = output / config
        directory = config.parent
        result = _read_json(directory / "result.json")
        launch = result.get("launch") or _read_json(directory / "launch.json")
        status = result.get("status") or _read_json(directory / "client_status.json")
        measure = result.get("measurement") or {}
        check = measure.get("half_window_check") or {}
        reasons = (
            result.get("invalid_reasons", [])
            if result
            else ["No collected run result."]
        )
        reasons = [
            *reasons,
            *(str(value) for value in (job.get("error"), status.get("error")) if value),
        ]
        row = {
            "name": job.get("name", directory.name),
            "profile": launch.get("profile", job.get("profile")),
            "backend": launch.get("backend"),
            "gpu_implementation": launch.get("gpu_implementation"),
            "repeat": launch.get("repeat", job.get("repeat")),
            "state": status.get("state", job.get("state", "missing")),
            "exit_code": status.get("exit_code", job.get("exit_code")),
            "measurement_status": result.get("measurement_status", "unavailable"),
            "unstable": not check["within_tolerance"]
            if "within_tolerance" in check
            else None,
            "client_wall_seconds": status.get(
                "client_wall_seconds", status.get("wall_seconds")
            ),
            "job_ids": ";".join(result.get("job_ids", status.get("job_ids", []))),
            "result": str(directory / "result.json") if result else None,
            "reason": "; ".join(dict.fromkeys(reasons)),
            **{key: measure.get(key) for key in RUN_FIELDS if key in measure},
        }
        for field in RUN_FIELDS:
            row.setdefault(field, None)
        rows.append(row)
        condition = _condition(launch, job)
        key = json.dumps(condition, sort_keys=True)
        group = groups.setdefault(key, {"condition": condition, "runs": []})
        group["runs"].append(row)
    for group in groups.values():
        members = group["runs"]
        valid = [row for row in members if row["measurement_status"] == "valid"]
        group["counts"] = _counts(members)
        group["metrics"] = {}
        for field in RATES:
            values = [row[field] for row in valid if _number(row.get(field))]
            group["metrics"][field] = {
                "values": values,
                "count": len(values),
                "mean": statistics.mean(values) if values else None,
                "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
            }
    counts = _counts(rows)
    state = (
        "running"
        if counts["running"]
        else "pending"
        if counts["pending"] or not rows
        else "completed"
        if counts["valid"] == len(rows)
        else "completed_with_missing"
    )
    summary = {
        "schema_version": 1,
        "state": state,
        "counts": counts,
        "planned_runs": len(jobs),
        "groups": list(groups.values()),
        "runs": rows,
        "aggregation": "Only measurement_status=valid observations enter statistics. Unstable valid observations remain included and are counted separately. Missing observations are never zero.",
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summary)
    temporary = output / "runs.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "runs.csv")
    return summary


def _counts(rows):
    failures = {"failed", "timeout", "launch_failed", "prepare_failed"}
    terminal = failures | {"completed", "interrupted"}
    running = {"client_running", "running"}
    return {
        "planned": len(rows),
        "pending": sum(row["state"] not in terminal | running for row in rows),
        "running": sum(row["state"] in running for row in rows),
        "completed": sum(row["state"] == "completed" for row in rows),
        "failed": sum(row["state"] in failures for row in rows),
        "interrupted": sum(row["state"] == "interrupted" for row in rows),
        "missing": sum(row["result"] is None for row in rows),
        "valid": sum(row["measurement_status"] == "valid" for row in rows),
        "invalid": sum(row["measurement_status"] == "invalid" for row in rows),
        "unavailable": sum(row["measurement_status"] == "unavailable" for row in rows),
        "unstable": sum(
            row["measurement_status"] == "valid" and row["unstable"] is True
            for row in rows
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        required=True,
        help="Existing campaign directory; recollect finalized runs and summarize scheduler submissions.",
    )
    args = parser.parse_args(argv)
    path = args.campaign.resolve()
    state = _read_json(path / "campaign.json")
    if not isinstance(state.get("jobs"), list):
        parser.error("Expected campaign.json with a jobs list")
    for job in state["jobs"]:
        config = Path(job["config"])
        if not config.is_absolute():
            config = path / config
        output = config.parent
        launch = _read_json(output / "launch.json")
        status = _read_json(output / "client_status.json")
        if launch and status.get("state") in {
            "completed",
            "failed",
            "timeout",
            "interrupted",
        }:
            collect_run(output, launch, status)
    summary = summarize_campaign(path, state["jobs"])
    print(
        json.dumps(
            {
                "summary": str(path / "summary.json"),
                "planned_runs": summary["planned_runs"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
