"""Offline checks for exact windows, failed runs and independent-run statistics."""

import contextlib
import csv
from datetime import datetime, timedelta
import importlib
import io
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
measurements = importlib.import_module("measurements")


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="non-gnn-measurements-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.launch = {
            "profile": "bert_large_msl128",
            "backend": "CSX",
            "sequence_length": 128,
            "effective_batch_size": 4,
            "max_optimizer_steps": 6,
            "warmup_steps": 2,
            "repeat": 1,
            "precision": "bfloat16",
        }
        self.status = {"state": "completed", "exit_code": 0, "wall_seconds": 100}

    @staticmethod
    def log(seconds=1, end=6, complete=True):
        lines = [
            "Job wsjob-measurement-test",
            f"Starting train loop 1 of 1, from global step 1 to {end} ({end} steps)",
        ]
        origin = datetime(2026, 9, 18)
        for step in range(1, end + 1):
            value = seconds(step) if callable(seconds) else step * seconds
            stamp = (origin + timedelta(seconds=value)).strftime(
                "%Y-%m-%d %H:%M:%S,%f"
            )[:-3]
            lines.append(
                f"{stamp} INFO | Train Device=CSX, Step={step}, Loss={1 / step}, Rate=999999, GlobalRate=999999"
            )
        if complete:
            lines.append("Training completed successfully!")
        return "\n".join(lines)

    def run_csx(self, name="run", text=None, status=None, launch=None):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        (directory / "console.log").write_text(self.log() if text is None else text)
        return measurements.collect_run(
            directory, launch or self.launch, status or self.status
        )

    def native(self, explicit=True):
        launch = dict(self.launch, backend="GPU", gpu_implementation="native")
        rows = [
            {
                "event": "train",
                "step": step,
                "loss": 1 / step,
                "elapsed_seconds": float(step),
                "samples": step * 4,
                "valid_tokens": step * 100,
                "loss_target_tokens": step * 10,
            }
            for step in range(1, 7)
        ]
        seconds = 3.9  # Separate synchronization/logging gap after warmup.
        summary = {
            "event": "summary",
            "warmup_steps": 2,
            "measured_optimizer_steps": 4,
            "elapsed_seconds": seconds,
            "samples": 16,
            "samples_per_second": 16 / seconds,
            "nominal_tokens_per_second": 2048 / seconds,
            "valid_tokens_per_second": 400 / seconds,
            "loss_target_tokens_per_second": 40 / seconds,
            "measured_valid_tokens": 400,
            "measured_loss_target_tokens": 40,
            "peak_allocated_bytes": 1000,
            "peak_reserved_bytes": 2000,
        }
        if explicit:
            summary.update(
                window_start_step=2,
                window_end_step=6,
                measurement_start_elapsed_seconds=2.1,
                measurement_end_elapsed_seconds=6.0,
                measurement_window_seconds=seconds,
            )
        return launch, [*rows, summary]

    def collect_native(self, launch, records, name="native"):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        (directory / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )
        return measurements.collect_run(directory, launch, self.status)

    def test_exact_window_and_artifact_inventory(self):
        directory = self.root / "run"
        directory.mkdir()
        sdk = directory / "cerebras_logs" / "run"
        sdk.mkdir(parents=True)
        (sdk / "performance.json").write_text('{"compiler_estimate": 999999}')
        result = self.run_csx()
        self.assertEqual(result["measurement_status"], "valid")
        metric = result["measurement"]
        self.assertEqual((metric["start_step"], metric["end_step"]), (2, 6))
        self.assertEqual(metric["training_window_seconds"], 4)
        self.assertEqual(metric["sequences"], 16)
        self.assertEqual(metric["nominal_tokens"], 2048)
        self.assertEqual(metric["nominal_tokens_per_second"], 512)
        self.assertIsNone(metric["valid_tokens"])
        self.assertIsNone(metric["loss_target_tokens"])
        self.assertTrue(metric["half_window_check"]["within_tolerance"])
        self.assertEqual(result["job_ids"], ["wsjob-measurement-test"])
        self.assertEqual(result["loss"]["first"], 1)
        self.assertEqual(result["loss"]["last"], 1 / 6)
        self.assertTrue(
            any(row["type"] == "sdk_artifact" for row in result["artifacts"])
        )
        with (directory / "step_metrics.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 6)
        self.assertEqual(
            json.loads((directory / "result.json").read_text())["measurement_status"],
            "valid",
        )

    def test_rejects_missing_nonfinite_restarts_and_activity(self):
        valid = self.log()
        mutations = {
            "missing": valid.replace("Step=2,", "Step=20,"),
            "nonfinite": valid.replace("Loss=1.0,", "Loss=nan,"),
            "duplicate": valid + "\n" + valid,
            "incomplete": valid.replace("Training completed successfully!", ""),
            "checkpoint": valid.replace("Step=4,", "Step=4,").replace(
                "2026-09-18 00:00:04", "Saving checkpoint\n2026-09-18 00:00:04"
            ),
            "evaluation": valid.replace(
                "2026-09-18 00:00:04", "Evaluation started\n2026-09-18 00:00:04"
            ),
            "same_time": valid.replace("00:00:03", "00:00:02"),
            "restore": "Loading checkpoint weights\n" + valid,
        }
        for name, text in mutations.items():
            with self.subTest(name=name):
                result = self.run_csx(name, text)
                self.assertEqual(result["measurement_status"], "invalid")
                self.assertTrue(result["invalid_reasons"])
                self.assertTrue(result["step_records"])
                json.loads(
                    (self.root / name / "result.json").read_text(),
                    parse_constant=lambda value: self.fail(f"Nonstandard JSON {value}"),
                )

    def test_failed_partial_and_missing_results_are_not_zero(self):
        result = self.run_csx(
            "partial",
            self.log(end=3, complete=False),
            {"state": "failed", "exit_code": 1},
        )
        self.assertEqual(result["status"]["state"], "failed")
        self.assertEqual(result["measurement_status"], "invalid")
        self.assertEqual(len(result["step_records"]), 3)
        self.assertIsNone(result["measurement"])
        no_steps = self.run_csx(
            "no_steps",
            "Compiler estimated performance 999999",
            {"state": "failed", "exit_code": 1},
        )
        self.assertEqual(no_steps["measurement_status"], "unavailable")
        self.assertIsNone(no_steps["measurement"])
        failed_after_training = self.run_csx(
            "failed_after_training", status={"state": "failed", "exit_code": 1}
        )
        self.assertEqual(failed_after_training["measurement_status"], "invalid")
        self.assertIsNotNone(failed_after_training["measurement"])

    def test_modelzoo_gpu_real_sdk_device_string(self):
        launch = dict(self.launch, backend="GPU", gpu_implementation="modelzoo")
        for device in ("cuda", "cuda:0"):
            result = self.run_csx(
                device.replace(":", "_"),
                self.log().replace("Device=CSX", "Device=" + device),
                launch=launch,
            )
            self.assertEqual(
                result["measurement_status"], "valid", result["invalid_reasons"]
            )
            self.assertEqual(result["step_records"][0]["device"], device)

    def test_native_clock_and_meaningful_token_counts(self):
        for explicit in (False, True):
            launch, records = self.native(explicit)
            result = self.collect_native(launch, records, str(explicit))
            self.assertEqual(
                result["measurement_status"], "valid", result["invalid_reasons"]
            )
            metric = result["measurement"]
            self.assertEqual(metric["training_window_seconds"], 3.9)
            self.assertEqual(metric["progress_endpoint_seconds"], 4)
            self.assertEqual(metric["valid_tokens"], 400)
            self.assertEqual(metric["loss_target_tokens"], 40)
            self.assertEqual(
                result["token_definitions"]["valid_tokens"],
                "nonpadding_input_positions",
            )
            self.assertEqual(
                result["token_definitions"]["loss_target_tokens"],
                "masked_language_model_positions",
            )
            self.assertEqual(metric["peak_reserved_bytes"], 2000)
        launch["profile"] = "llama3p2_1b_msl1024"
        result = self.collect_native(launch, records, "llama")
        self.assertEqual(
            result["token_definitions"]["valid_tokens"], "loss_target_positions"
        )

    def test_native_detects_count_clock_and_summary_corruption(self):
        for name, change in (
            ("count", lambda rows: rows[3].update(samples=500)),
            ("clock", lambda rows: rows[-1].update(measurement_end_elapsed_seconds=7)),
            (
                "bad_clock_type",
                lambda rows: rows[-1].update(measurement_end_elapsed_seconds="oops"),
            ),
            ("window", lambda rows: rows[-1].update(window_start_step=1)),
            ("tokens", lambda rows: rows[-1].update(measured_valid_tokens=999)),
            ("rate", lambda rows: rows[-1].update(samples_per_second=math.inf)),
            ("duplicate_summary", lambda rows: rows.append(dict(rows[-1]))),
            ("nan_loss", lambda rows: rows[0].update(loss=math.nan)),
        ):
            with self.subTest(name=name):
                launch, rows = self.native()
                change(rows)
                result = self.collect_native(launch, rows, name)
                self.assertEqual(result["measurement_status"], "invalid")
                self.assertTrue(result["invalid_reasons"])

    def test_zero_warmup_no_fabricated_csx_endpoint(self):
        launch = dict(self.launch, warmup_steps=0)
        result = self.run_csx("zero_csx", launch=launch)
        self.assertEqual(result["measurement_status"], "invalid")
        self.assertIn("Exact step 0", " ".join(result["invalid_reasons"]))
        launch, rows = self.native()
        launch["warmup_steps"] = 0
        rows[-1].update(
            warmup_steps=0,
            measured_optimizer_steps=6,
            elapsed_seconds=6.0,
            window_start_step=0,
            measurement_start_elapsed_seconds=0.0,
            measurement_window_seconds=6.0,
            samples=24,
            samples_per_second=4.0,
            nominal_tokens_per_second=512.0,
            measured_valid_tokens=600,
            measured_loss_target_tokens=60,
            valid_tokens_per_second=100.0,
            loss_target_tokens_per_second=10.0,
        )
        result = self.collect_native(launch, rows)
        self.assertEqual(
            result["measurement_status"], "valid", result["invalid_reasons"]
        )

    def test_changed_prepared_configuration_is_not_valid(self):
        directory = self.root / "config_check"
        directory.mkdir()
        config = b"trainer: original\n"
        (directory / "params.yaml").write_bytes(config)
        launch = dict(self.launch, params_sha256=hashlib.sha256(config).hexdigest())
        self.assertEqual(
            self.run_csx("config_check", launch=launch)["measurement_status"], "valid"
        )
        (directory / "params.yaml").write_text("trainer: changed\n")
        result = self.run_csx("config_check", launch=launch)
        self.assertEqual(result["measurement_status"], "invalid")
        self.assertIn("SHA256", " ".join(result["invalid_reasons"]))

    def test_inventory_records_links_without_following_them(self):
        directory = self.root / "links"
        directory.mkdir()
        target = self.root / "outside_run"
        target.mkdir()
        (target / "payload.bin").write_bytes(b"artifact")
        (directory / "latest").symlink_to(target, target_is_directory=True)
        (directory / "missing").symlink_to("unavailable-artifact")
        result = self.run_csx("links")
        entries = {row["path"]: row for row in result["artifacts"]}
        self.assertEqual(entries["latest"]["type"], "symlink")
        self.assertEqual(entries["latest"]["target"], str(target))
        self.assertTrue(entries["latest"]["target_exists"])
        self.assertFalse(entries["missing"]["target_exists"])
        self.assertNotIn("latest/payload.bin", entries)

    def test_aggregate_keeps_failure_missing_and_unstable_counts(self):
        jobs = []
        for name in ("one", "two", "unstable", "failed", "missing"):
            directory = self.root / name
            directory.mkdir()
            (directory / "launch.json").write_text(json.dumps(self.launch))
            jobs.append(
                {
                    "name": name,
                    "config": str(directory / "params.yaml"),
                    "state": "qsub_returned",
                }
            )
        self.run_csx("one")
        self.run_csx("two", self.log(seconds=2))
        self.run_csx("unstable", self.log(seconds=lambda step: step + max(0, step - 4)))
        self.run_csx(
            "failed",
            self.log(end=3, complete=False),
            {"state": "failed", "exit_code": 1},
        )
        summary = measurements.summarize_campaign(self.root, jobs)
        self.assertEqual(len(summary["groups"]), 1)
        group = summary["groups"][0]
        self.assertEqual(
            group["counts"],
            {
                "planned": 5,
                "pending": 1,
                "running": 0,
                "completed": 3,
                "failed": 1,
                "interrupted": 0,
                "missing": 1,
                "valid": 3,
                "invalid": 1,
                "unavailable": 1,
                "unstable": 1,
            },
        )
        metric = group["metrics"]["nominal_tokens_per_second"]
        self.assertEqual(metric["values"], [512, 256, 2048 / 6])
        self.assertIsNotNone(metric["sample_stddev"])
        self.assertEqual(group["metrics"]["valid_tokens_per_second"]["count"], 0)
        self.assertIsNone(group["metrics"]["valid_tokens_per_second"]["mean"])
        single = measurements.summarize_campaign(self.root, jobs[:1])
        self.assertIsNone(
            single["groups"][0]["metrics"]["nominal_tokens_per_second"]["sample_stddev"]
        )
        self.assertEqual(single["state"], "completed")

    def test_summary_reads_client_state_and_preserves_launch_errors(self):
        directory = self.root / "running"
        directory.mkdir()
        (directory / "launch.json").write_text(json.dumps(self.launch))
        status_path = directory / "client_status.json"
        status_path.write_text(json.dumps({"state": "client_running"}))
        jobs = [
            {
                "name": "running",
                "config": str(directory / "params.yaml"),
                "state": "qsub_returned",
            }
        ]
        summary = measurements.summarize_campaign(self.root, jobs)
        self.assertEqual(summary["state"], "running")
        self.assertEqual(summary["counts"]["running"], 1)
        status_path.write_text(
            json.dumps(
                {
                    "state": "failed",
                    "exit_code": 1,
                    "error": "client failed before collection",
                }
            )
        )
        jobs[0]["error"] = "scheduler response preserved"
        summary = measurements.summarize_campaign(self.root, jobs)
        self.assertEqual(summary["state"], "completed_with_missing")
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertIn("client failed before collection", summary["runs"][0]["reason"])
        self.assertIn("scheduler response preserved", summary["runs"][0]["reason"])
        self.assertIsNone(summary["runs"][0]["nominal_tokens_per_second"])

    def test_stale_completion_snapshot_cannot_erase_submission_failure(self):
        directory = self.root / "never_started"
        directory.mkdir()
        (directory / "launch.json").write_text(json.dumps(self.launch))
        old_jobs = [
            {
                "name": "never_started",
                "config": str(directory / "params.yaml"),
                "state": "prepared",
            }
        ]
        # A completed GPU job read old_jobs before waiting for the summary lock.
        # The parent then records a later qsub failure with no client artifacts.
        current_jobs = [
            {**old_jobs[0], "state": "launch_failed", "error": "qsub rejected"}
        ]
        (self.root / "campaign.json").write_text(json.dumps({"jobs": current_jobs}))
        measurements.summarize_campaign(self.root, current_jobs)
        summary = measurements.summarize_campaign(self.root, old_jobs)
        self.assertEqual(summary["state"], "completed_with_missing")
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertEqual(summary["counts"]["pending"], 0)
        self.assertEqual(summary["runs"][0]["state"], "launch_failed")
        self.assertIn("qsub rejected", summary["runs"][0]["reason"])

    def test_collect_refreshes_parent_summary_and_cli_recollects(self):
        run_dir = self.root / "child"
        run_dir.mkdir()
        launch = dict(self.launch, study_dir=str(self.root))
        jobs = [
            {
                "name": "child",
                "config": str(run_dir / "params.yaml"),
                "state": "qsub_returned",
            }
        ]
        (self.root / "campaign.json").write_text(json.dumps({"jobs": jobs}))
        (run_dir / "launch.json").write_text(json.dumps(launch))
        (run_dir / "client_status.json").write_text(json.dumps(self.status))
        self.run_csx("child", launch=launch)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["groups"][0]["counts"]["completed"], 1)
        (run_dir / "result.json").unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(measurements.main(["--campaign", str(self.root)]), 0)
        self.assertTrue((run_dir / "result.json").exists())
        with (self.root / "runs.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["measurement_status"], "valid")


if __name__ == "__main__":
    unittest.main()
