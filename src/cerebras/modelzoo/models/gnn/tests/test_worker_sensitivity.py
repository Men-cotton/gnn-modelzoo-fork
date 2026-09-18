"""R04: all-condition repeats and evidence-preserving summaries without remote jobs."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from cerebras.modelzoo.models.gnn.tools import autotune as tune
import test_autotune
from test_autotune import write_log


class WorkerSensitivityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)
        self.cli = [
            "--mode",
            "sensitivity",
            "--dataset",
            "arxiv",
            "--wsc-workers",
            "1",
            "--workers",
            "40",
            "4",
            "8",
            "10",
            "12",
            "16",
            "20",
            "--output",
            str(self.output),
            "--budget-sec",
            "200000",
        ]
        self.args = tune.parse_args(self.cli)
        self.base = tune.load_params_file(tune.GNN / "configs/autotune/arxiv_w40.yaml")

    def study(self, args=None):
        return tune.Study(args or self.args, self.base, self.output, {"test": "host"})

    def test_all_conditions_unstable_included_and_resume(self):
        def execute(cmd, log, timeout):
            params = yaml.safe_load(Path(cmd[cmd.index("fit") + 1]).read_text())
            loader = params["trainer"]["fit"]["train_dataloader"]
            self.assertTrue(loader["persistent_workers"])
            self.assertEqual(loader["prefetch_factor"], 2)
            self.assertEqual(
                params["trainer"]["init"]["backend"]["cluster_config"][
                    "num_workers_per_csx"
                ],
                1,
            )
            self.assertEqual(params["trainer"]["init"]["loop"]["max_steps"], 440)
            # Baseline is slower and nonstationary; it must still get three runs.
            write_log(log, end=440, unstable=loader["num_workers"] == 40)
            return {"status": "completed", "returncode": 0}

        with patch.object(tune, "execute", side_effect=execute) as mocked:
            study = self.study()
            self.assertEqual(study.run(), 0)
            self.assertEqual(mocked.call_count, 21)
            rows = study.state["trials"]
            workers = self.args.workers
            self.assertEqual(
                [r["knobs"]["num_workers"] for r in rows],
                workers + workers[::-1] + workers,
            )
            self.assertEqual(len({r["command"][-1] for r in rows}), 21)
            report = json.loads((self.output / "sensitivity.json").read_text())
            self.assertEqual(report["summary"][0]["unstable_runs"], 3)
            self.assertEqual(report["summary"][0]["measured_runs"], 3)
            self.assertGreater(report["summary"][0]["mean"], 0)
            self.assertEqual(report["summary"][0]["sample_stddev"], 0)
            self.assertFalse((self.output / "best.yaml").exists())
            self.assertEqual(self.study().run(), 0)
            self.assertEqual(mocked.call_count, 21)

    def test_missing_runs_and_sample_standard_deviation(self):
        study = self.study()
        key = tune.candidate_id(tune.candidate(40, persistent=True))
        study.state["trials"] = [
            dict(
                trial_id=f"sensitivity_{key}_r{i}",
                candidate_id=key,
                knobs=tune.candidate(40, persistent=True),
                phase="sensitivity",
                repeat=i,
                status="completed",
                measurement={
                    "throughput": value,
                    "metric": "nominal_slots_per_second",
                },
            )
            for i, value in enumerate([2.0, 4.0], 1)
        ]
        study.save()
        report = json.loads((self.output / "sensitivity.json").read_text())
        summary = report["summary"][0]
        self.assertEqual(summary["mean"], 3.0)
        self.assertAlmostEqual(summary["sample_stddev"], 2**0.5)
        self.assertEqual(summary["not_run"], 1)
        self.assertIsNone(report["summary"][1]["mean"])

    def test_cpu_preflight_does_not_silently_drop_baseline(self):
        with (
            patch.object(tune, "get_available_cpu_cores", return_value=20),
            patch.object(tune, "environment") as env,
            patch.object(tune, "execute") as execute,
        ):
            with self.assertRaisesRegex(ValueError, "No jobs submitted"):
                tune.main(self.cli)
            self.assertEqual(tune.main(self.cli + ["--dry-run", "--cache", "full"]), 0)
            env.assert_not_called()
            execute.assert_not_called()
        plan = json.loads((self.output / "plan.json").read_text())
        self.assertEqual(plan["workers"], self.args.workers)
        self.assertEqual(plan["blocked_workers"], [40])
        self.assertEqual(plan["planned_trials"], 21)
        a = yaml.safe_load((self.output / "preview_w40.yaml").read_text())
        b = yaml.safe_load((self.output / "preview_w04.yaml").read_text())
        self.assertEqual(a["trainer"]["fit"]["train_dataloader"]["cache_fraction"], 1.0)
        for config in (a, b):
            config["trainer"]["init"].pop("model_dir")
            config["trainer"]["init"]["backend"]["cluster_config"].pop("job_labels")
            config["trainer"]["fit"]["train_dataloader"].pop("num_workers")
        self.assertEqual(a, b)

    def test_timeout_pause_acknowledgement_and_budget_resume(self):
        with patch.object(
            tune, "execute", return_value={"status": "timeout"}
        ) as execute:
            self.assertEqual(self.study().run(), 2)
            self.assertEqual(execute.call_count, 1)
            with self.assertRaisesRegex(ValueError, "Confirm its job"):
                self.study().run()
        args = deepcopy(self.args)
        args.acknowledge_stopped_jobs = True
        study = self.study(args)
        study.state["used_sec"] = 199000
        study.save()
        with patch.object(
            tune,
            "execute",
            side_effect=test_autotune.AutotuneTests.fake_execute,
        ) as execute:
            self.assertEqual(study.run(), 2)
            execute.assert_not_called()
            args.budget_sec = 400000
            self.assertEqual(self.study(args).run(), 2)
            self.assertEqual(execute.call_count, 20)
        report = json.loads((self.output / "sensitivity.json").read_text())
        self.assertEqual(report["status"], "completed_with_missing_measurements")
        self.assertEqual(report["summary"][0]["failed_or_invalid_runs"], 1)
        self.assertEqual(report["summary"][0]["measured_runs"], 2)

    def test_failures_and_timeouts_continue_without_retry_or_acknowledgement(
        self,
    ):
        args = tune.parse_args(self.cli + ["--continue-on-failure"])
        args.workers = [4, 8]
        calls = []

        def execute(cmd, log, timeout):
            calls.append(cmd)
            if len(calls) <= 2:
                return {"status": "failed" if len(calls) == 1 else "timeout"}
            write_log(log, end=440)
            return {"status": "completed", "returncode": 0}

        with patch.object(tune, "execute", side_effect=execute):
            study = self.study(args)
            self.assertEqual(study.run(), 2)
            self.assertEqual(len(calls), 6)
            self.assertEqual(self.study(args).run(), 2)
            self.assertEqual(len(calls), 6)
        self.assertEqual(study.state["status"], "completed_with_missing_measurements")
        for row in study.state["trials"][:2]:
            self.assertTrue(row["remote_stop_unconfirmed"])
            self.assertIn("continued_after_failure_at", row)
            self.assertNotIn("job_stop_acknowledged_at", row)
        report = json.loads((self.output / "sensitivity.json").read_text())
        self.assertEqual(sum(r["failed_or_invalid_runs"] for r in report["summary"]), 2)
        self.assertEqual(sum(r["measured_runs"] for r in report["summary"]), 4)

    def test_continue_policy_does_not_ignore_user_interrupt(self):
        args = tune.parse_args(self.cli + ["--continue-on-failure"])
        with patch.object(
            tune, "execute", return_value={"status": "interrupted"}
        ) as run:
            self.assertEqual(self.study(args).run(), 2)
            self.assertEqual(run.call_count, 1)
            with self.assertRaisesRegex(ValueError, "Confirm its job"):
                self.study(args).run()

    def test_rejects_confounded_or_ambiguous_settings(self):
        for extra in (
            ["--prefetch-factors", "1", "2"],
            ["--confirm-steps", "800"],
            ["--wsc-workers", "0"],
        ):
            with self.assertRaises(SystemExit):
                tune.parse_args(self.cli + extra)


if __name__ == "__main__":
    unittest.main()
