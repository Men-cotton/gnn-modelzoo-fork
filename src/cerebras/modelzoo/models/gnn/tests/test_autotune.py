"""Host-only orchestration tests. No CSX jobs or datasets are required."""

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from cerebras.modelzoo.models.gnn.tools import autotune as tune
from cerebras.modelzoo.models.gnn.tools.measure_window import summarize


def write_log(path, end=240, seconds=1, unstable=False):
    origin = datetime(2026, 1, 1)
    lines = ["Job wsjob-autotune-test"]
    for step in range(10, end + 1, 10):
        elapsed = step * seconds + (max(0, step - 140) if unstable else 0)
        stamp = (origin + timedelta(seconds=elapsed)).strftime("%Y-%m-%d %H:%M:%S,%f")[
            :-3
        ]
        lines.append(
            f"{stamp} INFO | Train Device=CSX, Step={step}, Loss=1.0, Rate=9999, GlobalRate=9999"
        )
    lines.append("Training completed successfully!")
    path.write_text("\n".join(lines))


class AutotuneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.args = tune.parse_args(
            [
                "--dataset",
                "arxiv",
                "--output",
                str(self.output),
                "--workers",
                "0",
                "4",
                "--budget-sec",
                "100000",
            ]
        )
        self.base = tune.load_params_file(tune.GNN / "configs/autotune/arxiv_w40.yaml")

    def study(self, args=None):
        return tune.Study(
            args or self.args, self.base, self.output, {"test": "host only"}
        )

    @staticmethod
    def fake_execute(cmd, log, timeout):
        params = yaml.safe_load(Path(cmd[6]).read_text())
        loader = params["trainer"]["fit"]["train_dataloader"]
        init = params["trainer"]["init"]
        # Worker 4 is faster. The supplied GlobalRate deliberately contradicts the timing.
        write_log(
            log,
            end=init["loop"]["max_steps"],
            seconds=1 if loader["num_workers"] else 2,
        )
        return {"status": "completed", "returncode": 0}

    def test_windows_and_rejections(self):
        log = self.output / "train.log"
        write_log(log)
        result = summarize(log)
        self.assertEqual(result["nominal_slots_per_second"], 4096)
        self.assertTrue(result["half_window_check"]["within_tolerance"])
        write_log(log, unstable=True)
        self.assertFalse(summarize(log)["half_window_check"]["within_tolerance"])
        valid = log.read_text()
        for content in (
            valid.replace("Step=40,", "Step=41,"),
            valid.replace("Loss=1.0,", "Loss=nan,"),
            valid.replace("Training completed successfully!", ""),
            valid + "\n" + valid,
        ):
            log.write_text(content)
            with self.assertRaises(ValueError):
                summarize(log)

    def test_templates_and_model_conditions(self):
        paths = list((tune.GNN / "configs/autotune").glob("*.yaml"))
        self.assertEqual(len(paths), 12)
        for path in paths:
            base = tune.load_params_file(path)
            config = tune.get_backend("csx").prepare_config(
                base, tune.candidate(0), self.output / "model", 240, 7200
            )
            init = config["trainer"]["init"]
            fit = config["trainer"]["fit"]
            self.assertEqual(init["model"], base["trainer"]["init"]["model"])
            self.assertEqual(init["optimizer"], base["trainer"]["init"]["optimizer"])
            self.assertEqual(fit["train_dataloader"]["fanouts"], [15, 10, 5])
            self.assertEqual(init["loop"]["max_steps"], 240)
            self.assertIsNone(init["loop"]["steps_per_epoch"])
            self.assertFalse(init["checkpoint"]["autoload_last_checkpoint"])
            self.assertIsNone(init["checkpoint"]["steps"])
            self.assertIsNone(fit["val_dataloader"])
            self.assertIsNone(fit["ckpt_path"])
            self.assertIsNone(fit["train_dataloader"]["cache_fraction"])
            self.assertEqual(fit["train_dataloader"]["static_batch_cache_size"], 0)

    def test_adaptive_confirmation_resume_and_provenance(self):
        with patch.object(tune, "execute", side_effect=self.fake_execute) as execute:
            study = self.study()
            self.assertEqual(study.run(), 0)
            self.assertEqual(execute.call_count, 8)
            trials = study.state["trials"]
            confirmations = [
                r["knobs"]["num_workers"] for r in trials if r["phase"] == "confirm"
            ]
            self.assertEqual(confirmations, [4, 0, 0, 4, 4, 0])
            self.assertEqual(study.state["ranking"][0]["knobs"]["num_workers"], 4)
            self.assertEqual(len({r["command"][-1] for r in trials}), 8)
            self.assertEqual(trials[0]["job_ids"], ["wsjob-autotune-test"])
            self.assertEqual(self.study().run(), 0)
            self.assertEqual(execute.call_count, 8)
        self.assertEqual(
            yaml.safe_load((self.output / "best.yaml").read_text())["trainer"]["fit"][
                "train_dataloader"
            ]["num_workers"],
            4,
        )
        with self.assertRaisesRegex(ValueError, "changed"):
            tune.Study(self.args, self.base, self.output, {"test": "different source"})

    def test_budget_and_resume(self):
        study = self.study()
        study.state["used_sec"] = 95000
        study.save()
        with patch.object(tune, "execute", side_effect=self.fake_execute) as execute:
            self.assertEqual(study.run(), 2)
            execute.assert_not_called()
            self.assertEqual(study.state["status"], "budget_exhausted")
            args = deepcopy(self.args)
            args.budget_sec = 110000
            self.assertEqual(self.study(args).run(), 0)

    def test_failed_job_blocks_next_submission_and_requires_acknowledgement(self):
        with patch.object(
            tune, "execute", return_value={"status": "timeout"}
        ) as execute:
            self.assertEqual(self.study().run(), 2)
            self.assertEqual(execute.call_count, 1)
            with self.assertRaisesRegex(ValueError, "Confirm its job/process"):
                self.study().run()
        args = deepcopy(self.args)
        args.acknowledge_stopped_jobs = True
        with patch.object(tune, "execute", side_effect=self.fake_execute) as execute:
            resumed = self.study(args)
            self.assertEqual(resumed.run(), 0)
            self.assertEqual(
                execute.call_count, 4
            )  # Failed worker 0 is retained, not retried.
            self.assertEqual(resumed.state["trials"][0]["status"], "timeout")

    def test_unstable_repeat_disqualifies_candidate(self):
        with patch.object(tune, "execute", side_effect=self.fake_execute):
            study = self.study()
            study.run()
        study.state["trials"][-1]["status"] = "unstable"  # One slow worker repeat.
        self.assertEqual(len(tune.rank(study.state["trials"], "confirm", 3)), 1)

    def test_optional_loader_search(self):
        self.args.prefetch_factors = [1, 2]
        with patch.object(tune, "execute", side_effect=self.fake_execute) as execute:
            study = self.study()
            self.assertEqual(study.run(), 0)
        self.assertEqual(execute.call_count, 12)
        rows = [r for r in study.state["trials"] if r["phase"] == "loader"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["knobs"]["num_workers"] == 4 for r in rows))

    def test_real_subprocess_failure_and_timeout(self):
        log = self.output / "process.log"
        row = tune.execute(
            [sys.executable, "-c", "print('failure'); raise SystemExit(7)"], log, 5
        )
        self.assertEqual(row["returncode"], 7)
        self.assertIn("failure", log.read_text())
        row = tune.execute(
            [sys.executable, "-c", "import time; time.sleep(60)"], log, 0.05
        )
        self.assertEqual(row["status"], "timeout")

    def test_lock_and_dry_run(self):
        with tune.study_lock(self.output):
            with self.assertRaisesRegex(ValueError, "lock"):
                with tune.study_lock(self.output):
                    pass
        with (
            patch.object(tune, "environment") as env,
            patch.object(tune, "execute") as execute,
        ):
            self.assertEqual(
                tune.main(
                    [
                        "--dataset",
                        "products",
                        "--output",
                        str(self.output),
                        "--budget-sec",
                        "100000",
                        "--dry-run",
                    ]
                ),
                0,
            )
            env.assert_not_called()
            execute.assert_not_called()
        self.assertTrue((self.output / "plan.json").exists())
        self.assertFalse((self.output / "study.json").exists())


if __name__ == "__main__":
    unittest.main()
