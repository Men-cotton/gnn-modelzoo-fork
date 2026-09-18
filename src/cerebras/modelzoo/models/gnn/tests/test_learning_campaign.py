"""Learning-curve collection and sequential tuning contracts; no physical CSX."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from cerebras.modelzoo.models.gnn.tools import learning_campaign as campaign
from test_autotune import write_log


def learning_log(path, steps=60, every=20, accuracy=0.01):
    write_log(path)
    lines = [
        line for line in path.read_text().splitlines() if "GNN_INPUT_CONTRACT " in line
    ]
    for step in range(10, steps + 1, 10):
        lines.append(f"Train Device=CSX, Step={step}, Loss=2.0, Rate=1000")
        if step % every == 0:
            lines.extend(
                [
                    "Training completed successfully!",
                    f"Eval Device=CSX, GlobalStep={step}, Batch=8, Loss=3.0, Rate=1000",
                    "Avg Eval Loss: 3.0",
                    f"eval/masked_accuracy = {accuracy}",
                    "Evaluation completed successfully!",
                ]
            )
    path.write_text("\n".join(lines))


class LearningCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "study"
        self.output.mkdir()
        self.cli = [
            "--dataset",
            "arxiv",
            "--output",
            str(self.output),
            "--learning-steps",
            "60",
            "--eval-every",
            "20",
            "--workers",
            "2",
            "4",
            "--prefetch-factors",
            "1",
            "--budget-sec",
            "100000",
        ]
        self.args = campaign.parse_args(self.cli)
        self.base = campaign.shared_base("arxiv")
        self.provenance = {"test": "synthetic client logs"}

    def fake_execute(self, command, log, timeout):
        cfg = yaml.safe_load(Path(command[command.index("fit") + 1]).read_text())
        init = cfg["trainer"]["init"]
        if init["loop"].get("eval_frequency"):
            learning_log(log, init["loop"]["max_steps"], init["loop"]["eval_frequency"])
        else:
            workers = cfg["trainer"]["fit"]["train_dataloader"]["num_workers"]
            write_log(
                log, end=init["loop"]["max_steps"], seconds=1 if workers == 4 else 2
            )
        return {"status": "completed", "returncode": 0}

    def test_low_flat_accuracy_is_reported_without_learning_judgment(self):
        log = self.output / "learning.log"
        learning_log(log)
        curves = campaign.collect_learning(log, self.args)
        self.assertEqual(curves["review_status"], "pending_human_review")
        self.assertEqual(curves["final_validation_accuracy"], 0.01)
        self.assertNotIn("passed", curves)
        self.assertEqual(len(curves["evaluations"]), 3)

    def test_incomplete_duplicate_nonfinite_measurements_are_rejected(self):
        log = self.output / "learning.log"
        learning_log(log)
        valid = log.read_text()
        for text in (
            "\n".join(
                line for line in valid.splitlines() if "GNN_INPUT_CONTRACT " not in line
            ),
            valid.replace("Loss=2.0", "Loss=nan", 1),
            valid.replace("accuracy = 0.01", "accuracy = nan", 1),
            valid.replace("Avg Eval Loss: 3.0", "Avg Eval Loss: inf", 1),
            valid.replace("Step=60,", "Step=50,"),
            valid.rsplit("Evaluation completed successfully!", 1)[0],
            valid + "\n" + valid,
        ):
            with self.subTest(text=text[-80:]):
                log.write_text(text)
                with self.assertRaises(ValueError):
                    campaign.collect_learning(log, self.args)

    def test_real_full_validation_and_optimization_contract(self):
        for dataset, expected_steps in (("arxiv", 500), ("products", 1000)):
            args = campaign.parse_args(
                ["--dataset", dataset, "--output", str(self.output)]
            )
            base = campaign.shared_base(dataset)
            cfg = campaign.learning_config(
                base, args, campaign.autotune.candidate(0), self.output
            )
            init, fit = cfg["trainer"]["init"], cfg["trainer"]["fit"]
            self.assertEqual(init["loop"]["max_steps"], expected_steps)
            self.assertIsNone(init["loop"]["eval_steps"])
            self.assertEqual(init["optimizer"]["AdamW"]["eps"], 1e-6)
            self.assertEqual(init["optimizer"]["AdamW"]["betas"], [0.9, 0.999])
            self.assertEqual(fit["train_dataloader"]["split"], "train")
            self.assertEqual(fit["val_dataloader"]["split"], "valid")
            self.assertFalse(fit["val_dataloader"]["shuffle"])
            self.assertFalse(fit["val_dataloader"]["drop_last_batch"])
            self.assertFalse(fit["val_dataloader"]["use_fake_data"])
            self.assertEqual(fit["val_dataloader"]["static_batch_cache_size"], 0)
            self.assertIsNone(fit["val_dataloader"]["prefetch_factor"])

    def test_execution_failure_stops_before_tuning(self):
        with patch.object(
            campaign.autotune,
            "execute",
            return_value={"status": "failed", "returncode": 1},
        ) as execute:
            self.assertEqual(campaign.execute(self.args, self.base, self.provenance), 2)
            self.assertEqual(execute.call_count, 1)
        self.assertFalse((self.output / "tuning").exists())
        with self.assertRaisesRegex(ValueError, "Previous learning client"):
            campaign.execute(self.args, self.base, self.provenance)

    def test_complete_campaign_tunes_after_low_accuracy_and_exports_separate_routes(
        self,
    ):
        with patch.object(
            campaign.autotune, "execute", side_effect=self.fake_execute
        ) as execute:
            self.assertEqual(campaign.execute(self.args, self.base, self.provenance), 0)
            initial_count = execute.call_count
            self.assertGreater(initial_count, 6)
            self.assertEqual(campaign.execute(self.args, self.base, self.provenance), 0)
            self.assertEqual(execute.call_count, initial_count)
        folder = self.output / "handoff"
        selection = json.loads((folder / "selection.json").read_text())
        self.assertEqual(selection["review_status"], "pending_human_review")
        self.assertIsNone(selection["selected_learning"])
        configs = {
            name: yaml.safe_load((folder / name).read_text())
            for name in (
                "selected_csx.yaml",
                "selected_fixed_shape_gpu.yaml",
                "pyg_reference.yaml",
                "selected_csx_diagnostics.yaml",
                "selected_learning.yaml",
            )
        }
        fixed = configs["selected_fixed_shape_gpu.yaml"]
        pyg = configs["pyg_reference.yaml"]
        self.assertNotIn("backend", fixed["trainer"]["init"])
        self.assertIsNone(fixed["trainer"]["fit"]["train_dataloader"]["cache_fraction"])
        self.assertEqual(
            pyg["trainer"]["fit"]["train_dataloader"]["cache_fraction"], 0.0
        )
        self.assertFalse(
            fixed["trainer"]["init"]["model"]["task"]["compute_eval_metrics"]
        )
        self.assertIsNone(fixed["trainer"]["fit"]["val_dataloader"])
        self.assertEqual(
            fixed["trainer"]["init"]["optimizer"],
            self.base["trainer"]["init"]["optimizer"],
        )
        self.assertTrue(
            configs["selected_csx_diagnostics.yaml"]["trainer"]["fit"][
                "train_dataloader"
            ]["worker_diagnostics"]["enabled"]
        )
        self.assertEqual(
            configs["selected_learning.yaml"]["trainer"]["fit"]["val_dataloader"][
                "split"
            ],
            "valid",
        )

    def test_optional_selected_learning_collects_curves_without_accuracy_gate(self):
        args = campaign.parse_args(self.cli + ["--repeat-learning-with-selected"])
        with patch.object(campaign.autotune, "execute", side_effect=self.fake_execute):
            self.assertEqual(campaign.execute(args, self.base, self.provenance), 0)
        state = json.loads((self.output / "learning_campaign.json").read_text())
        self.assertEqual(state["selected"]["curves"]["final_validation_accuracy"], 0.01)
        self.assertEqual(state["selected"]["status"], "completed")

    def test_budget_resume_and_source_guard(self):
        args = deepcopy(self.args)
        args.budget_sec = args.trial_timeout_sec + 10
        with patch.object(
            campaign.autotune, "execute", side_effect=self.fake_execute
        ) as execute:
            self.assertEqual(campaign.execute(args, self.base, self.provenance), 2)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(campaign.execute(self.args, self.base, self.provenance), 0)
        with self.assertRaisesRegex(ValueError, "changed"):
            campaign.execute(self.args, self.base, {"source": "changed"})

    def test_preview_is_readonly_with_respect_to_jobs(self):
        with (
            patch.object(campaign.autotune, "environment") as environment,
            patch.object(campaign.autotune, "execute") as execute,
        ):
            self.assertEqual(campaign.main(self.cli + ["--dry-run"]), 0)
            environment.assert_not_called()
            execute.assert_not_called()
        plan = json.loads((self.output / "plan.json").read_text())
        self.assertEqual(plan["review_status"], "pending_human_review")
        self.assertEqual(
            plan["stages"],
            ["baseline_learning_curves", "input_autotune", "export_handoff"],
        )


if __name__ == "__main__":
    unittest.main()
