"""Seed ordering, independent repeats and failure/resume behavior without CSX."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from cerebras.modelzoo.models.gnn.tools import hpcasia_campaign as campaign
from test_autotune import write_log
from test_learning_campaign import learning_log


class HPCAsiaCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "study"
        self.output.mkdir()
        self.cli = [
            "--output",
            str(self.output),
            "--learning-steps",
            "60",
            "--measure-steps",
            "40",
        ]
        self.args = campaign.parse_args(self.cli)
        self.runs = campaign.plan(
            self.args, campaign.learning_campaign.shared_base("arxiv")
        )
        self.provenance = {"source": "synthetic test"}

    def fake_execute(self, command, log, timeout):
        config = yaml.safe_load(Path(command[command.index("fit") + 1]).read_text())
        init = config["trainer"]["init"]
        if init["loop"].get("eval_frequency"):
            learning_log(log, init["loop"]["max_steps"], init["loop"]["eval_frequency"])
        else:
            write_log(log, end=init["loop"]["max_steps"])
        return {"status": "completed", "returncode": 0}

    def test_seed_major_order_and_cache_is_only_control_difference(self):
        self.assertEqual([r["seed"] for r in self.runs], [42] * 5 + [43] * 5 + [44] * 5)
        labels = []
        for index, row in enumerate(self.runs):
            init, fit = (
                row["config"]["trainer"]["init"],
                row["config"]["trainer"]["fit"],
            )
            self.assertEqual(init["seed"], row["seed"])
            self.assertEqual(fit["train_dataloader"]["sampler_seed"], row["seed"])
            self.assertEqual(
                init["backend"]["cluster_config"]["num_workers_per_csx"], 1
            )
            labels.append(init["backend"]["cluster_config"]["job_labels"][0])
            for key, value in campaign.KNOBS.items():
                self.assertEqual(fit["train_dataloader"][key], value)
            self.assertEqual(fit["train_dataloader"]["static_batch_cache_size"], 0)
            self.assertFalse(fit["train_dataloader"]["drop_last_batch"])
            self.assertIsNone(fit["ckpt_path"])
            if index % 5 == 0:
                self.assertEqual(row["kind"], "learning")
                self.assertEqual(fit["val_dataloader"]["sampler_seed"], row["seed"])
                self.assertEqual(fit["val_dataloader"]["split"], "valid")
                self.assertFalse(fit["val_dataloader"]["shuffle"])
                self.assertIsNone(init["loop"]["eval_steps"])
                self.assertTrue(init["model"]["task"]["compute_eval_metrics"])
            else:
                self.assertIsNone(fit["val_dataloader"])
                self.assertIsNone(init["loop"]["eval_frequency"])
                self.assertFalse(init["model"]["task"]["compute_eval_metrics"])
                self.assertEqual(
                    row["kind"], "cache" if index % 5 == 4 else "throughput"
                )
                self.assertEqual(row["repeat"], 1 if index % 5 == 4 else index % 5)
                self.assertEqual(
                    fit["train_dataloader"]["cache_fraction"],
                    1.0 if index % 5 == 4 else None,
                )
        self.assertEqual(len(set(labels)), 15)
        self.assertEqual(len({r["directory"] for r in self.runs}), 15)
        cached = deepcopy(self.runs[4]["config"])
        uncached = deepcopy(self.runs[1]["config"])
        cached["trainer"]["fit"]["train_dataloader"]["cache_fraction"] = None
        for config in (cached, uncached):
            config["trainer"]["init"].pop("model_dir")
            config["trainer"]["init"]["backend"]["cluster_config"].pop("job_labels")
        self.assertEqual(cached, uncached)

    def test_full_execution_records_curves_and_all_repeats_and_resume_skips(self):
        with patch.object(
            campaign.autotune, "execute", side_effect=self.fake_execute
        ) as execute:
            self.assertEqual(campaign.execute(self.args, self.runs, self.provenance), 0)
            self.assertEqual(execute.call_count, 15)
            self.assertEqual(campaign.execute(self.args, self.runs, self.provenance), 0)
            self.assertEqual(execute.call_count, 15)
        summary = json.loads((self.output / "summary.json").read_text())
        for row in summary["seeds"]:
            self.assertEqual(row["final_validation_accuracy"], 0.01)
            self.assertEqual(row["throughput_repeats_completed"], 3)
            self.assertEqual(row["cache_to_uncached_ratio"], 1.0)
        self.assertEqual(len(list(self.output.glob("seed_*/*/result.json"))), 15)
        self.assertEqual(
            len(list(self.output.glob("seed_*/learning_r1/learning_curves.json"))), 3
        )

    def test_failed_client_stops_and_cannot_be_replayed(self):
        with patch.object(
            campaign.autotune, "execute", return_value={"status": "timeout"}
        ) as execute:
            self.assertEqual(campaign.execute(self.args, self.runs, self.provenance), 2)
            self.assertEqual(execute.call_count, 1)
        with self.assertRaisesRegex(ValueError, "Previous client"):
            campaign.execute(self.args, self.runs, self.provenance)
        result = json.loads(
            (self.output / "seed_42/learning_r1/result.json").read_text()
        )
        self.assertEqual(result["status"], "timeout")

    def test_budget_resume_and_source_guard(self):
        args = deepcopy(self.args)
        args.budget_sec = args.trial_timeout_sec + 10
        with patch.object(
            campaign.autotune, "execute", side_effect=self.fake_execute
        ) as execute:
            self.assertEqual(campaign.execute(args, self.runs, self.provenance), 2)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(campaign.execute(self.args, self.runs, self.provenance), 0)
            self.assertEqual(execute.call_count, 15)
        with self.assertRaisesRegex(ValueError, "changed"):
            campaign.execute(self.args, self.runs, {"source": "different"})

    def test_stability_flag_is_retained_without_dropping_run(self):
        measure = campaign.CSXBackend.measure

        def unstable(*args):
            result = measure(*args)
            result["half_window_check"]["within_tolerance"] = False
            return result

        with (
            patch.object(campaign.autotune, "execute", side_effect=self.fake_execute),
            patch.object(campaign.CSXBackend, "measure", side_effect=unstable),
        ):
            self.assertEqual(campaign.execute(self.args, self.runs, self.provenance), 0)
        state = json.loads((self.output / "campaign.json").read_text())
        self.assertFalse(
            state["runs"][1]["measurement"]["half_window_check"]["within_tolerance"]
        )
        self.assertEqual(state["runs"][1]["status"], "completed")

    def test_dry_run_materializes_all_configs_without_launch(self):
        with (
            patch.object(campaign.autotune, "execute") as execute,
            patch.object(campaign.autotune, "environment") as environment,
            patch.object(campaign.benchmark_launcher, "launch") as launch,
        ):
            self.assertEqual(campaign.main(self.cli + ["--detach", "--dry-run"]), 0)
            execute.assert_not_called()
            environment.assert_not_called()
            launch.assert_not_called()
        self.assertEqual(len(list(self.output.glob("seed_*/*/params.yaml"))), 15)
        with self.assertRaisesRegex(ValueError, "fresh output"):
            campaign.main(self.cli)

    def test_products_defaults_and_validation_match_dataset(self):
        args = campaign.parse_args(
            ["--dataset", "products", "--output", str(self.output)]
        )
        runs = campaign.plan(args, campaign.learning_campaign.shared_base("products"))
        self.assertEqual((args.learning_steps, args.eval_every), (1000, 40))
        self.assertEqual(len(runs), 15)
        for row in runs:
            fit = row["config"]["trainer"]["fit"]
            self.assertEqual(fit["train_dataloader"]["dataset"], "ogbn_products")
            self.assertEqual(
                row["config"]["trainer"]["init"]["model"]["architecture"]["n_class"], 47
            )
            if row["kind"] == "learning":
                self.assertEqual(fit["val_dataloader"]["dataset"], "ogbn_products")
                self.assertEqual(fit["val_dataloader"]["sampler_seed"], row["seed"])

    def test_invalid_learning_record_stops_before_throughput(self):
        def incomplete(command, log, timeout):
            log.write_text("Training completed successfully!\n")
            return {"status": "completed", "returncode": 0}

        with patch.object(
            campaign.autotune, "execute", side_effect=incomplete
        ) as execute:
            self.assertEqual(campaign.execute(self.args, self.runs, self.provenance), 2)
            self.assertEqual(execute.call_count, 1)
        result = json.loads(
            (self.output / "seed_42/learning_r1/result.json").read_text()
        )
        self.assertEqual(result["status"], "invalid_measurement")

    def test_detach_uses_existing_launcher_and_concrete_output(self):
        with patch.object(
            campaign.benchmark_launcher, "launch", return_value=0
        ) as launch:
            self.assertEqual(campaign.main(self.cli + ["--detach"]), 0)
        command = launch.call_args.args[0]
        self.assertIn("--foreground", command)
        self.assertEqual(command[-2:], ["--output", str(self.output)])


if __name__ == "__main__":
    unittest.main()
