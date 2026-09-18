"""CSX trial labels are SDK-valid and survive real study/configuration paths."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from cerebras.appliance.cluster_config import ClusterConfig
from cerebras.modelzoo.models.gnn.tools import autotune, worker_campaign
from cerebras.modelzoo.models.gnn.tools.job_labels import (
    MANAGED_KEYS,
    apply_job_labels,
    label_value,
    study_label,
)
from test_autotune import write_log


def labels(config):
    values = config["trainer"]["init"]["backend"]["cluster_config"]["job_labels"]
    # Exercise the installed SDK's actual descriptor, not a copy of its regex.
    assert ClusterConfig(job_labels=values).job_labels == values
    return dict(value.split("=", 1) for value in values)


class JobLabelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = autotune.load_params_file(
            autotune.GNN / "configs/autotune/arxiv_w40.yaml"
        )
        self.base["trainer"]["init"]["backend"]["cluster_config"]["job_labels"] = [
            "owner=alice",
            "gnn-study=old",
            "gnn-study=duplicate",
            "gnn-custom=keep",
        ]

    def test_sdk_valid_stable_labels_preserve_user_metadata(self):
        config = deepcopy(self.base)
        original = deepcopy(config)
        study = self.root / ("研究 study " + "x" * 100)
        trial = study / ("trial / " + "y" * 100)
        apply_job_labels(
            config, mode="sensitivity", repeat=12, trial_dir=trial, study_dir=study
        )
        actual = labels(config)
        self.assertEqual(actual["owner"], "alice")
        self.assertEqual(actual["gnn-custom"], "keep")
        self.assertEqual(actual["gnn-model"], "graphsage")
        self.assertEqual(actual["gnn-dataset"], "ogbn-arxiv")
        self.assertEqual(actual["gnn-cache"], "none")
        self.assertEqual(actual["gnn-workers"], "40")
        self.assertEqual(actual["gnn-repeat"], "12")
        values = config["trainer"]["init"]["backend"]["cluster_config"]["job_labels"]
        self.assertEqual(len(values), len(MANAGED_KEYS) + 2)
        self.assertTrue(all(1 <= len(v) <= 63 for v in actual.values()))
        apply_job_labels(
            config, mode="diagnostic", repeat=1, trial_dir=trial, study_dir=study
        )
        self.assertEqual(labels(config)["gnn-study"], actual["gnn-study"])
        self.assertEqual(labels(config)["gnn-mode"], "diagnostic")
        config["trainer"]["init"]["backend"]["cluster_config"]["job_labels"] = original[
            "trainer"
        ]["init"]["backend"]["cluster_config"]["job_labels"]
        self.assertEqual(config, original)
        self.assertNotEqual(label_value("a/b"), label_value("a-b"))
        self.assertNotEqual(label_value("x" * 100 + "a"), label_value("x" * 100 + "b"))
        self.assertNotEqual(
            study_label(self.root / "a/same"), study_label(self.root / "b/same")
        )

    def test_cache_label_distinguishes_bypass_empty_and_partial_wrappers(self):
        for fraction, expected in (
            (None, "none"),
            (0.0, "zero"),
            (0.5, "partial-0.5"),
            (1.0, "full"),
        ):
            with self.subTest(fraction=fraction):
                config = deepcopy(self.base)
                config["trainer"]["fit"]["train_dataloader"][
                    "cache_fraction"
                ] = fraction
                apply_job_labels(
                    config,
                    mode="intervention",
                    repeat=1,
                    trial_dir=self.root / "trial",
                    study_dir=self.root,
                )
                self.assertEqual(labels(config)["gnn-cache"], expected)

    def test_preview_matches_executed_sensitivity_trial_and_repeat(self):
        output = self.root / "study"
        cli = [
            "--mode",
            "sensitivity",
            "--dataset",
            "arxiv",
            "--wsc-workers",
            "1",
            "--workers",
            "2",
            "--cache",
            "full",
            "--output",
            str(output),
            "--budget-sec",
            "40000",
        ]
        with (
            patch.object(
                autotune, "load_params_file", return_value=deepcopy(self.base)
            ),
            patch.object(autotune, "environment") as environment,
            patch.object(autotune, "execute") as execute,
        ):
            self.assertEqual(autotune.main(cli + ["--dry-run"]), 0)
            environment.assert_not_called()
            execute.assert_not_called()
        preview = yaml.safe_load((output / "preview_w02.yaml").read_text())
        args = autotune.parse_args(cli)
        study = autotune.Study(args, self.base, output, {"test": "labels"})

        def execute(command, log, timeout):
            write_log(log, end=440)
            return {"status": "completed", "returncode": 0}

        with patch.object(autotune, "execute", side_effect=execute):
            for repeat in (1, 2):
                study.trial(
                    autotune.candidate(2, persistent=True), "sensitivity", repeat, 400
                )
        first = yaml.safe_load(
            (output / "sensitivity_w02_p2_s1_r1/params.yaml").read_text()
        )
        second = yaml.safe_load(
            (output / "sensitivity_w02_p2_s1_r2/params.yaml").read_text()
        )
        self.assertEqual(labels(preview), labels(first))
        self.assertEqual(labels(first)["gnn-cache"], "full")
        self.assertEqual(labels(second)["gnn-repeat"], "2")
        self.assertNotEqual(labels(first)["gnn-trial"], labels(second)["gnn-trial"])
        self.assertEqual(labels(first)["gnn-study"], labels(second)["gnn-study"])

    def test_campaign_labels_share_study_and_identify_nested_reference_trials(self):
        args = worker_campaign.parse_args(["--output", str(self.root / "campaign")])
        stages = worker_campaign.plan(args, self.base)
        all_labels = []
        for stage in stages:
            if "config" in stage:
                actual = labels(stage["config"])
                self.assertEqual(actual["gnn-mode"], stage["kind"])
                self.assertEqual(actual["gnn-repeat"], str(stage["repeat"]))
                self.assertEqual(
                    actual["gnn-cache"],
                    "full" if stage["control"] == "feature_cache" else "none",
                )
                self.assertEqual(actual["owner"], "alice")
            else:
                command = stage["command"]
                self.assertIn("--foreground", command)
                nested = autotune.parse_args(
                    ["--mode", "sensitivity"]
                    + [arg for arg in command[2:] if arg != "--foreground"]
                )
                config = autotune.prepare_config(
                    autotune.get_backend("csx"),
                    nested,
                    self.base,
                    autotune.candidate(4, persistent=True),
                    Path(stage["directory"]) / "sensitivity_w04_p2_s1_r1/model",
                    440,
                )
                actual = labels(config)
                self.assertEqual(actual["gnn-mode"], "sensitivity")
            all_labels.append(actual)
        self.assertEqual(len({row["gnn-study"] for row in all_labels}), 1)
        self.assertEqual(len({row["gnn-trial"] for row in all_labels}), len(stages))

    def test_pyg_configuration_does_not_acquire_csx_backend(self):
        args = autotune.parse_args(
            [
                "--backend",
                "pyg",
                "--dataset",
                "arxiv",
                "--output",
                str(self.root),
                "--budget-sec",
                "40000",
            ]
        )
        config = autotune.prepare_config(
            autotune.get_backend("pyg"),
            args,
            self.base,
            autotune.candidate(2),
            self.root / "model",
            440,
        )
        self.assertNotIn("backend", config["trainer"]["init"])


if __name__ == "__main__":
    unittest.main()
