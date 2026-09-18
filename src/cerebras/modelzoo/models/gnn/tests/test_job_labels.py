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
            "gnn-model=old",
            "run=old",
            "study=old",
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
        self.assertTrue(actual["run"].startswith("sage-arxiv-sens-trial-"))
        self.assertRegex(actual["study"], r"^[a-f0-9]{10}$")
        values = config["trainer"]["init"]["backend"]["cluster_config"]["job_labels"]
        self.assertEqual(len(values), 4)
        self.assertEqual(set(actual), {"run", "study", "owner", "gnn-custom"})
        self.assertEqual(set(actual) & MANAGED_KEYS, {"run", "study"})
        self.assertLessEqual(len(actual["run"]), 60)
        apply_job_labels(
            config, mode="diagnostic", repeat=1, trial_dir=trial, study_dir=study
        )
        self.assertEqual(labels(config)["study"], actual["study"])
        self.assertTrue(labels(config)["run"].startswith("sage-arxiv-diag-trial-"))
        config["trainer"]["init"]["backend"]["cluster_config"]["job_labels"] = original[
            "trainer"
        ]["init"]["backend"]["cluster_config"]["job_labels"]
        self.assertEqual(config, original)
        self.assertNotEqual(label_value("a/b"), label_value("a-b"))
        self.assertNotEqual(label_value("x" * 100 + "a"), label_value("x" * 100 + "b"))
        self.assertNotEqual(
            study_label(self.root / "a/same", compact=True),
            study_label(self.root / "b/same", compact=True),
        )

    def test_cache_label_distinguishes_bypass_empty_and_partial_wrappers(self):
        for fraction, expected in (
            (None, "nopersist"),
            (0.0, "cache0"),
            (0.5, "cache0.5"),
            (1.0, "cache1"),
        ):
            with self.subTest(fraction=fraction):
                config = deepcopy(self.base)
                config["trainer"]["fit"]["train_dataloader"]["cache_fraction"] = (
                    fraction
                )
                apply_job_labels(
                    config,
                    mode="intervention",
                    repeat=1,
                    trial_dir=self.root / "control_feature_cache_r1",
                    study_dir=self.root,
                )
                self.assertTrue(labels(config)["run"].endswith(f"-{expected}-r1"))

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
        self.assertEqual(
            labels(first)["run"], "sage-arxiv-sens-w2-pf2-persist-cache1-r1"
        )
        self.assertEqual(
            labels(second)["run"], "sage-arxiv-sens-w2-pf2-persist-cache1-r2"
        )
        self.assertEqual(labels(first)["study"], labels(second)["study"])

    def test_campaign_labels_share_study_and_identify_nested_reference_trials(self):
        args = worker_campaign.parse_args(["--output", str(self.root / "campaign")])
        stages = worker_campaign.plan(args, self.base)
        all_labels = []
        for stage in stages:
            if "config" in stage:
                actual = labels(stage["config"])
                self.assertIn(
                    "-diag-" if stage["kind"] == "diagnostic" else "-control-",
                    actual["run"],
                )
                self.assertTrue(actual["run"].endswith(f"-r{stage['repeat']}"))
                self.assertEqual(
                    "-cache1-" in actual["run"], stage["control"] == "feature_cache"
                )
                if stage["control"] == "static_batch":
                    self.assertIn("-static1-", actual["run"])
                self.assertEqual(actual["owner"], "alice")
            else:
                command = stage["command"]
                nested = autotune.parse_args(command[3:])
                self.assertFalse(nested.detach)
                config = autotune.prepare_config(
                    autotune.get_backend("csx"),
                    nested,
                    self.base,
                    autotune.candidate(4, persistent=True),
                    Path(stage["directory"]) / "sensitivity_w04_p2_s1_r1/model",
                    440,
                )
                actual = labels(config)
                self.assertEqual(
                    actual["run"],
                    f"sage-arxiv-vs{stage['workers']}-sens-w4-pf2-persist-r1",
                )
            all_labels.append(actual)
            self.assertLessEqual(len(actual["run"]), 60)
        self.assertEqual(len({row["study"] for row in all_labels}), 1)
        self.assertEqual(len({row["run"] for row in all_labels}), len(stages))

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

    def test_learning_and_tuning_paths_fit_without_hash_fallback(self):
        cases = (
            ("learning", "baseline", "learning-baseline", None),
            ("learning", "selected", "learning-selected", None),
            ("learning", "handoff/selected_learning", "handoff-learning", None),
            ("selected", "handoff/selected_csx", "handoff-selected", None),
            ("diagnostic", "handoff/diagnostic", "handoff-diag", None),
            ("diagnostic", "diagnostic_w40", "diag", None),
            ("autotune", "workers_w40_p2_s1_r1", "workers", None),
            ("autotune", "tuning/loader_w40_p2_s1_r1", "tune-loader", 0.0),
            ("autotune", "tuning/confirm_w40_p2_s1_r1", "tune-confirm", 0.5),
            ("autotune", "confirm_w40_p2_s1_r1", "confirm", 1.0),
        )
        for mode, path, stage, fraction in cases:
            with self.subTest(path=path, fraction=fraction):
                config = deepcopy(self.base)
                loader = config["trainer"]["fit"]["train_dataloader"]
                loader.update(
                    dataset="products",
                    dataset_profiles={},
                    dataset_name="ogbn-products",
                    persistent_workers=True,
                    cache_fraction=fraction,
                )
                apply_job_labels(
                    config,
                    mode=mode,
                    repeat=1,
                    trial_dir=self.root / path,
                    study_dir=self.root,
                )
                value = labels(config)["run"]
                self.assertTrue(
                    value.startswith(f"sage-products-{stage}-w40-pf2-persist")
                )
                # Canonical paths fit naturally; truncation would end in a hash.
                self.assertTrue(value.endswith("-r1"))
                self.assertLessEqual(len(value), 60)

    def test_unknown_parent_and_inconsistent_candidate_do_not_alias_known_trials(self):
        values = []
        paths = (
            "sensitivity_w40_p2_s0_r1",
            "other/sensitivity_w40_p2_s0_r1",
            "sensitivity_w40_p2_s0_r2",  # Does not agree with supplied repeat.
            "x" * 100 + "a/sensitivity_w40_p2_s0_r1",
            "x" * 100 + "b/sensitivity_w40_p2_s0_r1",
        )
        for path in paths:
            config = deepcopy(self.base)
            apply_job_labels(
                config,
                mode="sensitivity",
                repeat=1,
                trial_dir=self.root / path,
                study_dir=self.root,
            )
            values.append(labels(config)["run"])
        self.assertEqual(values[0], "sage-arxiv-sens-w40-pf2-nopersist-r1")
        self.assertEqual(len(set(values)), len(paths))
        self.assertTrue(all(len(value) <= 60 for value in values))


if __name__ == "__main__":
    unittest.main()
