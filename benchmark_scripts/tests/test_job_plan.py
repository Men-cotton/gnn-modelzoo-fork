"""Catalog validation, safe argv, compatibility and all-or-nothing preparation."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cerebras.modelzoo.tools import benchmark_launcher, job_plan


class JobPlanTests(unittest.TestCase):
    def test_unknown_jobs_sets_and_options(self):
        for argv in (
            ["--job", "unknown"],
            ["--set", "unknown"],
            ["--job", "hpcasia", "--output", "/tmp/example", "--bogus"],
        ):
            with (
                self.subTest(argv=argv),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
                patch.object(job_plan, "execute_command") as execute,
            ):
                job_plan.main(argv)
            execute.assert_not_called()

    def test_catalog_rejects_bad_schema_and_values(self):
        original = job_plan.load_catalog()
        variants = []
        for key, value in [("kind", []), ("script", "../../tmp/a.pbs"), ("extra", 1)]:
            bad = copy.deepcopy(original)
            bad["jobs"]["fixed-shape"][key] = value
            variants.append(bad)
        for key, value in [("version", True), ("sets", {"bad": ["missing"]})]:
            bad = copy.deepcopy(original)
            bad[key] = value
            variants.append(bad)
        bad = copy.deepcopy(original)
        bad["jobs"]["hpcasia"]["seeds"] = [True]
        variants.append(bad)
        bad = copy.deepcopy(original)
        bad["jobs"]["csx-worker"]["module"] = 123
        variants.append(bad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            for bad in variants:
                path.write_text(json.dumps(bad))
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    job_plan.load_catalog(path)

    def test_yaml_inheritance_invalid_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "base.yaml").write_text("trainer: {}\n")
            path = tmp / "child.yaml"
            path.write_text("extends: base.yaml\n")
            self.assertEqual(job_plan.config_path(path), str(path))
            for text in (
                "[1, 2]",
                "trainer: []",
                "trainer: {}\ntrainer: {}",
                "extends: child.yaml",
                "trainer: [",
            ):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    job_plan.config_path(path)

    def test_qsub_argv_is_literal_and_values_are_unambiguous(self):
        script = job_plan.script_path(
            "benchmark_scripts/pegasus/run_fixed_shape_gpu_nqsv.pbs"
        )
        value = "/tmp/a b;$(touch nope)`x`.yaml"
        command = job_plan.qsub_command(script, {"FIXED_SHAPE_CONFIG": value})
        self.assertEqual(command[2], "FIXED_SHAPE_CONFIG=" + value)
        with patch.object(job_plan.subprocess, "run") as run:
            job_plan.execute_command(command)
            run.assert_called_once_with(command, cwd=job_plan.ROOT, shell=False)
        for value in ("a,b", "a\nb", "a\rb", "a\0b"):
            with self.assertRaises(ValueError):
                job_plan.qsub_command(script, {"CONFIG": value})

    def test_set_json_and_dry_run_do_not_execute(self):
        with (
            patch.object(job_plan, "execute_command") as execute,
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            benchmark_launcher.main(
                ["jobs", "--set", "hpcasia-all", "--output", "/tmp/preview", "--json"]
            )
        execute.assert_not_called()
        self.assertEqual(len(json.loads(out.getvalue())["commands"]), 30)
        with (
            patch.object(job_plan, "execute_command") as execute,
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            job_plan.main(["--job", "pyg", "--only", "gcn_ogbn_arxiv", "--dry-run"])
        execute.assert_not_called()
        self.assertEqual(len(out.getvalue().splitlines()), 3)

    def test_late_invalid_config_prevents_any_submission(self):
        original = job_plan.config_path

        def validate(path, **kwargs):
            if "products" in str(path):
                raise ValueError("late invalid config")
            return original(path, **kwargs)

        with (
            patch.object(job_plan, "config_path", side_effect=validate),
            patch.object(job_plan, "execute_command") as execute,
        ):
            with self.assertRaisesRegex(ValueError, "late invalid"):
                job_plan.main(["--job", "pyg"])
        execute.assert_not_called()

    def test_missing_scheduler_does_not_create_output(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(job_plan.shutil, "which", return_value=None),
        ):
            path = Path(tmp) / "output"
            with self.assertRaisesRegex(ValueError, "qsub not found"):
                job_plan.main(["--job", "hpcasia", "--output", str(path)])
            self.assertFalse(path.exists())

    def test_legacy_help_and_selector_boundary(self):
        wrapper = (
            job_plan.ROOT / "benchmark_scripts/pegasus/submit_fixed_shape_gpu_nqsv.sh"
        )
        result = subprocess.run(
            [str(wrapper), "--help"], text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--measure-input", result.stdout)
        result = subprocess.run(
            [str(wrapper), "--job", "hpcasia", "--output", "/tmp/unused", "--dry-run"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("qsub", result.stdout)

    def test_csx_catalog_and_legacy_dry_run(self):
        catalog = job_plan.load_catalog()
        self.assertEqual(len(catalog["jobs"]["csx-config"]["configs"]), 8)
        self.assertEqual(catalog["presets"]["worker_diagnostics"]["workers"], [2, 40])
        wrapper = job_plan.ROOT / "benchmark_scripts/cerebras/run_modelzoo.sh"
        result = subprocess.run(
            [str(wrapper), "--config-index", "1", "--dry-run"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--target_device CSX", result.stdout)
        self.assertIn(str(job_plan.CONFIGS.parent), result.stdout)
        with patch.object(job_plan, "execute_command") as execute:
            execute.return_value.returncode = 0
            job_plan.main(["--job", "csx-config", "--config-index", "1"])
        self.assertEqual(execute.call_args.kwargs["cwd"], job_plan.CONFIGS.parent)
        with self.assertRaises(ValueError):
            job_plan.main(["--job", "csx-config", "--config-index", "9", "--dry-run"])

    def test_driver_arguments_preserve_campaign_conditions(self):
        with (
            patch.object(job_plan, "execute_command") as execute,
            patch.object(job_plan.Path, "cwd", return_value=Path("/tmp/caller")),
        ):
            execute.return_value.returncode = 0
            self.assertEqual(
                job_plan.main(
                    [
                        "--job",
                        "csx-worker",
                        "--",
                        "--foreground",
                        "--dry-run",
                        "--budget-sec",
                        "120",
                    ]
                ),
                0,
            )
        self.assertEqual(execute.call_args.kwargs["cwd"], Path("/tmp/caller"))
        command = execute.call_args.args[0]
        self.assertEqual(
            command[-5:],
            ["--detach", "--foreground", "--dry-run", "--budget-sec", "120"],
        )


if __name__ == "__main__":
    unittest.main()
