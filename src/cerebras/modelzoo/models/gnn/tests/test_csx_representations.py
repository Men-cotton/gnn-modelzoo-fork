"""Representation jobs share the ordinary source, environment and worker contracts."""

import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cerebras.modelzoo.tools.job_plan import ROOT
from cerebras.modelzoo.models.gnn.tools.csx_backend import backend_options
from cerebras.modelzoo.tools import job_plan, job_source


class CSXRepresentationTests(unittest.TestCase):
    def test_catalog_defaults_survive_detached_launch(self):
        catalog = job_plan.load_catalog()
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("csx-shared-tree", "csx-history-pilot", "csx-history"):
                job = catalog["jobs"][name]
                driver = importlib.import_module(job["module"])
                with patch.object(driver, "ROOT", Path(temporary)):
                    args = driver.parse_args(job["arguments"])
                    self.assertEqual(args.backend, "CSX")
                    self.assertTrue(args.detach)
                    self.assertTrue(
                        args.output.is_relative_to(Path(temporary) / "model_dirs")
                    )
                    self.assertFalse(args.output.exists())
                    with patch.object(
                        driver.benchmark_launcher, "launch", return_value=0
                    ) as launch:
                        driver.main([*job["arguments"], "--backend", "CPU"])
                    command, launcher_dir = launch.call_args.args
                    child = driver.parse_args(command[4:])
                    self.assertFalse(child.detach)
                    self.assertEqual(
                        launcher_dir,
                        child.output.with_name(child.output.name + "-launcher"),
                    )
                    self.assertFalse(child.output.exists())

    def test_reused_inputs_reject_preparation_overrides(self):
        catalog = job_plan.load_catalog()
        for name in ("csx-shared-tree", "csx-history-pilot", "csx-history"):
            driver = importlib.import_module(catalog["jobs"][name]["module"])
            with self.subTest(job=name), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    driver.parse_args(["--input", "prepared", "--dropout", "0"])
                self.assertEqual(failure.exception.code, 2)

    def test_history_refresh_device_reaches_later_epochs(self):
        from cerebras.modelzoo.models.gnn.tools.history_embedding import campaign

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            campaign.main(
                [
                    "--input",
                    "prepared",
                    "--refresh-device",
                    "cuda",
                    "--epochs",
                    "2",
                    "--dry-run",
                ]
            )
        commands = [shlex.split(line) for line in stdout.getvalue().splitlines()]
        refresh = next(command for command in commands if "refresh" in command)
        self.assertEqual(refresh[refresh.index("--refresh-device") + 1], "cuda")

    def test_history_dry_run_inherits_python_and_preserves_paths(self):
        from cerebras.modelzoo.models.gnn.tools.history_embedding import pilot

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                pilot.main(
                    [
                        "--input",
                        "input with spaces",
                        "--output",
                        str(output),
                        "--dry-run",
                    ]
                )
            commands = [shlex.split(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual(len(commands), 2)
            for command in commands:
                self.assertEqual(command[:3], [sys.executable, "-u", "-m"])
                self.assertEqual(
                    command[command.index("--bundle") + 1],
                    str(Path("input with spaces").resolve() / "bundle.pt"),
                )
                self.assertEqual(
                    command[command.index("--compile-dir") + 1], str(output / "compile")
                )
            self.assertFalse(output.exists())

    def test_source_handoff_preserves_input_paths(self):
        argv = job_source.absolute_arguments(
            [
                "--input",
                "input with spaces",
                "--data-dir=data/datasets",
                "--compile-dir",
                "compile cache",
            ]
        )
        self.assertEqual(
            argv,
            [
                "--input",
                str(Path("input with spaces").absolute()),
                "--data-dir=" + str(Path("data/datasets").absolute()),
                "--compile-dir",
                str(Path("compile cache").absolute()),
            ],
        )

    def test_csx_uses_existing_defaults_and_pinned_src(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "output"
            manifest = base / "run/manifest.json"
            for directory in (
                output,
                manifest.parent,
                base / "original",
                base / "input",
            ):
                directory.mkdir(parents=True)
            env = dict(
                JOB_SOURCE_ROOT=str(ROOT),
                JOB_SOURCE_ORIGINAL_ROOT=str(base / "original"),
                JOB_RUN_MANIFEST=str(manifest),
            )
            with patch.dict(os.environ, env):
                options = backend_options(
                    output, None, [base / "input/bundle.pt"], True
                )
            cluster = options["cluster_config"]
            self.assertEqual(cluster.num_csx, 1)
            self.assertEqual(cluster.num_workers_per_csx, 1)
            self.assertEqual(cluster.job_time_sec, 7200)
            self.assertIsNone(cluster.mgmt_address)
            self.assertEqual(cluster.python_paths, [str(ROOT / "src")])
            mounts = [mount.path for mount in cluster.mount_dirs]
            self.assertIn(str(base / "input"), mounts)
            self.assertIn(str(manifest.parent), mounts)
            self.assertEqual(options["compile_dir"], str(output / "compile"))
            self.assertTrue(options["compile_only"])
            self.assertTrue((output / "compile").is_dir())

    def test_worker_factories_import_without_checkout_root_on_path(self):
        code = """import json
from cerebras.modelzoo.models.gnn.tools.history_embedding.hardware import input_fn as history
from cerebras.modelzoo.models.gnn.tools.feature_sharing.core import input_fn as shared
print(json.dumps([history.__module__, shared.__module__]))
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=temporary,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
        self.assertTrue(
            all(
                name.startswith("cerebras.modelzoo.")
                for name in json.loads(result.stdout)
            )
        )


if __name__ == "__main__":
    unittest.main()
