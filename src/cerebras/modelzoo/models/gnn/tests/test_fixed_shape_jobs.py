"""Check the NQSV submission boundary with a local qsub stand-in."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[6]
SCRIPTS = ROOT / "benchmark_scripts/pegasus"


class FixedShapeJobTests(unittest.TestCase):
    def test_submit_uses_absolute_config_and_project_workdir(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            config = temp / "config with spaces.yaml"
            config.write_text("trainer: {}\n")
            qsub = temp / "qsub"
            qsub.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "print(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:]}))\n"
            )
            qsub.chmod(0o755)
            env = {**os.environ, "PATH": f"{temp}:{os.environ['PATH']}"}
            command = [
                str(SCRIPTS / "submit_fixed_shape_gpu_nqsv.sh"),
                "--config",
                config.name,
                "--compile",
                "--measure-neighbor-padding",
                "--measure-input",
            ]
            result = subprocess.run(
                command, cwd=temp, env=env, check=True, capture_output=True, text=True
            )
            submission = json.loads(result.stdout)
            source = Path(submission["cwd"])
            self.assertTrue(source.is_relative_to(ROOT / "model_dirs/job_runs"))
            self.assertEqual(source.name, "source")
            self.assertEqual(
                submission["args"][-1],
                str(source / "benchmark_scripts/pegasus/run_fixed_shape_gpu_nqsv.pbs"),
            )
            variables = dict(
                item.split("=", 1) for item in submission["args"][1].split(",")
            )
            self.assertEqual(variables["JOB_SOURCE_ROOT"], str(source))
            self.assertEqual(variables["PYTHONPATH"], str(source / "src"))
            frozen = Path(variables["FIXED_SHAPE_CONFIG"])
            self.assertEqual(frozen.read_text(), "trainer: {}\n")
            self.assertEqual(variables["FIXED_SHAPE_COMPILE"], "1")
            self.assertEqual(variables["FIXED_SHAPE_MEASURE_INPUT"], "1")
            self.assertEqual(variables["FIXED_SHAPE_MEASURE_NEIGHBOR_PADDING"], "1")
            # No qsub output: the dry run prints the command without executing it.
            result = subprocess.run(
                command + ["--dry-run"],
                cwd=temp,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("qsub", result.stdout)
            self.assertNotIn('"cwd":', result.stdout)

    def test_missing_config_and_ambiguous_qsub_value_fail(self):
        for name in ("submit_fixed_shape_gpu_nqsv.sh", "run_fixed_shape_gpu.sh"):
            result = subprocess.run(
                [str(SCRIPTS / name), "--dry-run"], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 2)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "a,b.yaml"
            config.write_text("trainer: {}\n")
            result = subprocess.run(
                [
                    str(SCRIPTS / "submit_fixed_shape_gpu_nqsv.sh"),
                    "--config",
                    str(config),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("commas", result.stderr)


if __name__ == "__main__":
    unittest.main()
