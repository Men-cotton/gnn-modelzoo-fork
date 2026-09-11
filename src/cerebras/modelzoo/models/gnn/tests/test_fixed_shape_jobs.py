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
            ]
            result = subprocess.run(
                command, cwd=temp, env=env, check=True, capture_output=True, text=True
            )
            submission = json.loads(result.stdout)
            self.assertEqual(submission["cwd"], str(ROOT))
            self.assertEqual(
                submission["args"],
                [
                    "-v",
                    f"FIXED_SHAPE_CONFIG={config},FIXED_SHAPE_COMPILE=1",
                    str(SCRIPTS / "run_fixed_shape_gpu_nqsv.pbs"),
                ],
            )
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
