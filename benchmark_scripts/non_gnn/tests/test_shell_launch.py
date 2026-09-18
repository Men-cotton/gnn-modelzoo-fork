"""Public shell failure handling without CUDA, a scheduler or cluster jobs."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class ShellLaunchTests(unittest.TestCase):
    def test_gpu_module_failure_records_result_without_starting_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "benchmark_scripts/pegasus"
            scripts.mkdir(parents=True)
            wrapper = scripts / "run_non_gnn_gpu.sh"
            wrapper.write_text((ROOT / wrapper.relative_to(root)).read_text())
            (root / "common.sh").write_text("")
            (scripts / "gpu_env.sh").write_text(
                "load_cuda_module() { echo 'test module unavailable' >&2; return 17; }\n"
                "require_cuda_toolkit() { return 0; }\n"
            )
            run_dir = root / "run with spaces"
            run_dir.mkdir()
            config = run_dir / "params.yaml"
            config.write_text("test: true\n")
            marker = root / "training-started"
            launch = {
                "backend": "GPU",
                "gpu_implementation": "native",
                "profile": "bert_large_msl128",
                "effective_batch_size": 2,
                "sequence_length": 128,
                "max_optimizer_steps": 4,
                "warmup_steps": 1,
                "params_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
            }
            (run_dir / "launch.json").write_text(json.dumps(launch))
            # Route the wrapper's uv command through this prepared interpreter;
            # execute the real record-only path, never install dependencies.
            uv = root / "uv"
            uv.write_text(
                f"#!{sys.executable}\nimport os, sys\n"
                "args = sys.argv[1:]\n"
                f"os.execv({sys.executable!r}, [{sys.executable!r}, {str(ROOT / 'benchmark_scripts/non_gnn/run.py')!r}, *args[args.index('--backend'):]])\n"
            )
            uv.chmod(0o755)
            result = subprocess.run(
                ["bash", str(wrapper), "--config", str(config)],
                env={
                    **os.environ,
                    "PATH": str(root) + os.pathsep + os.environ["PATH"],
                    "PYTHONPATH": str(ROOT / "src"),
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
            self.assertFalse(marker.exists())
            status = json.loads((run_dir / "client_status.json").read_text())
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["exit_code"], 17)
            self.assertIn("preflight", status["error"])
            measurement = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(measurement["measurement_status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
