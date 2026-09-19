"""Exercise NQSV variable serialization without submitting a GPU job."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PEGASUS = ROOT / "benchmark_scripts" / "pegasus"


class InputSweepSubmissionTest(unittest.TestCase):
    def test_worker_list_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            qsub = tmp / "qsub"
            qsub.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "assert sys.argv[1] == '-v'\n"
                "assert not any(c.isspace() for c in sys.argv[2])\n"
                "print(json.dumps(sys.argv[1:]))\n"
            )
            qsub.chmod(0o755)
            env = dict(os.environ, PATH=str(tmp) + os.pathsep + os.environ["PATH"])
            runner = tmp / "benchmark_scripts/pegasus/run_gpu_input_sweep.sh"
            runner.parent.mkdir(parents=True)
            runner.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            runner.chmod(0o755)
            for worker_args, expected in [
                ([], "2 4 8 12 16 24 32 40 48 64"),
                (["--workers", "4\t16  40"], "4 16 40"),
            ]:
                with self.subTest(workers=expected):
                    result = subprocess.run(
                        [str(PEGASUS / "submit_gpu_input_sweep_nqsv.sh"),
                         "--dataset", "products", "--backend", "both",
                         "--output", str(tmp / "output"), "--compile", *worker_args],
                        env=env, text=True, capture_output=True, check=True,
                    )
                    args = json.loads(result.stdout)
                    values = dict(item.split("=", 1) for item in args[1].split(","))
                    self.assertEqual(values["GPU_SWEEP_WORKERS"], expected.replace(" ", ":"))
                    result = subprocess.run(
                        ["bash", str(PEGASUS / "run_gpu_input_sweep_nqsv.pbs")],
                        env=dict(env, **values, PBS_O_WORKDIR=str(tmp)),
                        text=True, capture_output=True, check=True,
                    )
                    args = json.loads(result.stdout)
                    self.assertEqual(args[args.index("--workers") + 1], expected)
                    self.assertEqual(args[args.index("--dataset") + 1], "products")
                    self.assertIn("--compile", args)

    def test_reject_ambiguous_output_paths(self):
        for output in ("/tmp/gpu output", "/tmp/gpu,output", "/tmp/gpu\noutput"):
            with self.subTest(output=output):
                result = subprocess.run(
                    [str(PEGASUS / "submit_gpu_input_sweep_nqsv.sh"),
                     "--backend", "pyg", "--output", output, "--dry-run"],
                    text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("whitespace", result.stderr)


if __name__ == "__main__":
    unittest.main()
