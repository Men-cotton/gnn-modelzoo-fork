"""Exercise split NQSV submission without submitting a GPU job."""
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PEGASUS = ROOT / "benchmark_scripts" / "pegasus"


class InputSweepSubmissionTest(unittest.TestCase):
    def test_submits_one_job_per_dataset_backend_and_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            result = subprocess.run(
                [str(PEGASUS / "submit_gpu_input_sweep_nqsv.sh"),
                 "--dataset", "all", "--backend", "both", "--phase", "workers",
                 "--workers", "4\t16  40", "--output", str(tmp / "output"),
                 "--compile", "--dry-run"],
                text=True, capture_output=True, check=True,
            )
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            self.assertEqual(len(lines), 12)
            jobs = []
            for line in lines:
                args = shlex.split(line)
                self.assertEqual(args[args.index("-v") - 1], "qsub")
                values = {
                    item.split("=", 1)[0]: item.split("=", 1)[1]
                    for item in args[args.index("-v") + 1].split(",")
                }
                jobs.append(values)
            self.assertEqual(
                {(j["GPU_SWEEP_DATASET"], j["GPU_SWEEP_BACKEND"], j["GPU_SWEEP_WORKER"])
                 for j in jobs},
                {(d, b, w) for d in ("arxiv", "products")
                 for b in ("fixed_shape", "pyg") for w in ("4", "16", "40")},
            )
            self.assertTrue(all(j["GPU_SWEEP_PHASE"] == "workers" for j in jobs))
            self.assertTrue(all(j["GPU_SWEEP_COMPILE"] == "1" for j in jobs))

    def test_pbs_passes_one_worker_to_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runner = tmp / "benchmark_scripts/pegasus/run_gpu_input_sweep.sh"
            runner.parent.mkdir(parents=True)
            runner.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            runner.chmod(0o755)
            env = {
                "PBS_O_WORKDIR": str(tmp),
                "GPU_SWEEP_DATASET": "products",
                "GPU_SWEEP_OUTPUT": str(tmp / "output"),
                "GPU_SWEEP_WORKER": "16",
                "GPU_SWEEP_PHASE": "workers",
                "GPU_SWEEP_BACKEND": "pyg",
                "GPU_SWEEP_COMPILE": "1",
            }
            result = subprocess.run(
                ["bash", str(PEGASUS / "run_gpu_input_sweep_nqsv.pbs")],
                env=env, text=True, capture_output=True, check=True,
            )
            args = json.loads(result.stdout)
            self.assertEqual(args[args.index("--workers") + 1], "16")
            self.assertEqual(args[args.index("--dataset") + 1], "products")
            self.assertEqual(args[args.index("--phase") + 1], "workers")
            self.assertEqual(args[args.index("--backend") + 1], "pyg")
            self.assertEqual(args[args.index("--worker-suffix") + 1], "w16")
            self.assertIn(str(tmp / "output" / "products"), args)

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
