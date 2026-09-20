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
                self.assertIn("qsub", args)
                self.assertEqual(args[args.index("-l") + 1], "elapstim_req=24:00:00")
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
            self.assertEqual(args[args.index("--trial-timeout-sec") + 1], "1800")
            self.assertEqual(args[args.index("--budget-sec") + 1], "10800")

    def test_retry_limits_reach_payload_and_preserve_failure_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            submitted = subprocess.run(
                [str(PEGASUS / "submit_gpu_input_sweep_nqsv.sh"),
                 "--dataset", "products", "--backend", "fixed_shape", "--phase", "workers",
                 "--workers", "2", "--output", str(tmp / "output"), "--compile",
                 "--trial-timeout-sec", "10800", "--budget-sec", "33300",
                 "--walltime-hours", "10", "--record-resources", "--dry-run"],
                text=True, capture_output=True, check=True,
            )
            command = shlex.split(submitted.stdout)
            self.assertEqual(command[command.index("-l") + 1], "elapstim_req=10:00:00")
            env = dict(item.split("=", 1) for item in command[command.index("-v") + 1].split(","))
            env["PBS_O_WORKDIR"] = str(tmp)
            runner = tmp / "benchmark_scripts/pegasus/run_gpu_input_sweep.sh"
            runner.parent.mkdir(parents=True)
            runner.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(2)\n")
            runner.chmod(0o755)
            result = subprocess.run(["bash", str(PEGASUS / "run_gpu_input_sweep_nqsv.pbs")],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            args = json.loads(result.stdout)
            self.assertEqual(args[args.index("--trial-timeout-sec") + 1], "10800")
            self.assertEqual(args[args.index("--budget-sec") + 1], "33300")
            self.assertEqual(args[args.index("--workers") + 1], "2")
            self.assertEqual(args[args.index("--backend") + 1], "fixed_shape")
            self.assertIn("--compile", args)
            logs = list((tmp / "output/products/fixed_shape/workers/w2/resources").glob("*.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("runner_exit=2", logs[0].read_text())
            self.assertIn("MemTotal:", logs[0].read_text())

    def test_reject_inconsistent_time_limits_before_submission(self):
        invalid = [
            ["--trial-timeout-sec", "0"],
            ["--trial-timeout-sec", "01800"],
            ["--trial-timeout-sec", "11000"],
            ["--walltime-hours", "25"],
            ["--walltime-hours", "3", "--budget-sec", "10800"],
        ]
        for options in invalid:
            with self.subTest(options=options):
                result = subprocess.run(
                    [str(PEGASUS / "submit_gpu_input_sweep_nqsv.sh"), "--output", "/tmp/unused", *options, "--dry-run"],
                    text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("qsub", result.stdout)

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
