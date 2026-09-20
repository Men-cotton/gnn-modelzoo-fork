"""Check HPC Asia job expansion and PBS dispatch without a scheduler or GPU."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PEGASUS = ROOT / "benchmark_scripts/pegasus"


class HPCAsiaSubmissionTests(unittest.TestCase):
    def test_thirty_unique_jobs_and_four_hour_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "runs")
            p = subprocess.run(
                [
                    "bash",
                    str(PEGASUS / "submit_hpcasia_nqsv.sh"),
                    "--output",
                    output,
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            jobs = []
            for line in p.stdout.splitlines():
                args = shlex.split(line)
                self.assertEqual(args[args.index("-l") + 1], "elapstim_req=04:00:00")
                job = dict(
                    v.split("=", 1) for v in args[args.index("-v") + 1].split(",")
                )
                self.assertEqual(job["HPCASIA_TIMEOUT"], "14100")
                self.assertEqual(job["HPCASIA_COMPILE"], "1")
                jobs.append((job["HPCASIA_DATASET"], job["HPCASIA_RUN_ID"]))
            self.assertEqual(len(jobs), 30)
            self.assertEqual(len(set(jobs)), 30)
            self.assertEqual(sum("learning_" in run for _, run in jobs), 6)
            self.assertEqual(sum("throughput_" in run for _, run in jobs), 18)
            self.assertEqual(sum("cache_" in run for _, run in jobs), 6)
            self.assertFalse(Path(output).exists())

    def test_three_hour_dataset_subset(self):
        p = subprocess.run(
            [
                "bash",
                str(PEGASUS / "submit_hpcasia_nqsv.sh"),
                "--output",
                "/tmp/preview",
                "--dataset",
                "products",
                "--hours",
                "3",
                "--no-compile",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(len(p.stdout.splitlines()), 15)
        self.assertIn("elapstim_req=03:00:00", p.stdout)
        self.assertIn("HPCASIA_TIMEOUT=10500", p.stdout)
        self.assertIn("HPCASIA_COMPILE=0", p.stdout)

    def test_pbs_passes_single_run_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benchmark_scripts/pegasus").mkdir(parents=True)
            (root / "common.sh").write_text("")
            (root / "benchmark_scripts/pegasus/gpu_env.sh").write_text(
                "load_cuda_module() { :; }\nrequire_cuda_toolkit() { :; }\n"
            )
            uv = root / "uv"
            uv.write_text(
                '#!/usr/bin/env python3\nimport json,sys,os\nprint(json.dumps([sys.argv[1:],os.environ.get("NO_COMPILE")]))\n'
            )
            uv.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{root}:" + os.environ["PATH"],
                "PBS_O_WORKDIR": str(root),
                "HPCASIA_DATASET": "products",
                "HPCASIA_RUN_ID": "seed_44/cache_r1",
                "HPCASIA_OUTPUT": str(root / "output"),
                "HPCASIA_TIMEOUT": "14100",
                "NO_COMPILE": "1",
                "HPCASIA_COMPILE": "1",
            }
            p = subprocess.run(
                ["bash", str(PEGASUS / "run_hpcasia_nqsv.pbs")],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            args, no_compile = json.loads(p.stdout)
            self.assertIsNone(no_compile)
            self.assertEqual(args[args.index("--run-id") + 1], "seed_44/cache_r1")
            self.assertEqual(args[args.index("--measure-steps") + 1], "1600")
            self.assertEqual(args[args.index("--backend") + 1], "pyg")
            self.assertEqual(args[args.index("--trial-timeout-sec") + 1], "14100")
            self.assertEqual(
                args[args.index("--output") + 1],
                str(root / "output/products/seed_44/cache_r1"),
            )
            env["HPCASIA_COMPILE"] = "0"
            p = subprocess.run(
                ["bash", str(PEGASUS / "run_hpcasia_nqsv.pbs")],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(json.loads(p.stdout)[1], "1")

    def test_bad_output_and_hours_are_rejected(self):
        for args in (
            ["--output", "/tmp/a b"],
            ["--output", "/tmp/a,b"],
            ["--output", "/tmp/a", "--hours", "0"],
        ):
            p = subprocess.run(
                ["bash", str(PEGASUS / "submit_hpcasia_nqsv.sh"), *args, "--dry-run"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(p.returncode, 2)


if __name__ == "__main__":
    unittest.main()
