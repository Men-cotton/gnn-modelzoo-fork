"""Check uv project selection and argument preservation without launching jobs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "benchmark_scripts/research/run.sh"


class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="shared feature uv ")
        self.addCleanup(self.temp.cleanup)
        self.cwd = Path(self.temp.name)
        stub = self.cwd / "uv"
        stub.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "print(json.dumps(dict(argv=sys.argv[1:], path=os.environ['PYTHONPATH'], cwd=os.getcwd())))\n"
        )
        stub.chmod(0o755)
        self.env = {**os.environ, "PATH": str(self.cwd)+os.pathsep+os.environ["PATH"],
                    "PYTHONPATH": "/existing/import/path"}

    def call(self, *args):
        return subprocess.run([str(RUN), *args], cwd=self.cwd, env=self.env,
                              capture_output=True, text=True)

    def test_foreign_cwd_and_spaced_arguments(self):
        args = ["--input", "input with spaces", "--output", "output with spaces"]
        result = self.call("feature_sharing.device.pilot", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        observed = json.loads(result.stdout)
        self.assertEqual(observed["argv"], ["run", "--no-sync", "--project", str(ROOT),
            "--", "python", "-u", "-m", "benchmark_scripts.research.feature_sharing.device.pilot", *args])
        self.assertEqual(observed["path"].split(os.pathsep),
                         [str(ROOT), str(ROOT/"src"), "/existing/import/path"])
        self.assertEqual(observed["cwd"], str(self.cwd))

    def test_launcher_reuses_modelzoo_module_and_keeps_child_command(self):
        child = ["--output", "launcher path", "--", str(RUN),
                 "feature_sharing.device.pilot", "--help"]
        result = self.call("launcher", *child)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads(result.stdout)["argv"]
        self.assertEqual(argv[8], "cerebras.modelzoo.tools.benchmark_launcher")
        self.assertEqual(argv[9:], child)

    def test_unknown_module_does_not_call_uv(self):
        result = self.call("unrelated.module")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
