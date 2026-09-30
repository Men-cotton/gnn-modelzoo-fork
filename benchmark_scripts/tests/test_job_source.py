"""Real Git fixtures: source execution and imports, never a real scheduler."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from cerebras.modelzoo.tools import job_plan, job_source


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir()
        job_source.git(self.root, "init", "-q")
        self.write("src/cerebras/modelzoo/__init__.py", 'VERSION = "old"\n')
        self.write(
            "payload.py",
            "import cerebras.modelzoo as m; print(m.VERSION); print(m.__file__)\n",
        )
        self.write(
            "benchmark_scripts/test.pbs",
            '#!/bin/sh\ncd "$PBS_O_WORKDIR"\n.venv/bin/python payload.py\n',
        )
        self.write("params.yaml", "trainer: {}\n")
        catalog = {
            "version": 1,
            "jobs": {"test": {"kind": "fixed", "script": "benchmark_scripts/test.pbs"}},
            "sets": {"pair": ["test", "test"]},
            "presets": {"worker_diagnostics": {"workers": [2, 40]}},
        }
        self.write("benchmark_scripts/jobs.json", json.dumps(catalog))
        self.old = self.commit("old")
        (self.root / ".venv").symlink_to(
            job_plan.ROOT / ".venv", target_is_directory=True
        )

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit(self, message):
        # Only fixture files. Configuration is per-command, never user config.
        files = [
            "src/cerebras/modelzoo/__init__.py",
            "payload.py",
            "benchmark_scripts/test.pbs",
            "benchmark_scripts/jobs.json",
            "params.yaml",
        ]
        job_source.git(self.root, "add", "--", *files)
        job_source.git(
            self.root,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            message,
        )
        return job_source.git(self.root, "rev-parse", "HEAD").decode().strip()

    def prepare(self, run):
        run.prepare(
            {"resolved_commit": run.commit, "source_root": str(run.source), "jobs": []}
        )

    def payload(self, run):
        result = subprocess.run(
            [str(run.source / ".venv/bin/python"), "payload.py"],
            cwd=run.source,
            env=run.environment(),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        version, location = result.stdout.strip().splitlines()
        self.assertTrue(Path(location).is_relative_to(run.source))
        return version

    def test_resolve_once_dirty_excluded_and_actual_import(self):
        original = job_source.git
        refs = []

        def counting(root, *args):
            if args[:2] == ("rev-parse", "--verify"):
                refs.append(args)
            return original(root, *args)

        with (
            patch.object(job_source, "git", side_effect=counting),
            patch.dict(os.environ, {"PYTHONPATH": str(self.root / "src")}),
        ):
            run = job_source.SourceRun(self.root)
            self.write("src/cerebras/modelzoo/__init__.py", 'VERSION = "dirty"\n')
            self.write("OFFSET-GNN/never-copy", "untracked")
            self.prepare(run)
            self.assertEqual(self.payload(run), "old")
        self.assertEqual(len(refs), 1)
        self.assertFalse((run.source / "OFFSET-GNN").exists())
        self.assertIn(
            "dirty", (self.root / "src/cerebras/modelzoo/__init__.py").read_text()
        )
        self.assertEqual(
            json.loads(run.manifest.read_text())["imported_source"],
            str(run.source / "src/cerebras/modelzoo/__init__.py"),
        )

    def test_head_advance_and_parallel_run_independence(self):
        first = job_source.SourceRun(self.root)
        self.write("src/cerebras/modelzoo/__init__.py", 'VERSION = "new"\n')
        new = self.commit("new")
        second = job_source.SourceRun(self.root)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(self.prepare, (first, second)))
            self.assertEqual(
                list(pool.map(self.payload, (first, second))), ["old", "new"]
            )
        self.assertEqual(
            job_source.git(self.root, "rev-parse", "HEAD").decode().strip(), new
        )
        self.assertNotEqual(first.manifest, second.manifest)
        self.assertNotEqual(first.source, second.source)
        self.assertEqual(
            job_source.git(first.source, "rev-parse", "HEAD").decode().strip(), self.old
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            job_source.SourceRun(self.root, run_dir=first.run_dir)

    def test_explicit_old_commit_and_invalid_revision(self):
        self.write("src/cerebras/modelzoo/__init__.py", 'VERSION = "new"\n')
        self.commit("new")
        run = job_source.SourceRun(self.root, self.old)
        self.prepare(run)
        self.assertEqual(self.payload(run), "old")
        for revision in ("HEAD", "main", "--all", "abcd", "f" * 40):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                job_source.SourceRun(self.root, revision)

    def call_plan(self, extra):
        with (
            patch.object(job_plan, "ROOT", self.root),
            patch.object(
                job_plan, "CATALOG", self.root / "benchmark_scripts/jobs.json"
            ),
        ):
            return job_plan.main(
                [
                    "--catalog",
                    str(self.root / "benchmark_scripts/jobs.json"),
                    "--set",
                    "pair",
                    "--config",
                    str(self.root / "params.yaml"),
                    *extra,
                ]
            )

    def test_single_json_multiple_jobs_and_no_output_write(self):
        with (
            contextlib.redirect_stdout(io.StringIO()) as out,
            patch.object(job_plan, "execute_command") as execute,
        ):
            self.call_plan(["--json"])
        plan = json.loads(out.getvalue())
        self.assertEqual(len(plan["jobs"]), 2)
        self.assertEqual({j["resolved_commit"] for j in plan["jobs"]}, {self.old})
        self.assertEqual(len(plan["commands"]), 2)
        self.assertFalse(Path(plan["manifest_path"]).parent.exists())
        execute.assert_not_called()

    def test_mock_scheduler_runs_snapshot_payload_and_pbs_cwd(self):
        calls = []

        def scheduler(command, *, cwd, env):
            calls.append(command)
            self.assertEqual(env["JOB_SOURCE_COMMIT"], self.old)
            self.assertEqual(cwd, Path(env["JOB_SOURCE_ROOT"]))
            script = Path(command[-1])
            self.assertTrue(script.is_relative_to(cwd))
            # Real local PBS payload; emulate scheduler PBS_O_WORKDIR only.
            result = subprocess.run(
                ["sh", str(script)],
                cwd=cwd,
                env={**env, "PBS_O_WORKDIR": str(cwd)},
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("old\n" + str(cwd / "src"), result.stdout)
            self.write("src/cerebras/modelzoo/__init__.py", 'VERSION = "new"\n')
            (
                self.commit("advance" if len(calls) == 1 else "advance-again")
                if len(calls) == 1
                else None
            )
            return result

        with (
            patch.object(job_plan.shutil, "which", return_value="qsub"),
            patch.object(job_plan, "execute_command", side_effect=scheduler),
        ):
            self.assertEqual(self.call_plan([]), 0)
        self.assertEqual(len(calls), 2)

    def test_preparation_failure_stops_before_submit_and_records_hash(self):
        directory = self.root / "model_dirs/failed"
        original = job_source.git

        def fail(root, *args):
            if args[0] == "worktree":
                raise ValueError("fixture preparation failure")
            return original(root, *args)

        with (
            patch.object(job_source, "git", side_effect=fail),
            patch.object(job_plan.shutil, "which", return_value="qsub"),
            patch.object(job_plan, "execute_command") as execute,
        ):
            with self.assertRaisesRegex(ValueError, "fixture preparation failure"):
                self.call_plan(["--run-dir", str(directory)])
        execute.assert_not_called()
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "preparation_failed")
        self.assertEqual(manifest["resolved_commit"], self.old)

    def test_job_level_hash_revision_rejected(self):
        catalog = json.loads((self.root / "benchmark_scripts/jobs.json").read_text())
        for key in ("revision", "hash", "commit", "resolved_commit"):
            bad = copy.deepcopy(catalog)
            bad["jobs"]["test"][key] = self.old
            path = Path(self.tmp.name) / "bad.json"
            path.write_text(json.dumps(bad))
            with (
                patch.object(job_plan, "ROOT", self.root),
                self.assertRaises(ValueError),
            ):
                job_plan.load_catalog(path)

    def test_missing_committed_payload_does_not_fallback_to_worktree(self):
        run = job_source.SourceRun(self.root)
        self.write("benchmark_scripts/missing.pbs", "untracked")
        with self.assertRaisesRegex(ValueError, "Required tracked payload missing"):
            run.read_text(self.root / "benchmark_scripts/missing.pbs")

    def test_dirty_symlink_config_cannot_override_committed_blob(self):
        external = Path(self.tmp.name) / "external.yaml"
        external.write_text("trainer: {dirty: true}\n")
        (self.root / "params.yaml").unlink()
        (self.root / "params.yaml").symlink_to(external)
        run = job_source.SourceRun(self.root)
        token = job_source.ACTIVE.set(run)
        try:
            path = job_plan.config_path(self.root / "params.yaml")
            self.assertEqual(run.inputs[path]["content"], "trainer: {}\n")
        finally:
            job_source.ACTIVE.reset(token)

    def test_csx_mounts_and_dependency_paths_are_explicit(self):
        config = {
            "trainer": {
                "init": {
                    "backend": {
                        "cluster_config": {
                            "python_paths": [
                                "/extra/dependency",
                                str(self.root / "src"),
                            ]
                        }
                    }
                }
            }
        }
        run = job_source.SourceRun(self.root)
        job_source.pin_cluster_config(
            config, source=run.source, run_dir=run.run_dir, original=self.root
        )
        cluster = config["trainer"]["init"]["backend"]["cluster_config"]
        self.assertEqual(
            cluster["python_paths"], [str(run.source / "src"), "/extra/dependency"]
        )
        self.assertIn(str(run.run_dir), cluster["mount_dirs"])
        self.assertIn(str(self.root), cluster["mount_dirs"])

    def test_run_directory_source_overlap_rejected(self):
        for directory in (
            self.root / "src/generated",
            self.root / "benchmark_scripts/generated",
            self.root / ".git/run",
        ):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                job_source.SourceRun(self.root, run_dir=directory)


if __name__ == "__main__":
    unittest.main()
