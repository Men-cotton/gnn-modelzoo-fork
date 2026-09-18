"""Launcher tests use local stub drivers; no datasets or cluster jobs."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/cerebras/modelzoo/tools/benchmark_launcher.py"
SPEC = importlib.util.spec_from_file_location("benchmark_launcher", SOURCE)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="benchmark-launcher-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stub = self.root / "driver with spaces.py"
        self.stub.write_text(
            "import json, os, pathlib, sys, time\n"
            "out = pathlib.Path(sys.argv[1])\n"
            "(out/'observed.json').write_text(json.dumps(dict(cwd=os.getcwd(), args=sys.argv[2:], token=os.environ.get('BENCHMARK_LAUNCHER_TEST_TOKEN'), path=os.environ['PATH'])))\n"
            "print('driver started', flush=True)\n"
            "while not (out/'release').exists(): time.sleep(0.03)\n"
            "print('driver finished', flush=True)\n"
            "sys.exit(7)\n"
        )

    def wait_for(self, predicate):
        end = time.monotonic() + 15
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(0.03)
        self.fail("Local launcher did not reach the expected state")

    def cleanup_session(self, output):
        path = output / "launcher.json"
        if not path.exists():
            return
        state = json.loads(path.read_text())
        # Touch only the per-test server named in this temporary output.
        if state.get("socket"):
            subprocess.run(
                launcher.tmux_command(state["socket"], "kill-server"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def test_help_and_malformed_options_do_not_launch(self):
        output = self.root / "invalid"
        prefix = ["--output", str(output)]
        cases = [
            (["--help"], 0),
            (prefix, 2),
            (prefix + ["--"], 2),
            (prefix + ["python", "-c", "pass"], 2),
            (prefix + ["--typo", "--", "python"], 2),
            (prefix + ["--dry-run", "--", "python"], 2),
            (["--", "python"], 2),
        ]
        for arguments, expected in cases:
            with (
                self.subTest(arguments=arguments),
                patch.object(launcher, "launch") as launch,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                launcher.main(arguments)
            self.assertEqual(error.exception.code, expected)
            launch.assert_not_called()
            self.assertFalse(output.exists())

    def test_driver_launch_options_default_to_foreground_and_last_flag_wins(self):
        parser = argparse.ArgumentParser()
        launcher.add_arguments(parser)
        for arguments, detach in (
            ([], False),
            (["--detach"], True),
            (["--detach", "--foreground"], False),
            (["--foreground", "--detach"], True),
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(parser.parse_args(arguments).detach, detach)

    def test_cli_separator_preserves_all_command_options(self):
        output = self.root / "metadata"
        command = [
            "custom-driver",
            "--foreground",
            "--help",
            "--output",
            "results",
            "--",
        ]
        with patch.object(launcher, "launch", return_value=0) as launch:
            self.assertEqual(
                launcher.main(["--output", str(output), "--", *command]), 0
            )
        launch.assert_called_once_with(
            command, output, name="benchmark", foreground=False, session=None
        )

    def test_generic_command_creates_its_own_output_separate_from_metadata(self):
        metadata = self.root / "launcher-metadata"
        results = self.root / "new-driver-results"
        code = (
            "import pathlib, sys; "
            "out = pathlib.Path(sys.argv[1]); "
            "out.mkdir(exist_ok=False); "
            "(out / 'result.txt').write_text('new benchmark')"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cerebras.modelzoo.tools.benchmark_launcher",
                "--output",
                str(metadata),
                "--name",
                "new-benchmark",
                "--foreground",
                "--",
                sys.executable,
                "-c",
                code,
                str(results),
            ],
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "src")
                + os.pathsep
                + os.environ.get("PYTHONPATH", ""),
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads((metadata / "launcher.json").read_text())
        self.assertEqual((state["name"], state["exit_code"]), ("new-benchmark", 0))
        self.assertEqual((results / "result.txt").read_text(), "new benchmark")
        self.assertFalse((results / "launcher.json").exists())

    def test_existing_driver_lock_blocks_launcher(self):
        output = self.root / "locked"
        output.mkdir()
        with (output / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "driver holds"):
                launcher.launch(
                    [sys.executable, "-c", "pass"],
                    output,
                    name="campaign",
                    foreground=True,
                )
        self.assertFalse((output / "launcher.json").exists())

    def test_foreground_records_failure_and_preserves_resume_files(self):
        output = self.root / "resume"
        output.mkdir()
        (output / "study.json").write_text('{"existing":true}\n')
        (output / "driver.log").write_text("older run\n")
        command = [
            sys.executable,
            "-c",
            "print('foreground output'); raise SystemExit(3)",
        ]
        code = launcher.launch(command, output, name="sensitivity", foreground=True)
        self.assertEqual(code, 3)
        self.assertEqual((output / "study.json").read_text(), '{"existing":true}\n')
        state = json.loads((output / "launcher.json").read_text())
        self.assertEqual((state["status"], state["exit_code"]), ("failed", 3))
        self.assertIsNone(state["session"])
        self.assertIsNone(state["socket"])
        self.assertIn("started_at", state)
        self.assertIn("finished_at", state)
        log = (output / "driver.log").read_text()
        self.assertTrue(log.startswith("older run\n"))
        self.assertIn("foreground output", log)
        self.assertIn("exit_code=3", log)

    def test_interrupt_before_spawn_does_not_start_driver(self):
        output = self.root / "cancelled-before-spawn"
        write = launcher.write_json

        def interrupt_start(path, state):
            write(path, state)
            if state["status"] == "running" and "driver_pid" not in state:
                signal.raise_signal(signal.SIGTERM)

        with (
            patch.object(launcher, "write_json", side_effect=interrupt_start),
            patch.object(launcher.subprocess, "Popen") as child,
        ):
            code = launcher.launch(
                [sys.executable, "-c", "pass"],
                output,
                name="campaign",
                foreground=True,
            )
        self.assertEqual(code, 128 + signal.SIGTERM)
        child.assert_not_called()
        self.assertEqual(
            json.loads((output / "launcher.json").read_text())["status"],
            "interrupted",
        )

    @unittest.skipUnless(shutil.which("tmux"), "tmux unavailable")
    def test_detached_tmux_preserves_environment_arguments_and_rejects_duplicate(
        self,
    ):
        work = self.root / "cwd with spaces"
        work.mkdir()
        output = work / "run ' $(touch injected)"
        self.addCleanup(self.cleanup_session, output)
        original = Path.cwd()
        self.addCleanup(os.chdir, original)
        os.chdir(work)
        token = "current caller token ' $(not-a-command)"
        argument = "literal space ' \" $(touch injected-argv) `touch injected-backtick`"
        path = os.environ["PATH"]
        stale_socket = "benchmark-test-stale-" + uuid.uuid4().hex[:8]
        self.addCleanup(
            subprocess.run,
            launcher.tmux_command(stale_socket, "kill-server"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            launcher.tmux_command(
                stale_socket,
                "new-session",
                "-d",
                "-s",
                "stale",
                "/bin/sh",
                "-c",
                "sleep 60",
            ),
            env={
                **os.environ,
                "BENCHMARK_LAUNCHER_TEST_TOKEN": "old server environment",
            },
            check=True,
            capture_output=True,
        )
        with (
            patch.dict(
                os.environ,
                {"BENCHMARK_LAUNCHER_TEST_TOKEN": token, "SHELL": "/bin/false"},
            ),
            redirect_stdout(io.StringIO()) as printed,
        ):
            code = launcher.launch(
                [sys.executable, str(self.stub), str(output), argument],
                output,
                name="campaign",
                session="test_session",
            )
        self.assertEqual(code, 0)
        self.wait_for(lambda: (output / "observed.json").exists())
        observed = json.loads((output / "observed.json").read_text())
        self.assertEqual(
            observed,
            {
                "cwd": str(work),
                "args": [argument],
                "token": token,
                "path": path,
            },
        )
        self.assertFalse((work / "injected").exists())
        self.assertFalse((work / "injected-argv").exists())
        self.assertFalse((work / "injected-backtick").exists())
        state = json.loads((output / "launcher.json").read_text())
        self.assertIn("tmux -L " + state["socket"], printed.getvalue())
        self.wait_for(
            lambda: (
                "driver started"
                in subprocess.run(
                    launcher.tmux_command(
                        state["socket"],
                        "capture-pane",
                        "-p",
                        "-t",
                        "=" + state["session"] + ":",
                    ),
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
        )
        with self.assertRaisesRegex(ValueError, "Another launcher"):
            launcher.launch(
                [sys.executable, "-c", "pass"],
                output,
                name="campaign",
                session="different_name",
            )
        for request in (output / ".launcher").glob("*.json"):
            self.assertNotIn(token, request.read_text())
        self.assertNotIn(token, (output / "launcher.json").read_text())
        (output / "release").touch()
        self.wait_for(
            lambda: (
                json.loads((output / "launcher.json").read_text()).get("exit_code") == 7
            )
        )
        self.wait_for(
            lambda: not launcher.session_exists(state["socket"], state["session"])
        )
        self.assertIn("driver finished", (output / "driver.log").read_text())

    def test_foreground_interrupt_reaches_driver_and_is_recorded(self):
        output = self.root / "interrupted"
        code = "import importlib.util, pathlib, sys\n"
        code += f"s=importlib.util.spec_from_file_location('launcher', {str(SOURCE)!r}); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        code += "raise SystemExit(m.launch([sys.executable, '-c', 'import time; time.sleep(60)'], pathlib.Path(sys.argv[1]), name='sensitivity', foreground=True))\n"
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(output)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        self.wait_for(
            lambda: (
                (output / "launcher.json").exists()
                and "driver_pid" in json.loads((output / "launcher.json").read_text())
            )
        )
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=10), 128 + signal.SIGTERM)
        state = json.loads((output / "launcher.json").read_text())
        self.assertEqual(state["status"], "interrupted")
        with self.assertRaises(ProcessLookupError):
            os.kill(state["driver_pid"], 0)

    def test_driver_interrupt_cleans_up_separately_sessioned_client(self):
        # Drivers can launch clients in separate sessions; HUP must become TERM
        # so the driver's cleanup handler can stop and reap those clients.
        client = self.root / "client.py"
        client.write_text(
            "import pathlib, signal, sys, time\n"
            "out = pathlib.Path(sys.argv[1])\n"
            "def stop(*unused):\n"
            "    (out/'client-cleaned').touch()\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "(out/'client-ready').touch()\n"
            "time.sleep(60)\n"
        )
        driver = self.root / "driver.py"
        driver.write_text(
            "import os, pathlib, signal, subprocess, sys, time\n"
            "out = pathlib.Path(sys.argv[1])\n"
            "client = subprocess.Popen([sys.executable, sys.argv[2], str(out)], start_new_session=True)\n"
            "def stop(*unused):\n"
            "    client.terminate()\n"
            "    client.wait(timeout=5)\n"
            "    (out/'driver-cleaned').touch()\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "(out/'client-pid').write_text(str(client.pid))\n"
            "while not (out/'client-ready').exists(): time.sleep(0.01)\n"
            "(out/'driver-ready').touch()\n"
            "time.sleep(60)\n"
        )
        for number in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=number):
                output = self.root / f"interrupted-{number}"
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(SOURCE),
                        "--output",
                        str(output),
                        "--foreground",
                        "--",
                        sys.executable,
                        str(driver),
                        str(output),
                        str(client),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.addCleanup(
                    lambda child=process: child.kill() if child.poll() is None else None
                )
                self.wait_for(lambda: (output / "driver-ready").exists())
                client_pid = int((output / "client-pid").read_text())
                process.send_signal(number)
                self.assertEqual(process.wait(timeout=10), 128 + number)
                self.assertTrue((output / "client-cleaned").exists())
                self.assertTrue((output / "driver-cleaned").exists())
                state = json.loads((output / "launcher.json").read_text())
                self.assertEqual(state["status"], "interrupted")
                self.assertEqual(state["exit_code"], 128 + number)
                for pid in (state["driver_pid"], client_pid):
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
