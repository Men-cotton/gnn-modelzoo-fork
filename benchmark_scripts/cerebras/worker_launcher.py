"""Launch worker studies under tmux without retaining environment credentials."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
GNN = ROOT / "src/cerebras/modelzoo/models/gnn"
PREFLIGHT = """import json, runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
module = runpy.run_path(sys.argv[1], run_name='worker_launcher_preflight')
args = module['parse_args'](sys.argv[2:])
print('WORKER_LAUNCHER_OUTPUT=' + json.dumps(str(args.output.resolve())))
"""


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


@contextmanager
def output_lock(output, *, wait=False):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise ValueError(
                "Another launcher is running for this output"
            ) from exc
        yield


def tmux_command(socket, *args):
    # A dedicated server inherits this caller's environment, not an old server's
    # environment or ~/.tmux.conf. No environment snapshot is written to disk.
    return ["tmux", "-L", socket, "-f", "/dev/null", *args]


def session_exists(socket, session):
    return (
        subprocess.run(
            tmux_command(socket, "has-session", "-t", "=" + session),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def check_available(output):
    state_path = output / "launcher.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if (
            state.get("status") in ("starting", "running")
            and state.get("socket")
            and session_exists(state["socket"], state["session"])
        ):
            raise ValueError(
                "Another tmux launcher is starting for this output"
            )
    # Also detect drivers started without this launcher. Do not hold their lock:
    # the driver must acquire it itself, including when resuming an old study.
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another driver holds this study's lock") from exc


def supervise(request, *, foreground):
    output = Path(request["output"])
    status = {
        **request,
        "status": "running",
        "started_at": timestamp(),
        "pid": os.getpid(),
    }
    interrupted = None
    stop_deadline = None
    process = None

    def signal_driver(number):
        # Drivers install TERM cleanup handlers, while HUP has Python's default
        # immediate termination. Preserve HUP in our own status, but let nested
        # drivers run their cleanup before their process groups disappear.
        number = signal.SIGTERM if number == signal.SIGHUP else number
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            pass

    def forward_signal(number, _frame):
        nonlocal interrupted, stop_deadline
        if interrupted is None:
            interrupted = number
            # Campaign cleanup can wait 30 seconds for its nested sensitivity
            # launcher, whose own client cleanup needs a shorter allowance.
            stop_deadline = time.monotonic() + (
                60 if request["kind"] == "campaign" else 20
            )
            if process is not None and process.poll() is None:
                signal_driver(number)

    old_handlers = {
        number: signal.signal(number, forward_signal)
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    code = 1
    with (output / "driver.log").open("ab", buffering=0) as log:

        def emit(data):
            log.write(data)
            try:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            except (BrokenPipeError, OSError):
                pass

        emit(
            f"[launcher] start {status['started_at']} id={request['launch_id']}\n".encode()
        )
        write_json(output / "launcher.json", status)
        try:
            if interrupted is not None:
                raise InterruptedError("Launch cancelled before driver startup")
            process = subprocess.Popen(
                request["command"],
                cwd=request["cwd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status["driver_pid"] = process.pid
            write_json(output / "launcher.json", status)
            if interrupted is not None:
                signal_driver(interrupted)
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                drain_deadline = None
                while selector.get_map() or process.poll() is None:
                    if (
                        stop_deadline is not None
                        and time.monotonic() >= stop_deadline
                    ):
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        stop_deadline = None
                    if process.poll() is not None:
                        if drain_deadline is None:
                            drain_deadline = time.monotonic() + 1
                        elif time.monotonic() >= drain_deadline:
                            # An orphan can inherit stdout after the driver is
                            # gone. Do not wait forever for that descriptor.
                            break
                    for key, _ in selector.select(timeout=0.2):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if chunk:
                            emit(chunk)
                        else:
                            selector.unregister(key.fileobj)
            code = process.wait()
            if code < 0:
                code = 128 - code
            if interrupted is not None:
                code = 128 + interrupted
        except OSError as exc:
            status["error"] = str(exc)
            emit(f"[launcher] {exc}\n".encode())
        finally:
            if interrupted is not None:
                code = 128 + interrupted
            if process is not None:
                process.stdout.close()
            for number, handler in old_handlers.items():
                signal.signal(number, handler)
            status.update(
                status=(
                    "interrupted"
                    if interrupted is not None
                    else ("completed" if code == 0 else "failed")
                ),
                finished_at=timestamp(),
                exit_code=code,
            )
            write_json(output / "launcher.json", status)
            emit(
                f"[launcher] end {status['finished_at']} exit_code={code} id={request['launch_id']}\n".encode()
            )
    return code


def run_request(path):
    request = json.loads(path.read_text())
    output = Path(request["output"])
    with output_lock(output, wait=True):
        current = json.loads((output / "launcher.json").read_text())
        if current["launch_id"] != request["launch_id"]:
            raise ValueError("Launcher request was superseded")
        return supervise(request, foreground=False)


def launch(command, output, *, kind, foreground=False, session=None):
    output = output.resolve()
    launch_id = uuid.uuid4().hex
    socket = (
        "gnn-worker-"
        + hashlib.sha256(str(output).encode()).hexdigest()[:8]
        + "-"
        + launch_id[:8]
    )
    if session is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise ValueError(
            "--tmux-session accepts letters, digits, '-' and '_' only"
        )
    if not foreground and shutil.which("tmux") is None:
        raise ValueError("tmux is required; install it or use --foreground")
    with output_lock(output):
        check_available(output)
        request = dict(
            launch_id=launch_id,
            kind=kind,
            output=str(output),
            cwd=str(Path.cwd()),
            command=command,
            created_at=timestamp(),
            socket=None if foreground else socket,
            session=None if foreground else (session or "gnn-" + kind),
        )
        if foreground:
            return supervise(request, foreground=True)
        request_dir = output / ".launcher"
        request_dir.mkdir(exist_ok=True, mode=0o700)
        path = request_dir / (request["launch_id"] + ".json")
        write_json(path, request)
        write_json(output / "launcher.json", {**request, "status": "starting"})
        # Multiple tmux command arguments execute /bin/sh directly, bypassing
        # the server's default shell. shlex.join preserves every Python argv.
        shell_command = "exec " + shlex.join(
            [sys.executable, str(Path(__file__).resolve()), "_run", str(path)]
        )
        result = subprocess.run(
            tmux_command(
                socket,
                "new-session",
                "-d",
                "-s",
                request["session"],
                "-c",
                request["cwd"],
                "/bin/sh",
                "-c",
                shell_command,
            ),
            capture_output=True,
            text=True,
        )
        if result.returncode:
            write_json(
                output / "launcher.json",
                {
                    **request,
                    "status": "failed",
                    "finished_at": timestamp(),
                    "exit_code": result.returncode,
                    "error": result.stderr.strip(),
                },
            )
            raise ValueError("tmux launch failed: " + result.stderr.strip())
    print(f"Output: {output}")
    print(f"Session: {request['session']} (socket: {socket})")
    print(
        "Attach: "
        + shlex.join(
            tmux_command(
                socket, "attach-session", "-t", "=" + request["session"]
            )
        )
    )
    print("Status: " + shlex.join(["cat", str(output / "launcher.json")]))
    print("Log: " + shlex.join(["tail", "-f", str(output / "driver.log")]))
    print(
        "The tmux session exits when the driver finishes; logs and exit status remain."
    )
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_run":
        return run_request(Path(argv[1]))
    if not argv or argv[0] not in ("campaign", "sensitivity"):
        raise ValueError("Expected campaign or sensitivity launcher")
    kind = argv.pop(0)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--tmux-session")
    options, driver_args = parser.parse_known_args(argv)
    target = (
        GNN
        / "tools"
        / ("worker_campaign.py" if kind == "campaign" else "autotune.py")
    )
    defaults = (
        []
        if kind == "campaign"
        else ["--backend", "csx", "--mode", "sensitivity"]
    )
    command = [sys.executable, "-u", str(target), *defaults, *driver_args]
    if any(arg in ("-h", "--help", "--dry-run") for arg in driver_args):
        if any(arg in ("-h", "--help") for arg in driver_args):
            print(
                "Launcher: detached tmux by default; --foreground runs here; --tmux-session NAME sets its name.\n",
                flush=True,
            )
        os.execv(sys.executable, command)
    # Reuse the driver's complete validation; this call does not run main(),
    # inspect the cluster, create output directories or launch training.
    checked = subprocess.run(
        [sys.executable, "-c", PREFLIGHT, str(target), *defaults, *driver_args],
        capture_output=True,
        text=True,
    )
    if checked.returncode:
        sys.stderr.write(checked.stdout + checked.stderr)
        return checked.returncode
    marker = "WORKER_LAUNCHER_OUTPUT="
    output = Path(
        json.loads(
            next(
                line[len(marker) :]
                for line in checked.stdout.splitlines()
                if line.startswith(marker)
            )
        )
    )
    # Pass the same resolved path chosen by preflight, including campaign's
    # generated default, so launch records and driver output cannot diverge.
    command += ["--output", str(output)]
    return launch(
        command,
        output,
        kind=kind,
        foreground=options.foreground,
        session=options.tmux_session,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"worker launcher: {exc}", file=sys.stderr)
        raise SystemExit(2)
