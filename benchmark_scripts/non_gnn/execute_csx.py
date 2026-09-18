#!/usr/bin/env python3
"""Keep a prepared CSX client alive after the campaign exits and record its result."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run import check_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = args.config.resolve()
    output = config.parent
    launch = json.loads((output / "launch.json").read_text())
    if (
        launch["backend"] != "CSX"
        or hashlib.sha256(config.read_bytes()).hexdigest() != launch["params_sha256"]
    ):
        parser.error("Expected an unchanged, prepared CSX config")
    status = {
        "state": "client_running",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    status_path = output / "client_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    try:
        check_runtime("CSX")
        with (output / "console.log").open("x") as log:
            completed = subprocess.run(
                launch["command"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        status.update(
            state="completed" if completed.returncode == 0 else "failed",
            exit_code=completed.returncode,
        )
    except Exception as exc:
        status.update(state="failed", exit_code=1, error=str(exc))
    status["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return status["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
