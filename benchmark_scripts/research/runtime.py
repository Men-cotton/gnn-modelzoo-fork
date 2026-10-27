"""Subprocess entrypoints use the same uv project as the parent driver."""

from pathlib import Path


def module_command(module):
    project = Path(__file__).resolve().parents[2]
    return [
        "uv",
        "run",
        "--no-sync",
        "--project",
        str(project),
        "--",
        "python",
        "-u",
        "-m",
        module,
    ]
