"""One committed source tree per run; no shared checkout mutation or dependency sync."""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

ACTIVE = ContextVar("job_source", default=None)
VERSION_FLAGS = {
    "--revision",
    "--commit",
    "--hash",
    "--source-hash",
    "--source-revision",
}


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, timeout=60, check=False
    )
    if result.returncode:
        raise ValueError("git " + args[0] + ": " + result.stderr.decode().strip())
    return result.stdout


class SourceRun:
    def __init__(self, root, revision=None, run_dir=None):
        self.root = Path(root).resolve()
        if revision is not None and not re.fullmatch(r"[0-9a-fA-F]{7,40}", revision):
            raise ValueError(
                "--revision must be an explicit commit hash (7..40 hex digits)"
            )
        # The sole resolution of a mutable ref in this run.
        self.commit = (
            git(self.root, "rev-parse", "--verify", (revision or "HEAD") + "^{commit}")
            .decode()
            .strip()
        )
        self.run_dir = (
            Path(run_dir).resolve()
            if run_dir
            else self.root
            / "model_dirs/job_runs"
            / (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
                + uuid.uuid4().hex
            )
        )
        if any(
            self.run_dir.is_relative_to(self.root / name)
            for name in ("src", "benchmark_scripts", ".git")
        ):
            raise ValueError("Run directory must be outside tracked source directories")
        self.source = self.run_dir / "source"
        self.manifest = self.run_dir / "manifest.json"
        if self.run_dir.exists() or self.run_dir.is_symlink():
            raise ValueError("Run directory already exists: " + str(self.run_dir))
        if (
            any(c in str(self.run_dir) for c in ",\n\r\x00")
            or self.run_dir == self.root
            or self.root.is_relative_to(self.run_dir)
        ):
            raise ValueError("Invalid run directory")
        self.files = {}
        for entry in git(self.root, "ls-tree", "-rz", "--full-tree", self.commit).split(
            b"\0"
        ):
            if entry:
                meta, name = entry.split(b"\t", 1)
                mode, kind, oid = meta.decode().split()
                self.files[name.decode()] = (mode, kind, oid)
        self.blobs = {}
        self.inputs = {}
        self.document = {}

    def relative(self, path):
        return Path(os.path.abspath(path)).relative_to(self.root).as_posix()

    def exists(self, path):
        try:
            name = self.relative(path)
        except ValueError:
            return Path(path).is_file()
        return name in self.files and self.files[name][0] in ("100644", "100755")

    def read_text(self, path):
        try:
            name = self.relative(path)
        except ValueError:
            return Path(path).read_text()
        if not self.exists(path):
            raise ValueError(
                f"Required tracked payload missing at {self.commit}: {name}"
            )
        oid = self.files[name][2]
        if oid not in self.blobs:
            self.blobs[oid] = git(self.root, "cat-file", "blob", oid).decode()
        return self.blobs[oid]

    def config(self, path, stack=()):
        import yaml
        from cerebras.modelzoo.common.utils.run.config_loader import (
            _deep_merge,
            _normalize_extends,
        )
        from cerebras.modelzoo.common.utils.utils import UniqueKeyLoader

        path = Path(os.path.abspath(path))
        if path in stack:
            raise ValueError("Cyclic YAML extends: " + str(path))
        value = yaml.load(self.read_text(path), Loader=UniqueKeyLoader) or {}
        if not isinstance(value, dict):
            raise ValueError("Config requires a mapping")
        merged = {}
        for parent in _normalize_extends(value.pop("extends", None), path):
            merged = _deep_merge(
                merged,
                self.config(
                    Path(os.path.abspath(path.parent / parent)), (*stack, path)
                ),
            )
        return _deep_merge(merged, value)

    def freeze_config(self, path):
        import yaml

        path = Path(os.path.abspath(path))
        value = self.config(path)
        if not isinstance(value.get("trainer"), dict):
            raise ValueError("Config requires a trainer mapping: " + str(path))
        # Prepared non-GNN configs carry a command/source contract in launch.json.
        # Re-prepare in the pinned campaign instead of replaying a mutable command.
        if (
            not path.is_relative_to(self.root)
            and (path.parent / "launch.json").exists()
        ):
            raise ValueError(
                "Prepared launch.json cannot be reused; prepare via a pinned campaign"
            )

        def rebase(item):
            if isinstance(item, str):
                return item.replace(str(self.root / "src"), str(self.source / "src"))
            if isinstance(item, list):
                return [rebase(x) for x in item]
            if isinstance(item, dict):
                return {k: rebase(v) for k, v in item.items()}
            return item

        value = rebase(value)
        # Both legacy runconfig and trainer cluster configuration must locate pinned Python.
        if "runconfig" in value:
            value["runconfig"]["python_paths"] = [str(self.source / "src")]
        init = value["trainer"].get("init", {})
        cluster = (
            init.get("backend", {}).get("cluster_config")
            if isinstance(init.get("backend"), dict)
            else None
        )
        if isinstance(cluster, dict):
            cluster["mount_dirs"] = list(
                dict.fromkeys(
                    [*cluster.get("mount_dirs", []), str(self.run_dir), str(self.root)]
                )
            )
            cluster["python_paths"] = [str(self.source / "src")]
        content = yaml.safe_dump(value, sort_keys=False)
        key = hashlib.sha256(content.encode()).hexdigest()
        target = self.run_dir / "inputs" / (key + ".yaml")
        self.inputs[str(target)] = {
            "original": str(path),
            "sha256": key,
            "content": content,
        }
        return str(target)

    def rebase(self, value):
        # Commands are argv, never evaluated by a shell.
        value = str(value)
        for relative in ("src/", "benchmark_scripts/", ".venv/"):
            value = value.replace(
                str(self.root / relative.rstrip("/")) + "/",
                str(self.source / relative.rstrip("/")) + "/",
            )
        return value

    def environment(self):
        env = os.environ.copy()
        env.update(
            JOB_SOURCE_ORIGINAL_ROOT=str(self.root),
            JOB_SOURCE_ROOT=str(self.source),
            JOB_SOURCE_COMMIT=self.commit,
            JOB_RUN_MANIFEST=str(self.manifest),
            PYTHONPATH=str(self.source / "src"),
            PYTHONDONTWRITEBYTECODE="1",
            UV_NO_SYNC="1",
            UV_OFFLINE="1",
            UV_PROJECT_ENVIRONMENT=str(self.root / ".venv"),
            BENCHMARK_HWINFO_DIR=str(self.run_dir / "outputs/hwinfo"),
        )
        env["PATH"] = str(self.root / ".venv/bin") + os.pathsep + env.get("PATH", "")
        return env

    def pin_command(self, command):
        # Frozen input/output paths may themselves reside beneath root: preserve them.
        pinned = []
        for value in command:
            if str(self.run_dir) in value:
                pinned.append(value)
            else:
                pinned.append(self.rebase(value))
        if command[0] == "qsub":
            env = self.environment()
            keys = (
                "JOB_SOURCE_ORIGINAL_ROOT",
                "JOB_SOURCE_ROOT",
                "JOB_SOURCE_COMMIT",
                "JOB_RUN_MANIFEST",
                "PYTHONPATH",
                "PYTHONDONTWRITEBYTECODE",
                "UV_NO_SYNC",
                "UV_OFFLINE",
                "UV_PROJECT_ENVIRONMENT",
                "BENCHMARK_HWINFO_DIR",
            )
            values = []
            for key in keys:
                if any(c in env[key] for c in ",\n\r\x00"):
                    raise ValueError("Unsafe PBS environment value")
                values.append(key + "=" + env[key])
            if "-v" in pinned:
                index = pinned.index("-v") + 1
                pinned[index] += "," + ",".join(values)
            else:
                pinned[1:1] = ["-v", ",".join(values)]
        return pinned

    def save(self, status, **extra):
        if self.manifest.exists():
            saved = json.loads(self.manifest.read_text())
            if "nested_jobs" in saved:
                self.document["nested_jobs"] = saved["nested_jobs"]
        self.document.update(status=status, **extra)
        temporary = self.manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.document, indent=2) + "\n")
        temporary.replace(self.manifest)

    def prepare(self, document):
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.document = document
        self.document["dependency_environment"] = str(self.root / ".venv")
        self.document["input_data"] = str(
            self.root / "src/cerebras/modelzoo/models/gnn/data"
        )
        self.save("preparing")
        try:
            git(self.root, "worktree", "add", "--detach", str(self.source), self.commit)
            bindings = []
            for relative, target, role in (
                (".venv", self.root / ".venv", "mutable dependencies; no sync/install"),
                (
                    "src/cerebras/modelzoo/models/gnn/data",
                    self.root / "src/cerebras/modelzoo/models/gnn/data",
                    "existing input data",
                ),
                ("model_dirs", self.run_dir / "outputs", "run output"),
            ):
                link = self.source / relative
                if link.exists() or link.is_symlink():
                    raise ValueError(
                        "Binding collides with tracked source: " + relative
                    )
                if role == "run output":
                    target.mkdir()
                if target.exists():
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(target, target_is_directory=True)
                    bindings.append(
                        {"path": str(link), "target": str(target), "role": role}
                    )
            for path, info in self.inputs.items():
                target = Path(path)
                target.parent.mkdir(exist_ok=True)
                target.write_text(info["content"])
            self.document["bindings"] = bindings
            # Tracked source is read-only; bindings are deliberately excluded.
            for name, (mode, kind, _) in self.files.items():
                if kind == "blob" and mode in ("100644", "100755"):
                    (self.source / name).chmod(0o555 if mode == "100755" else 0o444)
            python = self.source / ".venv/bin/python"
            if not python.exists():
                raise ValueError(
                    "Existing .venv/bin/python required; no installation attempted"
                )
            probe = "import pathlib,cerebras.modelzoo as m; p=pathlib.Path(m.__file__).resolve(); assert p.is_relative_to(pathlib.Path(__import__('os').environ['JOB_SOURCE_ROOT'])), str(p); print(p)"
            result = subprocess.run(
                [str(python), "-c", probe],
                cwd=self.source,
                env=self.environment(),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode:
                raise ValueError("Pinned import preflight failed: " + result.stderr)
            self.save("prepared", imported_source=result.stdout.strip())
        except BaseException as exc:
            self.save("preparation_failed", error=str(exc))
            raise


def require_pinned():
    """Fail closed for direct remote drivers; use the jobs entrance to create a run."""
    root = os.environ.get("JOB_SOURCE_ROOT")
    if not root or not Path(__file__).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(
            "Remote execution requires a pinned source run: benchmark_launcher jobs (preview with --dry-run)"
        )


def pin_entry(entry, argv, *, root=None):
    """Re-enter a legacy driver/wrapper once, before any preparation or dispatch."""
    import argparse

    entry = Path(entry).absolute()
    root = Path(root) if root else Path(__file__).resolve().parents[4]
    selected = os.environ.get("JOB_SOURCE_ROOT")
    if selected and root.resolve() == Path(selected).resolve():
        require_pinned()
        return None
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--revision")
    parser.add_argument("--source-run-dir", type=Path)
    args, remaining = parser.parse_known_args(argv)
    run = SourceRun(root, args.revision, args.source_run_dir)
    relative = entry.relative_to(root)
    if not run.exists(entry):
        raise ValueError("Required entry missing at " + run.commit)
    # Historical entries must contain the pinning handoff API. No silent fallback.
    if "pin_entry" not in run.read_text(entry) and "job_source" not in run.read_text(
        entry
    ):
        raise ValueError(
            "Selected commit lacks source handoff API for " + str(relative)
        )
    remaining = absolute_arguments(remaining)
    executable = (
        [str(run.source / ".venv/bin/python")] if entry.suffix == ".py" else ["bash"]
    )
    command = [*executable, str(run.source / relative), *remaining]
    job = {
        "job": relative.as_posix(),
        "resolved_commit": run.commit,
        "source_root": str(run.source),
        "cwd": str(run.source),
        "commands": [command],
    }
    document = {
        "schema_version": 1,
        "resolved_commit": run.commit,
        "source_root": str(run.source),
        "manifest_path": str(run.manifest),
        "jobs": [job],
        "commands": [command],
        "cwd": str(run.source),
        "dependency_environment": str(root / ".venv"),
    }
    if "--dry-run" in remaining or "--json" in remaining:
        print(json.dumps(document, indent=2))
        return 0
    run.prepare(document)
    try:
        run.save("running")
        os.chdir(run.source)
        os.execvpe(command[0], command, run.environment())
    except BaseException as exc:
        run.save("failed", error=str(exc))
        raise


def absolute_arguments(argv):
    argv = list(argv)
    paths = {
        "--output",
        "--output-dir",
        "--data-dir",
        "--vocab-file",
        "--base-config",
        "--config",
        "--execute-config",
        "--study-dir",
        "--data-root",
        "--base",
        "--output-root",
    }
    for index, item in enumerate(argv):
        key, separator, value = item.partition("=")
        if key in paths:
            if separator:
                argv[index] = key + "=" + str(Path(value).absolute())
            elif index + 1 < len(argv):
                argv[index + 1] = str(Path(argv[index + 1]).absolute())
    return argv


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    result = pin_entry(args.entry, remaining)
    if result is None:
        raise ValueError("Entry is already pinned; invoke its payload directly")
    return result


def pin_cluster_config(config, *, source=None, run_dir=None, original=None):
    """Bind CSX workers to the same source and explicit shared input/output mounts."""
    source = source or os.environ.get("JOB_SOURCE_ROOT")
    if not source:
        return config
    source = Path(source)
    run_dir = Path(run_dir or Path(os.environ["JOB_RUN_MANIFEST"]).parent)
    original = Path(original or os.environ.get("JOB_SOURCE_ORIGINAL_ROOT", source))
    cluster = (
        config.setdefault("trainer", {})
        .setdefault("init", {})
        .setdefault("backend", {})
        .setdefault("cluster_config", {})
    )
    cluster["python_paths"] = [
        str(source / "src"),
        *[
            p
            for p in cluster.get("python_paths", [])
            if p not in (str(original / "src"), str(source / "src"))
            and "$MODELZOO_ROOT" not in p
        ],
    ]
    cluster["mount_dirs"] = list(
        dict.fromkeys([*cluster.get("mount_dirs", []), str(run_dir), str(original)])
    )
    return config


if __name__ == "__main__":
    raise SystemExit(main())
