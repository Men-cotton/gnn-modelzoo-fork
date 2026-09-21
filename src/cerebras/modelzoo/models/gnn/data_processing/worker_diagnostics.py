"""Bounded, opt-in observations of the actual host-side DataLoader.

Only the enabled factory imports this module. JSONL contains metadata/counters,
never batch contents. An optional bounded subprocess observes resources during
stalls. No SDK mutation, PMU access or background threads are used.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import sys
import time
import uuid
import weakref
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .worker_diagnostics_config import WorkerDiagnosticsConfig
from .samplers.neighbor_tree import GraphSAGENeighborSamplerDataset

logger = logging.getLogger(__name__)
MAX_PROCESSES = 128
MAX_THREADS_PER_PROCESS = 256


def _finish_resource_monitor(process: subprocess.Popen) -> None:
    """Reap our observer when its loader is released; never signal other PIDs."""
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _read(path: Path) -> dict[str, Any]:
    try:
        return {"value": path.read_text().strip()}
    except OSError as exc:
        return {"error": type(exc).__name__, "errno": exc.errno}


def _unescape_mount(value: str) -> str:
    for encoded, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def cgroup_snapshot(proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Read this process's cgroups and ancestors visible through its mounts.

    Paths are resolved from membership and mount roots, including namespace-root
    membership. Ancestors outside a mount are explicitly outside the observation.
    """
    membership = _read(proc_root / "self/cgroup")
    mounts = _read(proc_root / "self/mountinfo")
    result = {
        "membership": membership,
        "hierarchies": [],
        "hidden_ancestors_checked": False,
    }
    if "value" not in membership or "value" not in mounts:
        result["mountinfo_error"] = mounts.get("error")
        return result
    for line in membership["value"].splitlines():
        _, controllers, member = line.split(":", 2)
        matched = False
        for mount in mounts["value"].splitlines():
            before, after = mount.split(" - ", 1)
            fields, fs = before.split(), after.split()
            v2 = not controllers and fs[0] == "cgroup2"
            v1 = fs[0] == "cgroup" and bool(
                set(controllers.split(",")) & {"cpu", "cpuset"} & set(fs[2].split(","))
            )
            if not (v2 or v1):
                continue
            root = PurePosixPath(_unescape_mount(fields[3]))
            point = Path(_unescape_mount(fields[4]))
            path = PurePosixPath(member)
            # Do not follow namespace paths containing '..' out of the mount.
            if not path.is_absolute() or ".." in path.parts:
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                if member != "/":
                    continue
                relative = PurePosixPath(".")
            current = point / str(relative)
            hierarchy = {
                "version": 2 if v2 else 1,
                "controllers": controllers,
                "mount": str(point),
                "mount_root": str(root),
                "ancestors": [],
            }
            names = (
                (
                    "cpu.max",
                    "cpu.stat",
                    "cpu.weight",
                    "cpu.pressure",
                    "cpuset.cpus.effective",
                )
                if v2
                else (
                    "cpu.cfs_quota_us",
                    "cpu.cfs_period_us",
                    "cpu.stat",
                    "cpu.shares",
                    "cpuset.cpus",
                    "cpuset.effective_cpus",
                )
            )
            while True:
                values = {name: _read(current / name) for name in names}
                entry = {"path": str(current), "files": values}
                try:
                    if v2:
                        quota, period = values["cpu.max"]["value"].split()
                        entry["quota_cores"] = (
                            None if quota == "max" else int(quota) / int(period)
                        )
                    else:
                        quota = int(values["cpu.cfs_quota_us"]["value"])
                        period = int(values["cpu.cfs_period_us"]["value"])
                        entry["quota_cores"] = None if quota < 0 else quota / period
                except (KeyError, ValueError, ZeroDivisionError):
                    pass
                hierarchy["ancestors"].append(entry)
                if current == point:
                    break
                current = current.parent
            result["hierarchies"].append(hierarchy)
            matched = True
        if not matched:
            result.setdefault("unresolved_memberships", []).append(line)
    return result


def _task_stat(path: Path) -> dict[str, Any]:
    raw = _read(path)
    if "value" not in raw:
        return raw
    try:
        fields = raw["value"].rsplit(")", 1)[1].split()
        return {
            "state": fields[0],
            "ppid": int(fields[1]),
            "user_ticks": int(fields[11]),
            "system_ticks": int(fields[12]),
            "start_ticks": int(fields[19]),
        }
    except (IndexError, ValueError):
        return {"error": "unparseable_stat"}


def process_snapshot(pid: int) -> dict[str, Any]:
    """Raw CPU counters for one process and a bounded set of its threads."""
    root = Path("/proc") / str(pid)
    result = {
        "pid": pid,
        "sample_monotonic_ns": time.monotonic_ns(),
        "cgroup": _read(root / "cgroup"),
        "comm": _read(root / "comm"),
        "stat": _task_stat(root / "stat"),
        "schedstat": _read(root / "schedstat"),
    }
    status = _read(root / "status")
    if "value" in status:
        keys = {
            "Threads",
            "Cpus_allowed_list",
            "Mems_allowed_list",
            "voluntary_ctxt_switches",
            "nonvoluntary_ctxt_switches",
        }
        result["status"] = {
            key: value.strip()
            for line in status["value"].splitlines()
            for key, value in [line.split(":", 1)]
            if key in keys
        }
    else:
        result["status"] = status
    try:
        tids = sorted(
            (p for p in (root / "task").iterdir() if p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        result["threads_truncated"] = len(tids) > MAX_THREADS_PER_PROCESS
        result["threads"] = [
            {
                "tid": int(t.name),
                "comm": _read(t / "comm"),
                "stat": _task_stat(t / "stat"),
                "schedstat": _read(t / "schedstat"),
                "wchan": _read(t / "wchan"),
            }
            for t in tids[:MAX_THREADS_PER_PROCESS]
        ]
    except OSError as exc:
        result["threads_error"] = type(exc).__name__
    return result


def _source(cls: type) -> dict[str, Any]:
    try:
        path = Path(inspect.getfile(cls)).resolve()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    except (OSError, TypeError) as exc:
        return {"error": type(exc).__name__}


class Recorder:
    """Append whole records to a process-specific file; never retain an FD."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.host = socket.gethostname()
        self.pid = None
        self.failed = False

    def emit(self, event: str, **values: Any) -> None:
        pid = os.getpid()
        if self.pid != pid:
            self.pid, self.failed = pid, False
        if self.failed:
            return
        record = {
            "schema_version": 1,
            "event": event,
            "hostname": self.host,
            "pid": pid,
            "ppid": os.getppid(),
            "unix_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **values,
        }
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / f"{pid}.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        except OSError as exc:
            self.failed = True
            logger.warning("GNN worker diagnostics disabled for PID %s: %s", pid, exc)


class ObservedNeighborDataset(GraphSAGENeighborSamplerDataset):
    """Measure sampling/gather only while ObservedDataset requests a batch."""

    phase_times = None

    def _phase(self, name, function, *args):
        if self.phase_times is None:
            return function(*args)
        start, cpu_start = time.monotonic_ns(), time.process_time_ns()
        result = function(*args)
        end, cpu_end = time.monotonic_ns(), time.process_time_ns()
        self.phase_times[name] = {
            "wall_ns": end - start,
            "process_cpu_ns": cpu_end - cpu_start,
        }
        return result

    def _sample_layers(self, target_nodes, target_mask):
        return self._phase(
            "sampling", super()._sample_layers, target_nodes, target_mask
        )

    def _gather_features(self, layer_nodes, layer_masks):
        return self._phase("gather", super()._gather_features, layer_nodes, layer_masks)


class ObservedDataset(Dataset):
    """Time a bounded number of complete batch generations per process."""

    def __init__(self, source: Dataset, recorder: Recorder, max_batches: int):
        self.source, self.recorder, self.max_batches = source, recorder, max_batches
        self.pid = None
        self.count = 0

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int | tuple[int, int]) -> Any:
        if self.pid != os.getpid():
            self.pid, self.count = os.getpid(), 0
        if self.count >= self.max_batches:
            return self.source[index]
        ordinal = self.count
        self.count += 1
        start, cpu_start = time.monotonic_ns(), time.process_time_ns()
        phases = {}
        if isinstance(self.source, ObservedNeighborDataset):
            self.source.phase_times = phases
        try:
            batch = self.source[index]
        finally:
            if isinstance(self.source, ObservedNeighborDataset):
                self.source.phase_times = None
        end, cpu_end = time.monotonic_ns(), time.process_time_ns()
        info = get_worker_info()
        self.recorder.emit(
            "batch_generated",
            batch_index=int(index[1] if isinstance(index, tuple) else index),
            input_epoch=int(index[0]) if isinstance(index, tuple) else 0,
            process_batch_ordinal=ordinal,
            phases=phases,
            worker_id=info.id if info else None,
            start_ns=start,
            end_ns=end,
            wall_ns=end - start,
            process_cpu_ns=cpu_end - cpu_start,
        )
        if ordinal == 0:
            tensors = []

            def layout(value, name):
                if isinstance(value, torch.Tensor):
                    tensors.append(
                        dict(
                            name=name,
                            shape=list(value.shape),
                            dtype=str(value.dtype),
                            device=str(value.device),
                            logical_bytes=value.numel() * value.element_size(),
                        )
                    )
                elif isinstance(value, dict):
                    for key, item in value.items():
                        layout(item, f"{name}.{key}")
                elif isinstance(value, (list, tuple)):
                    for index, item in enumerate(value):
                        layout(item, f"{name}[{index}]")

            layout(batch, "batch")
            self.recorder.emit(
                "batch_layout",
                tensors=tensors,
                logical_bytes=sum(t["logical_bytes"] for t in tensors),
                note="Logical tensor sizes, not resident memory; shared storage may be counted more than once.",
            )
        return batch


def _worker_initialized(worker_id: int) -> None:
    info = get_worker_info()
    # Cerebras can invoke this callback in the local inspection process.
    if info is None:
        return
    info.dataset.recorder.emit(
        "worker_initialized",
        worker_id=worker_id,
        num_workers=info.num_workers,
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        source=_source(type(info.dataset.source)),
        affinity=(
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
    )


class ObservedDataLoader(DataLoader):
    """Keep PyTorch's iterator/worker lifecycle, observing only enabled runs."""

    def __init__(
        self, dataset: Dataset, *, diagnostics: WorkerDiagnosticsConfig, **kwargs: Any
    ):
        directory = (
            Path(diagnostics.output_dir).expanduser()
            / f"loader-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}"
        )
        self.recorder = Recorder(directory)
        self.diagnostics = diagnostics
        self.observed_batches = self.snapshots = self.iterations = 0
        self.last_snapshot_ns = None
        self.resource_monitor_started = False
        original_class = type(dataset)
        super().__init__(
            ObservedDataset(dataset, self.recorder, diagnostics.max_batches),
            worker_init_fn=_worker_initialized,
            **kwargs,
        )
        sources = {
            "dataset": _source(original_class),
            "diagnostics": _source(type(self)),
        }
        for name in (
            "cerebras.modelzoo.models.gnn.data_processing.processor",
            "cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree",
        ):
            module = sys.modules.get(name)
            if module is not None and getattr(module, "__file__", None):
                try:
                    path = Path(module.__file__).resolve()
                    sources[name] = {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                except OSError as exc:
                    sources[name] = {"error": type(exc).__name__}
        self.recorder.emit(
            "loader_created",
            settings=self._settings(),
            diagnostics=diagnostics.model_dump(),
            sources=sources,
            torch_version=str(torch.__version__),
            cerebras_pytorch_version=getattr(
                sys.modules.get("cerebras.pytorch"), "__version__", None
            ),
            torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            clock_ticks_per_second=os.sysconf("SC_CLK_TCK"),
            affinity=(
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            cgroup=cgroup_snapshot(),
        )
        logger.info("GNN worker diagnostics: %s", directory)

    def _settings(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "num_workers",
                "prefetch_factor",
                "persistent_workers",
                "pin_memory",
                "batch_size",
            )
        }

    def __iter__(self):
        iterator = super().__iter__()
        if self.diagnostics.resource_monitor and not self.resource_monitor_started:
            self.resource_monitor_started = True
            from . import worker_resources

            try:
                self.recorder.directory.mkdir(parents=True, exist_ok=True)
                output = self.recorder.directory / f"resources-{os.getpid()}.jsonl"
                command = [
                    sys.executable,
                    worker_resources.__file__,
                    "--parent",
                    str(os.getpid()),
                    "--start-ticks",
                    str(
                        worker_resources.task_stat(Path("/proc/self/stat"))[
                            "start_ticks"
                        ]
                    ),
                    "--output",
                    str(output),
                    "--interval",
                    str(min(60, self.diagnostics.snapshot_interval_seconds)),
                    "--samples",
                    str(max(1, self.diagnostics.max_snapshots)),
                ]
                if self.diagnostics.resource_monitor_pss:
                    command.append("--pss")
                with (self.recorder.directory / "resources.stderr").open("a") as errors:
                    observer = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=errors,
                    )
                weakref.finalize(self, _finish_resource_monitor, observer)
                self.recorder.emit(
                    "resource_monitor_started",
                    observer_pid=observer.pid,
                    output=str(output),
                    source=_source(worker_resources),
                )
            except (OSError, KeyError) as exc:
                self.recorder.emit("resource_monitor_error", error=type(exc).__name__)
        self.iterations += 1
        if (
            self.observed_batches >= self.diagnostics.max_batches
            and self.snapshots >= self.diagnostics.max_snapshots
        ):
            return iterator
        # Read-only use of PyTorch's process handles; child initialization is
        # independently recorded by get_worker_info() in each actual child.
        pids = [w.pid for w in getattr(iterator, "_workers", ()) if w.pid is not None]
        if self.iterations <= max(
            1, self.diagnostics.max_batches, self.diagnostics.max_snapshots
        ):
            self.recorder.emit(
                "iterator_started",
                iteration=self.iterations,
                settings=self._settings(),
                worker_pids=pids,
            )
        return ObservedIterator(iterator, self, pids)

    def snapshot(self, pids: list[int]) -> None:
        if self.snapshots >= self.diagnostics.max_snapshots:
            return
        start = time.monotonic_ns()
        if (
            self.last_snapshot_ns is not None
            and start - self.last_snapshot_ns
            < self.diagnostics.snapshot_interval_seconds * 1e9
        ):
            return
        self.last_snapshot_ns = start
        self.snapshots += 1
        all_pids = [os.getpid(), *pids]
        processes = [process_snapshot(pid) for pid in all_pids[:MAX_PROCESSES]]
        cgroup = cgroup_snapshot()
        self.recorder.emit(
            "snapshot",
            processes=processes,
            processes_truncated=len(all_pids) > MAX_PROCESSES,
            cgroup=cgroup,
            collection_wall_ns=time.monotonic_ns() - start,
        )


class ObservedIterator:
    def __init__(self, iterator: Any, loader: ObservedDataLoader, pids: list[int]):
        self.iterator, self.loader, self.pids = iterator, loader, pids
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        loader = self.loader
        loader.snapshot(self.pids)
        if loader.observed_batches >= loader.diagnostics.max_batches:
            return next(self.iterator)
        ordinal = loader.observed_batches
        start, cpu_start = time.monotonic_ns(), time.process_time_ns()
        batch = next(self.iterator)
        end, cpu_end = time.monotonic_ns(), time.process_time_ns()
        loader.observed_batches += 1
        loader.recorder.emit(
            "batch_received",
            loader_batch_ordinal=ordinal,
            iteration=loader.iterations,
            iteration_batch_index=self.index,
            start_ns=start,
            end_ns=end,
            wall_ns=end - start,
            process_cpu_ns=cpu_end - cpu_start,
        )
        self.index += 1
        return batch
