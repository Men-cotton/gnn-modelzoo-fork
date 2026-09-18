"""Bounded Linux resource observer; executable without importing torch or the SDK."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import socket
import time


def read(path: Path) -> dict:
    try:
        return {"value": path.read_text().strip()}
    except OSError as exc:
        return {"error": type(exc).__name__, "errno": exc.errno}


def task_stat(path: Path) -> dict:
    value = read(path)
    if "value" not in value:
        return value
    try:
        fields = value["value"].rsplit(")", 1)[1].split()
        return dict(
            state=fields[0],
            ppid=int(fields[1]),
            minflt=int(fields[7]),
            majflt=int(fields[9]),
            user_ticks=int(fields[11]),
            system_ticks=int(fields[12]),
            start_ticks=int(fields[19]),
            rss_pages=int(fields[21]),
        )
    except (ValueError, IndexError):
        return {"error": "unparseable_stat"}


def unescape(value: str) -> str:
    for source, target in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(source, target)
    return value


def cgroups(pid: int, proc: Path = Path("/proc")) -> dict:
    """Read visible CPU/memory cgroups, including enclosing mounted ancestors."""
    membership = read(proc / str(pid) / "cgroup")
    mounts = read(proc / str(pid) / "mountinfo")
    result = dict(membership=membership, hierarchies=[], hidden_ancestors_checked=False)
    if "value" not in membership or "value" not in mounts:
        result["mountinfo"] = mounts
        return result
    for line in membership["value"].splitlines():
        _, controllers, member = line.split(":", 2)
        for mount in mounts["value"].splitlines():
            before, after = mount.split(" - ", 1)
            fields, fs = before.split(), after.split()
            v2 = not controllers and fs[0] == "cgroup2"
            selected = set(controllers.split(",")) & set(fs[2].split(","))
            if not v2 and not (
                fs[0] == "cgroup" and selected & {"cpu", "cpuset", "memory"}
            ):
                continue
            root, point = PurePosixPath(unescape(fields[3])), Path(unescape(fields[4]))
            path = PurePosixPath(member)
            if not path.is_absolute() or ".." in path.parts:
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                if member != "/":
                    continue
                relative = PurePosixPath(".")
            if v2:
                names = (
                    "cpu.max",
                    "cpu.stat",
                    "cpu.pressure",
                    "cpuset.cpus.effective",
                    "memory.current",
                    "memory.peak",
                    "memory.max",
                    "memory.high",
                    "memory.events",
                    "memory.events.local",
                    "memory.stat",
                    "memory.pressure",
                    "memory.swap.current",
                    "memory.swap.max",
                    "io.pressure",
                )
            elif "memory" in selected:
                names = (
                    "memory.usage_in_bytes",
                    "memory.max_usage_in_bytes",
                    "memory.limit_in_bytes",
                    "memory.failcnt",
                    "memory.oom_control",
                    "memory.stat",
                    "memory.memsw.usage_in_bytes",
                    "memory.memsw.limit_in_bytes",
                )
            else:
                names = (
                    "cpu.cfs_quota_us",
                    "cpu.cfs_period_us",
                    "cpu.stat",
                    "cpu.shares",
                    "cpuset.cpus",
                    "cpuset.effective_cpus",
                )
            current = point / str(relative)
            ancestors = []
            while True:
                ancestors.append(
                    dict(
                        path=str(current),
                        files={name: read(current / name) for name in names},
                    )
                )
                if current == point:
                    break
                current = current.parent
            result["hierarchies"].append(
                dict(
                    version=2 if v2 else 1,
                    controllers=controllers,
                    mount=str(point),
                    ancestors=ancestors,
                )
            )
    return result


def process(pid: int, pss: bool = False, proc: Path = Path("/proc")) -> dict:
    root = proc / str(pid)
    row = dict(
        pid=pid,
        stat=task_stat(root / "stat"),
        status=read(root / "status"),
        io=read(root / "io"),
        schedstat=read(root / "schedstat"),
        wchan=read(root / "wchan"),
        cgroup=read(root / "cgroup"),
    )
    if pss:
        row["smaps_rollup"] = read(root / "smaps_rollup")
    return row


def descendants(
    parent: int, proc: Path = Path("/proc"), limit: int = 128
) -> tuple[list[int], bool]:
    """Read children of every thread; avoid scanning unrelated processes."""
    pending, seen = [parent], set()
    while pending and len(seen) < limit:
        pid = pending.pop(0)
        if pid in seen or pid == os.getpid():
            continue
        seen.add(pid)
        for child_file in (proc / str(pid) / "task").glob("*/children"):
            for value in read(child_file).get("value", "").split():
                if value.isdigit():
                    pending.append(int(value))
    return sorted(seen), bool(pending)


def sample(parent: int, pss: bool = False) -> dict:
    started = time.monotonic_ns()
    pids, truncated = descendants(parent)
    result = dict(
        event="resource_snapshot",
        unix_time_ns=time.time_ns(),
        monotonic_ns=started,
        hostname=socket.gethostname(),
        observer_pid=os.getpid(),
        parent_pid=parent,
        clock_ticks_per_second=os.sysconf("SC_CLK_TCK"),
        processes=[process(pid, pss) for pid in pids],
        processes_truncated=truncated,
        cgroup=cgroups(parent),
        host_pressure={
            name: read(Path("/proc/pressure") / name)
            for name in ("cpu", "memory", "io")
        },
    )
    try:
        shm = os.statvfs("/dev/shm")
        result["shared_memory"] = dict(
            path="/dev/shm",
            total_bytes=shm.f_blocks * shm.f_frsize,
            available_bytes=shm.f_bavail * shm.f_frsize,
            free_bytes=shm.f_bfree * shm.f_frsize,
        )
    except OSError as exc:
        result["shared_memory"] = {"error": type(exc).__name__}
    result["collection_wall_ns"] = time.monotonic_ns() - started
    return result


def observe(
    parent: int,
    start_ticks: int,
    output: Path,
    interval: float,
    samples: int,
    pss: bool = False,
) -> None:
    """Exit on PID reuse/death or sample budget; never signal the target process."""
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive output prevents two observers from silently interleaving records.
    with output.open("x") as stream:
        reason = "sample_budget"
        for _ in range(samples):
            identity = task_stat(Path("/proc") / str(parent) / "stat")
            if (
                identity.get("start_ticks") != start_ticks
                or identity.get("state") == "Z"
            ):
                reason = "parent_exited_or_reused"
                break
            started = time.monotonic()
            stream.write(json.dumps(sample(parent, pss), allow_nan=False) + "\n")
            stream.flush()
            time.sleep(max(0, interval - (time.monotonic() - started)))
        stream.write(
            json.dumps(
                dict(
                    event="resource_monitor_stopped",
                    reason=reason,
                    unix_time_ns=time.time_ns(),
                )
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--start-ticks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--pss", action="store_true")
    args = parser.parse_args()
    if not 0.1 <= args.interval <= 60 or not 1 <= args.samples <= 1000:
        parser.error("Require interval 0.1..60 and samples 1..1000")
    observe(
        args.parent,
        args.start_ticks,
        args.output,
        args.interval,
        args.samples,
        args.pss,
    )


if __name__ == "__main__":
    main()
