from __future__ import annotations

import os
from typing import Optional


def get_available_cpu_cores() -> Optional[int]:
    """Return the CPU cores available to this process."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


def validate_num_workers(num_workers: int, *, context: str) -> int:
    """Reject worker counts that exceed the CPUs available to the runtime."""
    available_cpu_cores = get_available_cpu_cores()
    if available_cpu_cores is not None and num_workers > available_cpu_cores:
        raise ValueError(
            f"{context}: num_workers={num_workers} exceeds the available CPU "
            f"cores in this runtime ({available_cpu_cores})."
        )
    return num_workers
