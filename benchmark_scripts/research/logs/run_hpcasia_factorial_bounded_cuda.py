"""Run the unchanged factorial with a bounded native CUDA memory allocator."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys


def main():
    if any(argument in ("-h", "--help") for argument in sys.argv[1:]):
        from benchmark_scripts.research.logs import run_hpcasia_factorial as experiment
        experiment.main()
        return

    settings = "expandable_segments:True,garbage_collection_threshold:0.85"
    fraction = 0.90
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = settings
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from benchmark_scripts.research.logs import run_hpcasia_factorial as experiment

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    probe = torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    segments = torch.cuda.memory_snapshot()
    expandable = any(segment.get("is_expandable", False) for segment in segments)
    del probe, segments
    torch.cuda.empty_cache()
    if not expandable:
        raise RuntimeError("Expandable CUDA segments were not observed")

    source = Path(__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    total = torch.cuda.get_device_properties(0).total_memory
    runtime = dict(
        allocator_backend=torch.cuda.memory.get_allocator_backend(),
        cuda_allocator_configuration=settings,
        cuda_memory_fraction=fraction,
        device_total_memory_bytes=total,
        allocator_limit_bytes=int(total * fraction),
        expandable_segment_probe_passed=expandable,
        wrapper_path=str(source),
        wrapper_sha256=digest,
    )
    original_write = experiment.reference.write_json

    def write_json(path, value):
        if Path(path).name == "manifest.json":
            value["execution_environment"] = runtime
            value["local_source_sha256"][str(source)] = digest
        return original_write(path, value)

    experiment.reference.write_json = write_json
    try:
        experiment.main()
    finally:
        experiment.reference.write_json = original_write


if __name__ == "__main__":
    main()
