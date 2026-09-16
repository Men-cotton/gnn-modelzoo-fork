# Observe GNN input workers without cluster administration

This ExecPlan follows agent/PLANS.md and is maintained through implementation.

## Purpose / Big Picture


An opt-in neighbor-loader diagnostic will establish the actual loader settings and processes executing on a Cerebras PyTorch Worker, and record process/thread CPU counters, visible CPU cgroup constraints, and bounded batch timings. The default continues to use the original PyTorch loader and dataset. No CSL interfaces, privileged profiling, remote jobs, or SDK modifications are involved.

## Progress


- [x] 2026-09-16: Inspected the neighbor factory, configuration, native fixed-shape entry point, local Cerebras PyTorch factory handling, and existing tests.
- [x] 2026-09-16: Implemented settings, bounded records, sampling/gather timing, and both factory entry points.
- [x] 2026-09-16: Verified real fork/spawn processes, payload equivalence, disabled-path isolation, cgroup v1/v2 fixtures, and Cerebras serialization.
- [x] 2026-09-16: Documented results and prepared the scoped feature for commit.

## Surprises & Discoveries


The initial sandboxed test could not open a local Unix socket; it was stopped and passed with the socket restriction lifted. The factory test uses the existing SDK fw_user_serialize/fw_user_deserialize functions, with no additional dependency.

Cerebras PyTorch changes its local inspection loader to num_workers=0. Therefore settings must be captured again at iterator startup, and worker_init_fn must check get_worker_info() before identifying a call as a subprocess. The remote InputGeneratorPyTorch calls the input factory independently.

## Decision Log


On 2026-09-16, choose factory-time opt-in construction of a diagnostic DataLoader and dataset wrapper. Keep all default per-batch paths unchanged. Use bounded synchronous snapshots at batch boundaries rather than a background thread, avoiding thread/fork interactions and shutdown hooks. This captures interval CPU counters but cannot sample during an indefinitely blocked next call. Write separate JSONL files per process in a factory-specific directory on a user-selected Worker-visible filesystem. Record total dataset generation, sampling/gather phases and parent next times; proprietary shared-memory copies and the separate cs_worker_app remain outside this probe.

## Outcomes & Retrospective


Implemented the default-disabled diagnostic. Twelve diagnostic tests, two existing loader-setting tests, and eight existing fixed-shape tests pass. The diagnostic test compares every training loss exactly between enabled and disabled runs on CPU and CUDA. Existing tests also exercised CUDA forward/backward parity and float32/float16 GPU training. Ruff, Black, and diff whitespace checks pass. Remote CSX deployment remains untested; the actual Cerebras PyTorch serializer and input factory were exercised locally. No privileged profiling was added.

## Context and Orientation


All feature code lives under src/cerebras/modelzoo/models/gnn. data_processing/processor.py validates the YAML configuration and forwards settings to data_processing/samplers/neighbor_tree.py. Its factory is executed by Cerebras PyTorch on a Worker. fixed_shape_gpu.py builds the same neighbor loader for native PyTorch. New data_processing/worker_diagnostics.py provides opt-in recording; tests/test_worker_diagnostics.py uses tiny in-memory graphs without downloads. docs/worker_diagnostics.md explains output and limits.

## Plan of Work


First add a default-disabled worker_diagnostics configuration, forwarding it through both entry points and rejecting enabled full-graph use. In the neighbor factory retain the exact ordinary loader when disabled. When enabled, wrap its dataset to measure a bounded number of getitem calls per process and use a loader subclass to measure a bounded number of next calls and periodic snapshots. Record actual child initialization, source file hashes, effective settings and visible cgroup ancestors. Keep exceptions in diagnostic I/O from breaking training; record or warn about unavailable observations.

Then add CPU-only tests using real PyTorch subprocesses. Disabled-path tests must fail if recording or diagnostic construction is reached. Enabled/disabled batches must be bitwise identical, including cached batches, multiple epochs, and persistent workers. Synthetic cgroup v1/v2 files check hierarchy resolution and explicit limitations. Test local Cerebras input-factory inspection and serialized factory execution. Run existing loader and fixed-shape tests alongside the new tests.

## Concrete Steps


From src/cerebras/modelzoo/models/gnn run:

    UV_CACHE_DIR=/tmp/gnn-worker-uv-cache uv run --no-sync python -m unittest discover -s tests -p 'test_worker_diagnostics.py' -v
    UV_CACHE_DIR=/tmp/gnn-worker-uv-cache uv run --no-sync python -m unittest discover -s tests -p 'test_loader_settings.py' -v
    UV_CACHE_DIR=/tmp/gnn-worker-uv-cache uv run --no-sync python -m unittest discover -s tests -p 'test_fixed_shape_gpu.py' -v

Use temporary directories for records. From the repository root run git diff --check and review the staged files, then commit with a Conventional Commit subject. Preserve the pre-existing untracked OFFSET-GNN directory.

## Validation and Acceptance


All tests must pass. Enabling diagnostics yields parseable per-process JSONL, measured worker IDs and parent PID, raw process/thread counters and visible cgroup state. Failures reading diagnostic resources are represented rather than replacing missing counters with zero. Disabled loaders retain the original classes and perform no diagnostic reads or clocks while iterating. No empirical claim of negligible enabled overhead is required: the explicit default-off switch isolates its bounded observation cost.

## Idempotence and Recovery


Every factory creates a unique recording directory; repeated runs do not truncate earlier evidence. Temporary tests clean their own output. Only named feature files are staged. No remote state changes are required.

## Artifacts and Notes


Final diagnostic test: 12 tests passed in 14.540 seconds. Existing loader tests: 2 passed; existing fixed-shape tests: 8 passed. Commands above were run with OUTDATED_IGNORE=1 and timeout 120. Disabled-path call tracing covers both zero and two workers and finds zero calls into the diagnostic implementation.

Static checks from the GNN directory used uv run --no-sync ruff check --no-cache on the new Python files, and uv run --no-sync black --check on all six changed Python files; both pass. git diff --check passes.

The collate function moved from a lambda to a module-level function so spawn workers can serialize it while retaining the same batching behavior.

## Interfaces and Dependencies


Use Python standard-library filesystem/time/hash/JSON utilities, existing PyTorch, and existing Pydantic. WorkerDiagnosticsConfig provides enabled, output_dir, max_batches, snapshot_interval_seconds, and max_snapshots. Linux /proc and cgroup readings are best effort and scoped to this loader and its direct observed children. All measurements describe host-side CPU work and do not synchronize CUDA or infer WSE timings.

Revision 2026-09-16: initial plan records the opt-in boundary, Worker-only provenance requirement, and local test strategy.

Revision 2026-09-16: recorded implementation, bounded phase timings, actual SDK serialization tests, successful CPU/CUDA validation, and remote-deployment limits.
