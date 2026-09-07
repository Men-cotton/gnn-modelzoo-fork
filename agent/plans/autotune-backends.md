# Share auto tuning between CSX and PyG

This plan follows agent/PLANS.md and is maintained with the implementation.

## Purpose / Big Picture

Users can run tools/autotune.py with --backend csx or --backend pyg. Both paths use the same sequential search, cumulative client budget, isolated trials, resume journal, stable-window filtering, and repeated finalist ranking. PyG runs one CUDA GPU through pyg_graphsage.py, records actual seed counts and synchronized wall times, and does not require the CSX launcher. Archived GPU logs inform initial measurement lengths; they do not establish new GPU performance.

## Progress

- [x] (2026-09-07) Inspect the existing tuner and PyG training, configuration and loader paths.
- [x] (2026-09-07) Locate eight completed arxiv/products GPU throughput logs from January 30 and June 22; exclude evaluation runs, GCN and the invalid June 19 bucket.
- [x] (2026-09-07) Save reproducible window analysis and its limitations; regenerated JSON is byte-identical.
- [x] (2026-09-07) Extract backend configuration, launch and measurement operations and specialize environment checks; preserve one Study implementation.
- [x] (2026-09-07) Add PyG benchmark records and support zero workers and disabled validation/checkpoints.
- [x] (2026-09-07) Verify host orchestration, configuration, measurement and the GPU path; prepare the separate follow-up commit to 39ccb20.

## Surprises & Discoveries

PyG interprets cache_fraction=None as automatic GPU caching. The CSX tuner uses None to bypass its GraphCache. PyG also rejects zero workers, converts nullable loop fields to int, and unconditionally constructs a validation loader and saves last.pt. Its legacy profiler imports a Cerebras RateTracker although execution uses PyTorch CUDA. This is now lazy and excluded from benchmark runs. The sandbox hid the local GPU, but execution outside it exposed an RTX 4060 Laptop GPU, allowing real CUDA smoke tests rather than only mocked timing. The June logs' 40-to-440 windows differ from the long 40-to-end reference by -0.53 to +1.49 percent; the worst half-window difference is 1.34 percent. January cached arxiv still fails the 2 percent screen slightly (2.08 percent).

## Decision Log

Use one GPU for the new PyG backend, with the existing GraphSAGE entry point. Multiple GPU coordination is outside this change. Default CSX windows stay 40+200 and 40+400. Default PyG windows are 40+400 and 40+800, with at least three confirmations. The latter is a longer measurement proposal, not evidence that arxiv logs establish behavior beyond 500 steps. Keep every unstable historical result. PyG uses cache_fraction=0.0 to preserve the existing uncached GPU feature-fetch path. Backend measurements expose a generic throughput value plus an explicit metric name; CSX nominal slots and PyG actual seeds must not be compared as identical metrics.

## Outcomes & Retrospective

The shared tuner now selects CSX or PyG with --backend. Both backend previews, environment checks and historical evidence regeneration pass. Host tests cover selection, resume, budgets, failure retention, real zero/positive-worker loaders and SDK import isolation. A real CUDA test on an RTX 4060 Laptop GPU executes the training loop; a second exercises the PyG runner, actual NeighborLoader, GraphSAGE, uncached feature fetching and structured measurement over 60 steps on a synthetic graph. CUDA compilation is disabled and precision is FP32 in that fixture. The measured 20-to-60 window contains exactly 140 actual seeds, no validation and no last.pt. These are functional smoke tests, not measurements of the full arxiv/products configurations or H100 performance. No CSX jobs were submitted.

## Context and Orientation

The repository root is /home/mencotton/research/gnn-modelzoo. Before this change, tools/autotune.py under src/cerebras/modelzoo/models/gnn combined CSX-specific commands and Study orchestration; it now delegates launch/configuration/measurement to tools/autotune_backends.py. tools/measure_window.py parses CSX timestamp logs. reference/pyg/runner.py prepares GPU data/models and invokes reference/pyg/train.py; reference/pyg/data.py creates NeighborLoader objects. The read-only evidence root is /home/mencotton/research/Wafer-GNN-manager/artifacts/raw_logs. Preserve OFFSET-GNN and all unrelated work.

## Plan of Work

First create tools/study_pyg_windows.py and docs/pyg_window_evidence.json using the eight named logs and exact logged boundaries. Record file hashes, cache markers, comparison windows and rejected/missing intervals. Then add tools/autotune_backends.py with CSX and PyG implementations for configuration, command, environment and measurement. Keep scheduling and ranking in Study, using a metric name and numeric throughput from either backend. Add a structured PyG progress parser in tools/measure_pyg.py. Modify PyG train_model to emit cumulative seeds, finite-loss status, and wall time after CUDA synchronization at log boundaries, only when a benchmark configuration is present. Support disabled validation/checkpoint saving without altering ordinary runs. Load the existing SDK tracker only for legacy non-benchmark runs so GPU autotuning imports and runs without it. Validate with synthetic logs, fake clients, and a tiny real PyTorch/PyG loader; run CUDA validation only if a GPU is available.

## Milestones

The evidence milestone is complete when the eight historical logs reproduce docs/pyg_window_evidence.json exactly. Missing endpoints and unstable intervals remain visible rather than being interpolated or deleted.

The backend milestone is complete when CSX and PyG previews emit the appropriate entry points and 240/440-step coarse configs, and fake-client studies share selection, budgets and resume behavior. PyG must preserve its uncached path with cache_fraction=0.0.

The execution milestone is complete when the tiny PyG runner fixture performs actual sampling, GraphSAGE forward/backward and optimizer steps on CUDA, emits actual seed counts and saves no checkpoint. The host-only alternative mocks CUDA placement/timing and is reported separately. The complete test suite and diff checks precede the requested separate commit.

## Concrete Steps

Run commands from src/cerebras/modelzoo/models/gnn using the prepared environment:

    uv run --no-sync tools/study_pyg_windows.py --artifacts /home/mencotton/research/Wafer-GNN-manager/artifacts --output docs/pyg_window_evidence.json
    uv run --no-sync tools/autotune.py --backend pyg --dataset arxiv --output /tmp/pyg-autotune-preview --budget-sec 86400 --dry-run
    uv run --no-sync -- python -m unittest discover -s tests -v

The preview must create PyG commands and 440-step configs without submitting a job. The normal command runs only on a prepared GPU allocation; run CSX and PyG studies in different output directories. Verify git diff --check, stage exact changed paths, and create one new conventional commit.

## Validation and Acceptance

Both backend mock studies must select the faster candidate, alternate confirmations, preserve failed/unstable trials, resume completed work without relaunch, and enforce budgets. The CSX tests from the first commit remain valid. PyG tests must reject missing endpoints, nonfinite losses, mixed runs and nominal-only logs for new tuning; cumulative actual seed counts must handle a short tail batch. A loader test must demonstrate num_workers=0 and propagation of prefetch/persistence for positive counts. GPU absence is reported, not replaced with a host claim of device validation. Reverse or mutation tests must expose the old configuration/measurement failures.

## Idempotence and Recovery

Analysis only reads source logs and writes the selected output JSON. Trial directories and checkpoint paths stay isolated. Backend changes are part of study identity, so resuming CSX results as PyG is rejected. PyG clients are local processes and can continue after a confirmed process-group termination; CSX failures retain the existing remote-job acknowledgement gate. A tuner lost while a client was running cannot establish that the process is gone and must keep the acknowledgement gate on restart.

## Artifacts and Notes

Eight archived logs supply historical timing evidence, with each source identified by relative path and SHA-256. Raw logs remain unchanged in the sibling repository. New measured seeds/s must not be assigned retrospectively to old nominal-only logs.

## Interfaces and Dependencies

The shared Study calls a backend's prepare_config(base, knobs, model_dir, steps, job_time_sec, warmup_steps), command(config, model_dir), and measure(log, start, end, tolerance). Each measurement includes throughput, metric and half_window_check. PyG runtime records include completed step, monotonic elapsed seconds, cumulative actual seed count, and finite loss information. Python, PyTorch, PyG and local ModelZoo source are required for GPU execution; the CSX backend separately checks Cerebras 2.10.0.

Plan created September 7 from current code and read-only archived-log inspection.

September 7 completion update: backend extraction, evidence regeneration, host checks and real RTX 4060 CUDA smoke tests completed. The initial GPU-unavailable observation applied only inside the sandbox. The standard PyG profiler retains its previous SDK tracker through a lazy import; GPU benchmark mode avoids that dependency.
