# Unattended input-pipeline campaign

This plan follows agent/PLANS.md. It adds one command that runs CSX experiments sequentially, preserving normal throughput measurements, and observes resources concurrently.

## Purpose / Big Picture

The fixed nested DataLoader increased arxiv throughput by 65% at two workers. Four workers are another 16% faster, but forty workers exhausted streamer memory. An unattended campaign should collect lower-risk mechanism comparisons before exploring larger worker counts, and retain useful evidence if a later run fails.

## Progress

- [x] Inspected sensitivity runner, diagnostic instrumentation, and prior run evidence.
- [x] Add optional independent remote memory/CPU observer and validate on synthetic processes.
- [x] Add sequential campaign, configuration previews, bounded execution, resume, and summary.
- [x] Keep Grafana collection outside the campaign, as requested by the user.
- [x] Verify the 37-run dry-run and worker campaign/diagnostic/sensitivity tests without CSX jobs.

## Surprises & Discoveries

The existing diagnostic snapshots run at batch boundaries and omit RSS/PSS, memory cgroups and shared-memory capacity. They cannot observe a stalled next() call. The sensitivity runner fixes prefetch to two and static cache to zero; intervention runs therefore need separately generated configurations. A client failure does not establish remote job termination.

## Decision Log

- Use a stdlib subprocess observer launched only for opt-in diagnostic runs after iterator construction. It samples independently, exits on parent death/PID reuse or a bounded deadline, and stores errors rather than inventing zero counters.
- Run ordinary performance trials with instrumentation disabled. Run prefetched-depth, full host feature cache, and repeated fixed-batch controls separately at four workers. Repeated fixed batches are a mechanism probe, not a training accuracy result.
- Complete controls before increasing workers to 8, 12 and 16. At the user's request, skip failed/timed-out runs and continue the planned campaign, including inside sensitivity stages via --continue-on-failure. Retain failure counts and unconfirmed remote termination explicitly; never acknowledge jobs as stopped or cancel unrelated jobs. User interruption and budget exhaustion still stop submissions.
- Independent resource observation runs during diagnostics; CSX training clients remain sequential to avoid resource contention.
- Repeat a four-worker reference alongside each larger worker count: 37 runs (21 ordinary, 12 interventions, four diagnostics). Keep stage-specific reference summaries to expose temporal drift.
- Retrieve Grafana data after execution. The campaign retains raw logs, SDK metadata and step timestamps for matching the later queries; it has no Grafana client or authentication settings.

## Context and Orientation

benchmark_scripts/cerebras/run_worker_sensitivity.sh wraps tools/autotune.py below src/cerebras/modelzoo/models/gnn. The new campaign wrapper calls it for ordinary studies, and uses existing config preparation and window measurement helpers for mechanism and diagnostic runs. data_processing/worker_diagnostics.py observes real loaders; a new worker_resources.py process reads Linux counters without importing torch.

## Plan of Work

Added bounded remote observation (memory cgroups, shared memory, process CPU/RSS/PSS, pressure). The campaign driver keeps explicit plans, separate run directories and summaries, source identity checks, and safe restart behavior. Grafana data is retrieved separately after execution.

## Concrete Steps

From the GNN directory, use uv run --no-sync for Python and unittest. At repository root, invoke bash benchmark_scripts/cerebras/run_worker_campaign.sh --dry-run --output /tmp/unique-preview. Dry-run generates every configuration and never calls cszoo. The same wrapper without --dry-run is intended for the remote shared repository, after transferring these changes.

## Validation and Acceptance

Validate disabled instrumentation, synthetic real worker observations, cgroup v1/v2 memory fixtures, observer lifecycle, intervention config differences, resume refusal for changed sources, failure continuation without retries, and stopping after explicit interruption. Verify ordinary configs match existing sensitivity output and diagnostics remain separate from main throughput. Tests must not download graphs or launch CSX jobs.

## Idempotence and Recovery

Each output directory holds an immutable plan and state, protected by a file lock. Completed and failed phases are skipped on a matching resume. Changed sources/settings require a fresh directory. Failed or timed-out clients do not stop later submissions; later scheduled repetitions of that condition still run. This does not establish remote termination: surviving jobs may affect later measurements. Interrupted clients stop submission and block resume until the user selects a fresh output and the remaining experiments after inspecting jobs. Missing measurements remain failures, with final exit code 2 and an archive.

## Outcomes & Retrospective

Implemented and locally verified, including real multiprocess DataLoaders, parent-exit monitoring, disabled-path equivalence and source-identity refusal. Dry-run generated all 37 expected trials with matching nested sensitivity plans, all enabling failure continuation. The three Grafana-specific tests were removed with that client. Failure-continuation and interruption tests cover the outer campaign and inner sensitivity study. Remote CSX execution remains unverified; no CSX jobs were submitted.

## Interfaces and Dependencies

Use the existing Python 3.11 / Cerebras 2.10 / editable ModelZoo environment, PyYAML and the standard library. Linux observation requires only readable procfs and cgroup mounts; unavailable values are errors, not zeros.

Revision 2026-09-18: completed implementation and recorded local validation. The first sandboxed PyTorch test could not create IPC sockets; rerunning with the required local execution permission passed. Observer lifetime was fixed after a ResourceWarning exposed a prematurely released process handle. No remote job or external write was needed for validation.

Revision 2026-09-18: removed the newly added Grafana exporter, integration, options and tests at the user's request. The user authorized publishing the remaining campaign and diagnostic changes.

Revision 2026-09-18: the user requested continuing after failed submissions. Added explicit sensitivity continuation, enabled it in the campaign, and kept failures visible instead of treating them as successes. Explicit interruptions still stop.
