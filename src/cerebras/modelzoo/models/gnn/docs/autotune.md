# CSX and GPU input auto tuner

For HPC Asia runs, start with [learning curves followed by tuning](learning_campaign.md).
Researchers assess learning from the recorded curves; the driver checks execution integrity.

For HPC Asia R04, use the [worker sensitivity workflow](worker_sensitivity.md): every worker count receives the same independent repeats and a mean/sample-standard-deviation report. The defaults below describe the winner-selection mode.

`tools/autotune.py` uses the configurations in `configs/autotune/`. It runs one `python -u -m cerebras.modelzoo.cli.main fit` client
at a time with `--backend csx` (the default). `--backend pyg` launches the existing
`pyg_graphsage.py` path on one CUDA GPU. Search, budgets, resume and finalist
confirmation share one implementation. No dependency synchronization or dataset
downloads are performed.

The entry command selects the prepared Python environment with `uv run --no-sync`.
Environment checks and training clients use that same interpreter. Direct Python
invocation runs in the foreground; add `--detach` to use the shared tmux launcher.
See [launching and monitoring](worker_launch.md) for session and log handling.

## PyG on one GPU

Run on an existing GPU allocation with the prepared PyTorch/PyG environment,
local ModelZoo source installed, and offline datasets present:

```bash
cd src/cerebras/modelzoo/models/gnn
uv run --no-sync tools/autotune.py \
  --backend pyg --dataset arxiv --output autotune_runs/pyg_arxiv \
  --budget-sec 14400 --trial-timeout-sec 1800 --dry-run
```

Remove `--dry-run` to run trials. `--backend csx` selects CSX.
Use a separate output directory per backend/dataset. The PyG backend operates
inside the allocation; it does not submit PBS jobs and must not itself be
launched with `torchrun`. The GPU environment check verifies CUDA and PyG sampling
support and records GPU name/capacity, PyTorch/PyG/CUDA versions and selected
runtime environment variables. It does not require `cszoo` or load the Cerebras
SDK. The PyG profiler outside benchmark mode uses the SDK RateTracker.

PyG defaults to 40 warm-up + 400 measured steps, then 40+800 for three independent
finalist runs. Confirm window stability on the target allocation.
The initial positive-worker baseline uses prefetch 2 and persistent workers on. Zero workers use None/False. Optional
`--prefetch-factors 1 2 4` tests both persistence values in the shared search.
`pin_memory` stays fixed at the inherited value. Compilation follows the existing
PyG runner: enabled unless `NO_COMPILE` has a nonempty value. To study eager
execution, set `NO_COMPILE=1` for the whole study; changing it requires new output.

The PyG objective is actual trained seed nodes/s. Each `[Autotune]` JSON progress
record contains a cumulative seed count, wall time after CUDA synchronization,
and a finite-loss check covering every step. Endpoint differences include input
loading, transfer and GPU computation. Tail batches contribute their actual
number of seeds. The parser requires the exact start/midpoint/end records and
one completed run, and rejects evaluation or missing seed counters. For manual
measurement use `tools/measure_pyg.py LOG --start-step 40 --end-step 440`.

PyG trials disable validation and checkpoint saving and use `cache_fraction: 0.0`
(the existing uncached GPU feature-fetch path). PyG interprets `null` as automatic
GPU caching, unlike CSX; the backend handles this difference explicitly. Shape,
optimizer, precision and sampling conditions stay fixed within each study.
Current CSX trials derive actual seed counts from a runtime sampler contract;
PyG trials count consumed seeds. Rankings store
`metric`, `median_throughput`, `min_throughput` and `max_throughput` and are separate
for each backend. Source and settings must remain fixed within a study.

The timeout kills the local process group, including loader workers. A returned
PyG failure or timeout is retained and the search can proceed; Ctrl-C/SIGTERM
pauses it. If the tuner disappears before recording termination, resume still
requires `--acknowledge-stopped-jobs` after independently checking the preceding process.
`job_time_sec` applies only to CSX; the client timeout and total budget apply to both.
`best.yaml` is a backend-specific bounded benchmark configuration: 840 total steps
for default PyG, 440 for default CSX.

## Matched fixed-shape GPU measurements

`--backend fixed_shape` uses the same fixed batches, model and CPU GraphCache as
CSX. Run it inside an existing GPU allocation after reviewing the learning curves.
The handoff preserves explicit AdamW eps/betas, precision, architecture and seeds.
GPU worker counts should be tuned for that allocation separately. For example,
from the repository root:

```bash
uv run --no-sync python src/cerebras/modelzoo/models/gnn/tools/autotune.py \
  --backend fixed_shape --dataset arxiv \
  --base-config model_dirs/learning_arxiv/handoff/selected_fixed_shape_gpu.yaml \
  --workers 0 2 4 8 --prefetch-factors 1 2 --compile \
  --measure-steps 400 --confirm-steps 800 --repeats 3 \
  --trial-timeout-sec 1800 --budget-sec 86400 \
  --output model_dirs/fixed_shape_gpu_arxiv --dry-run
```

Remove `--dry-run` to execute. Defaults are eager; `--compile` is recorded as a
study condition. The native GPU parser measures synchronized endpoint differences,
including logging between endpoints, and sums actual consumed masks in that
interval. It rejects overlapping evaluation/checkpoint work and the tuner rejects
AMP-skipped optimizer updates. CSX and GPU now expose the same seed-rate numerator;
their measurement boundaries remain host-observed training-loop boundaries.
Detailed `--measure-input` and neighbor-padding measurements belong in separate
diagnostic runs because they add overhead.

After selecting GPU input settings, use `--mode sensitivity` with one `--workers`
value, `--prefetch-factor` and `--[no-]persistent-workers` to make three fresh
repetitions of the selected configuration. Use the same 40--840 window for CSX
and GPU. Report independent runs, their dispersion and each backend's settings.
The selected CSX knobs are a starting point for GPU, not evidence of GPU optimality.
Ordinary `--backend pyg` is an additional baseline: its sampling and GraphSAGE
parameterization differ from the fixed-shape model.

## CSX: prepare and preview

Use the existing editable ModelZoo environment (Python 3.11, Cerebras 2.10.0),
with datasets and cluster access already configured. Run from the GNN directory:

```bash
cd src/cerebras/modelzoo/models/gnn
uv run --no-sync tools/autotune.py \
  --dataset arxiv --output autotune_runs/arxiv --budget-sec 86400 --dry-run
```

The preview writes `plan.json` and resolved `preview_w*.yaml` files without
launching a client or importing the Cerebras runtime. It can be inspected before
running the same command without `--dry-run`:

```bash
uv run --no-sync tools/autotune.py \
  --dataset arxiv --output autotune_runs/arxiv --budget-sec 86400
```

Use `--dataset products --output autotune_runs/products` for a separate study.
Run datasets sequentially. Do not start independent studies concurrently.
A lock prevents two tuners from writing or submitting from the same study.
Follow one trial's progress using the `train.log` path printed at submission.

The default worker order is `40, 0, 4, 8, 16, 32`. Counts exceeding the launch
process's CPU affinity are reported and omitted, matching the existing runtime
validator. Thus a 20-core launch environment cannot measure w40. This filter is
not a measurement of remote input-worker CPU or memory availability: the chosen
counts must also fit those allocations. Restrict candidates with, for example,
`--workers 40 8 16`. A selected w40 always runs first.

## CSX search and measurement

1. Each worker count gets one continuous 240-step training run. The first 40
   steps are excluded from measurement; timestamps at steps 40 and 240 measure
   completed steps 41–240.
2. The fastest two eligible settings are selected (`--top-k`). With
   `--prefetch-factors 1 2 4`, each selected nonzero worker count also tests those
   prefetch factors with persistent workers both off and on, using 240 steps.
   The best settings from these measurements proceed to confirmation.
3. Finalists each run 440 steps three times. Measurement uses steps 40–440;
   candidate order reverses between rounds. `--repeats` must be at least three.
4. `best.yaml` is written only after confirmation completes. Candidates must pass
   every repeat. Final ranking uses the median, with minimum and maximum recorded.

The objective is **actual target seed nodes/s, excluding padded target slots**.
Every CSX trial enables `measure_batch_accounting`. The real input sampler logs
`GNN_INPUT_CONTRACT`: ordered batch seed counts, supervised counts excluding label
`-100`, target/label digests, batch size and traversal settings. The parser sums the
schedule for completed steps `(start, end]` and divides by the same endpoint time
difference used for the nominal rate:

```text
sum(real seeds in completed batches start+1 ... end) / (t[end] - t[start])
```

The result also retains `nominal_slots_per_second` and
`supervised_targets_per_second`. Missing or conflicting contracts, replayed static
batches, checkpoint restore, evaluation before the endpoint, and multiple input
streamers are rejected. The count is derived from the actual sampler schedule and
completed steps; it is not an on-device counter. Trials use a fresh nonrestartable
loader, a single uninterrupted training executor, and one input streamer.
Batch size (4096), fanouts, model,
precision, optimizer and seeds stay fixed. `Rate`, `GlobalRate`, and cumulative
`performance.json` values do not determine ranking.

A measurement requires exit code zero, exactly one training completion marker,
finite logged losses, increasing steps/timestamps, and exact endpoints. It
compares the two equal halves of the measurement window; their symmetric
percentage difference must be at most 2%. Missing midpoints or unstable windows
are excluded. This is a screening heuristic, not proof of stationarity or a
confidence interval. Unstable runs remain in the results. Investigate them with
a new study and, for example, `--warmup-steps 80 --measure-steps 400
--confirm-steps 800`. Warm-up must be a positive multiple of 10; measured windows
must be positive multiples of 20 so `log_steps: 10` supplies all timestamps.

`static_batch_cache_size` is zero, `cache_fraction` is null, and fake data,
validation, checkpoint saving and checkpoint autoload are disabled. Each trial
has a fresh model directory. This search does not vary `num_workers_per_csx`,
model shape, batch size, fanouts, or learning quality.

The neighbor DataLoader receives `prefetch_factor` and `persistent_workers`
from `GNNDataProcessor`. For zero workers it uses `prefetch_factor=None` and
`persistent_workers=False`. CSX worker-only tuning defaults to prefetch 2 and
persistence off; `--persistent-workers` selects persistence on. Sensitivity mode
defaults to persistence on. `pin_memory` retains automatic CUDA-dependent behavior
and is not an exploration variable.

## Budgets and resume

`--job-time-sec 7200` sets the CSX job limit. `--trial-timeout-sec 9000` separately
limits the client, including queueing and compilation. `--budget-sec` is required
and limits the sum of client wall times across invocations. It excludes local
planning, environment checks, time between trials, and downtime between resumes.
A trial starts only if its entire client allowance plus a 10-second termination
reserve fits the remaining budget. As a result the search can stop before using
the entire budget. Compilation costs and queue times are not separately inferred;
the recorded client wall time and training window have distinct meanings.

Re-run the same command and output directory to resume. Completed, unstable and
invalid measurements are retained and skipped. The cumulative budget can be
increased with `--budget-sec`; search settings, resolved base config, source hash,
Git commit and environment must match. A changed study needs a new output path.
The environment snapshot includes the actual package inventory, Python, backend dependencies,
Git status, and a hash of the Python sources and worker shell entrypoints. The full resolved
trial configuration is saved and hashed, avoiding inheritance changes mid-study.

For CSX, a failed client, timeout, Ctrl-C or SIGTERM pauses the entire study. Stopping a
local client does not establish that its remote job ended. Inspect its log/job
ID and confirm remote termination before resuming with
`--acknowledge-stopped-jobs`. This records the acknowledgement and retains the
failed trial; it does not cancel remote jobs or retry failures. If a tuner died
before recording elapsed time, resume conservatively charges that trial's full
client allowance. No further jobs start until this acknowledgement is provided.

Exit status is zero for a study with a confirmed winner (or a successful preview),
and 2 for budget exhaustion, an unconfirmed job stop, no valid candidate, or
invalid input. Budget exhaustion is resumable, not a winner selection.

## Outputs

- `study.json`: settings, environment, cumulative budget use, trials, finalists,
  final ranking and study status.
- `base.yaml`: resolved initial configuration.
- `<phase>_<candidate>_r<N>/params.yaml`, `train.log`, `result.json`, `model/`:
  isolated configuration, client output, status/measurement and SDK artifacts.
  Results include discovered `wsjob-*` IDs and paths to `performance.json` files
  when present. Missing IDs remain an empty list; raw logs are retained.
- `best.yaml`: the winning **benchmark configuration** (default CSX: 440 steps;
  PyG: 840 steps), still with
  validation and checkpointing disabled. Apply its input settings to a separate
  training configuration when evaluating learning quality. To remeasure it, use
  a fresh `--model_dir` on every invocation.

Standalone measurement remains available:

```bash
uv run --no-sync tools/measure_window.py path/to/train.log \
  --start-step 40 --end-step 240
```

## Local verification

From the repository root, using the prepared environment:

```bash
uv run --no-sync -- python -m unittest discover \
  -s src/cerebras/modelzoo/models/gnn/tests -v
```

Tests use synthetic progress logs, mock CSX clients, real local subprocesses and
real PyTorch DataLoaders on a tiny graph. They cover measurement rejection,
selection and repeat ordering, budgets, resume, job-stop handling, output
isolation, locking and DataLoader argument propagation. CUDA tests run when a GPU
is available and otherwise skip. The PyG runner fixture uses a tiny synthetic
graph, real NeighborLoader and GraphSAGE on CUDA, FP32 and `NO_COMPILE=1`; it
checks exact seed counts, disabled evaluation and absent checkpoint output. The
host loop test mocks CUDA timing/placement and executes real CPU autograd.

Local tests do not establish physical CSX execution, remote scheduling or
throughput on the full arxiv/products models. Record those measurements on the
target systems before using them in a performance comparison.
