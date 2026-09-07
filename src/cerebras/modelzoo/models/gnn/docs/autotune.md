# CS-3 input auto tuner

`tools/autotune.py` uses the configurations and measurement script from
`cs3_autotune_overrides.zip`. It runs one `uv run --no-sync -- cszoo fit` client
at a time. No dependency synchronization or dataset downloads are performed.

## Prepare and preview

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

## Search and measurement

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

The objective is **nominal seed-node slots/s, including padding**:

```text
4096 * (end_step - start_step) / (timestamp[end_step] - timestamp[start_step])
```

It does not measure valid unpadded nodes/s. Batch size (4096), fanouts, model,
precision, optimizer and seeds stay fixed. `Rate`, `GlobalRate`, and cumulative
`performance.json` values do not determine ranking. The original ZIP's
`autotune_evidence.json` is historical evidence supplied with that archive; the
auto tuner does not treat those numbers as new CS-3 measurements.

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

The neighbor DataLoader now receives `prefetch_factor` and
`persistent_workers` from `GNNDataProcessor`. For zero workers it uses
`prefetch_factor=None` and `persistent_workers=False`. The imported ZIP settings
explicitly use prefetch 2 and persistence off for positive workers, preserving
the previous effective PyTorch behavior for the worker-only comparison.
Existing configurations with `persistent_workers: True` now actually enable it.
`pin_memory` retains the existing automatic CUDA-dependent behavior and is not
an exploration variable.

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
The environment snapshot includes the actual package inventory, Python, SDK and
uv versions, Git status, and a hash of the Python sources. The full resolved
trial configuration is saved and hashed, avoiding inheritance changes mid-study.

A failed client, timeout, Ctrl-C or SIGTERM pauses the entire study. Stopping a
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
- `best.yaml`: the winning **440-step benchmark configuration**, still with
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
isolation, locking and DataLoader argument propagation. They do not validate
CS-3 submission, remote scheduling or device throughput.
