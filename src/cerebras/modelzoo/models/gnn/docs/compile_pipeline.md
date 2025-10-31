# Cerebras Compile Pipeline

This note documents how the GNN workloads in the ModelZoo travel from a YAML
configuration to a compiled program on the Cerebras Wafer-Scale Engine (WSE).
It focuses on the pieces that live in this repository, calls out the hand-off
points to the proprietary `cerebras.pytorch` package, and highlights the knobs
that influence compilation.

## Execution Entry Points Inside ModelZoo

- Runs start from the CLI (or `run.py`) which delegates to
  `cerebras.modelzoo.common.run_utils.run` and `main`
  (`src/cerebras/modelzoo/common/run_utils.py#L54`). Those helpers parse the YAML
  config, resolve registry entries (`src/cerebras/modelzoo/registry/registry.yaml`)
  and call `run_trainer` (`src/cerebras/modelzoo/trainer/utils.py#L54`).
- `run_trainer` converts the parsed config into a `Trainer` instance via
  `configure_trainer_from_config`, wiring callbacks, dataloaders, optimizers,
  and precision settings (`src/cerebras/modelzoo/trainer/utils.py#L246`).
- For the GNN recipes, the configured dataloaders come from
  `GNNDataProcessor`, which dispatches to either the full-graph or neighbor
  sampling pipelines (`src/cerebras/modelzoo/models/gnn/data.py#L112`,
  `src/cerebras/modelzoo/models/gnn/pipelines/`).

## Backend Bring-Up and Compilation

- The first callback that runs is `BackendCallback`. It either reuses the active
  backend or constructs a new one with `cstorch.backend`, then aligns its
  artifact directory with the trainer (`src/cerebras/modelzoo/trainer/callbacks/backend.py#L36`).
  Backend selection happens before any model code executes.
- `ModelCallback.setup` instantiates the model under the backend device context
  (so parameters land on the lazy device) and immediately calls
  `cstorch.compile`. The returned callable is stored on the trainer as
  `compiled_model` (`src/cerebras/modelzoo/trainer/callbacks/model.py#L33`).
- Once compiled, training and validation use the compiled handle rather than
  the eager module. The forward path lives in `Trainer.forward`, which always
  invokes `trainer.compiled_model` (`src/cerebras/modelzoo/trainer/trainer.py#L706`).

## Graph Capture and Step Execution

- `Trainer.training_step` and `Trainer.validation_step` are decorated with
  `@cstorch.trace` (`src/cerebras/modelzoo/trainer/trainer.py#L660`,
  `src/cerebras/modelzoo/trainer/trainer.py#L918`). The first invocation traces the
  lazy graph and triggers compilation; later calls reuse the cached executable
  unless `retrace_every_iteration` is enabled.
- `cstorch.utils.data.DataExecutor` drives execution, supplying batches and
  handling checkpoint cadence according to the run schedule
  (`src/cerebras/modelzoo/trainer/trainer.py#L862`).
- When the backend is marked compile-only or validate-only,
  `trainer.backend.is_e2e_execution` becomes `False`, so checkpoint load/save
  short-circuits and the run exits after compilation
  (`src/cerebras/modelzoo/trainer/trainer.py#L1182`).

## Configuration Hooks That Affect Compilation

- YAML `trainer.init.backend` fields are translated into backend constructor
  arguments by `create_backend_from_config`. CSX entries allow you to set
  compile directories, `compile_only`/`validate_only`, and cluster topology
  before the backend is instantiated (`src/cerebras/modelzoo/trainer/utils.py#L210`).
- Global and scoped debug/performance flags propagate to
  `cstorch.backends` through the `GlobalFlags` and `_ScopedFlags` callbacks.
  They expose toggles such as micro-batch sizing, retracing, or IR dumps
  (`src/cerebras/modelzoo/trainer/callbacks/flags.py`).
- Precision callbacks enable mixed-precision on CSX, select the half dtype, and
  configure gradient scaling prior to compilation
  (`src/cerebras/modelzoo/trainer/callbacks/precision.py`).
- `RestartableTrainer` can prefetch compile-only jobs for alternate `num_csx`
  values, priming the compile cache before the main execution
  (`src/cerebras/modelzoo/trainer/restartable_trainer.py#L352`).

## GNN Data Pipelines and Trainer Integration

- `GNNDataProcessorConfig` validates dataset settings, resolves dataset aliases,
  and chooses either the full-graph or neighbor-sampling pipeline
  (`src/cerebras/modelzoo/models/gnn/data.py#L33`).
- The neighbor sampler produces GraphSAGE-style mini-batches, while the
  full-graph processor packages a single sparse graph snapshot with masks for
  split selection (`src/cerebras/modelzoo/models/gnn/pipelines/full_graph.py`,
  `src/cerebras/modelzoo/models/gnn/pipelines/neighbor_sampling.py`).
- The resulting dataloaders are consumed by `Trainer.fit`/`Trainer.validate`,
  so the compile pipeline operates identically to other ModelZoo models—only
  the data producers differ.

## Boundary With `cerebras.pytorch`

- `cstorch.compile`, `@cstorch.trace`, and the lazy backend machinery live in
  the `cerebras.pytorch` wheel that ships alongside the ModelZoo.
  The repository vendors the wheel’s Python sources under
  `pip-packages/cerebras/pytorch`, so you can inspect modules such as
  `backend/ltc_backend.py` or `core/compile.py` directly.
- The active virtualenv contains the same package at
  `.venv/lib/python3.8/site-packages/cerebras/pytorch`. Use
  ```bash
  python -c "import cerebras.pytorch as cstorch; print(cstorch.__file__)"
  ```
  to confirm which copy is on your `PYTHONPATH`.
- Low-level components remain binary-only: CSL kernels, appliance daemons,
  and the compiled `cerebras_pytorch_lib` bridge do not ship as source in either
  location, so hardware-level implementations are still opaque.

## Lowering to Wafer-Scale Execution

- `cstorch.compile` prepares the module for lazy execution and registers it
  with the backend implementation (`cerebras/pytorch/core/compile.py`).
  Parameters move onto a `LazyDevice`, and method wrappers ensure repeated
  calls hit the compiled path.
- During the traced step the backend records the lazy graph, enforces graph
  consistency across iterations, and packages weights plus IR fragments
  (`cerebras/pytorch/core/compile.py#L118`,
  `cerebras/pytorch/backend/ltc_backend.py#L70`).
- The backend builds a CIRH string (Cerebras’ MLIR dialect), prepares
  cross-compile state, and submits the compile request to the appliance layer
  (`cerebras/pytorch/backend/ltc_backend.py#L426` onwards). Successful compiles
  are cached so later runs with the same hash do not resubmit work.

### CIRH Lowering Internals

- When `initial_mark_step` runs, the backend clears any stale lazy state and
  executes `torch._lazy.mark_step()` inside a `cerebras_pytorch_lib`
  context manager. That call asks the C++ extension to materialize the traced
  graph into a CIRH payload and feed it through the registered compile callback
  (`pip-packages/cerebras/pytorch/backend/ltc_backend.py#L232`,
  `pip-packages/cerebras/pytorch/backend/ltc_backend.py#L673`).
- The repository vendors the native lowering toolchain alongside the Python
  bindings: `lib_torch_mlir_ltc.so` bridges PyTorch Lazy Tensor IR to MLIR,
  `torch-cirh-opt` applies the CIRH optimization pipeline, and
  `cerebras_pytorch_lib` orchestrates graph capture and callback invocation
  (`pip-packages/cerebras/pytorch/lib/`).
- Cerebras-specific ops are injected directly into the traced program via
  `torch.ops.cirh.*` wrappers. Examples include scope boundaries and sparse
  matmul kernels defined in `pip-packages/cerebras/pytorch/nn/functional.py`,
  which ensures the generated MLIR uses CIRH dialect operators.
- Custom decompositions keep the lazy graph in a CIRH-friendly form. For
  instance, the `aten.triu` rewrite in
  `pip-packages/cerebras/pytorch/decomp/_decompositions.py:267`
  intentionally differentiates two `CIRH::Arange` nodes so both survive the
  optimization passes.

## Appliance Compile Service Hand-Off

- The appliance client (`cerebras/pytorch/backend/ltc_backend.py`) delegates to
  `ApplianceMode` and `ApplianceManager` in `cerebras/pytorch/core/appliance.py`
  to manage worker images, ship materialized tensors, and launch execution.
- `ApplianceManager.compile` coordinates with Cluster Management to obtain a
  compile slot, then issues the gRPC `compile` request to the remote compile
  service (`cerebras/appliance/appliance_manager.py#L840` onwards). Returned
  artifacts include the WSE program image, task maps, and a new
  `CrossCompileState`.
- When `Trainer.fit` transitions into execute mode, the same appliance client
  programs the wafers, streams activations back through DataExecutor, and
  handles checkpoint persistence via `cstorch.saver`.

## Operational Constraints

- Ahead-of-time compilation means the first traced iteration fixes the static
  graph. Avoid Python-side control flow that produces divergent graphs—see
  `limitations.md` for coding patterns to follow
  (`src/cerebras/modelzoo/models/gnn/docs/limitation.md`).
- Compile scheduling is global per backend. The first compile locks cluster
  topology and resource sizing, so schedule the highest-resource job first if
  you plan to run train and eval back-to-back.
