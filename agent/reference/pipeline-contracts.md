# GNN pipeline contracts

Use this reference when extending loaders, changing training semantics, or upgrading the SDK. Paths below are relative to `src/cerebras/modelzoo/models/gnn/`. Runtime versions come from `pyproject.toml` and the active environment; historical experiment versions come from their saved metadata.

Implementation revision: `327b56a` contains the GNN runtime fixes and regression tests audited on 2026-09-18, including `task/loss.py` and `gpu_policy.py`. Use that revision or a descendant when applying these contracts. The physical CSX validation boundary remains as described below.

## Loader ownership and data

`GNNDataProcessor.create_dataloader()` returns a native PyTorch loader. ModelZoo's `create_dataloader_from_config` in repository-root `src/cerebras/modelzoo/trainer/utils.py` owns the Cerebras wrapper. Validate changes through its serialized remote factory, because the SDK's local inspection can change the local loader to zero workers. `tests/test_trainer_loader.py` and `tests/test_loader_regressions.py` exercise this boundary with real child processes.

The SDK can invoke `worker_init_fn` during local inspection: record a child initialization only when `torch.utils.data.get_worker_info()` is not None. Keep collate functions at module scope so spawn workers can serialize them; lambdas and nested functions do not satisfy that contract.

CSX input Worker replicas and PyTorch child workers are separate quantities. The GNN loaders currently support one CSX input streamer. More streamers need global-batch distribution and consistent epoch/evaluation ordering before the guard in `data_processing/runtime/csx.py` can be removed.

Neighbor datasets contain complete logical target batches; the outer PyTorch loader uses batch size 1. `drop_last_batch` applies to the logical batch. Its default is False, preserving the padded final batch. Full-graph loaders contain one graph. Both paths propagate prefetch, persistence and pinning settings; an omitted/null split defaults to train at the processor boundary.

CSX and native `fixed_shape_gpu` use CPU GraphCache. The ModelZoo GPU loader also uses CPU cache with child workers; zero-worker GPU caching returns CUDA tensors and disables pinning. All emitted tensors, including full-graph adjacency, must occur in standard containers visible to the SDK.

papers100M retains compact CSR indexes after releasing the original edge tensor. Recreating a loader must reuse those indexes. Implicit undirected neighborhoods deduplicate reciprocal edges, repeated edges and self-loops per sampled node. Large-graph throughput and memory must be measured separately from tiny-graph correctness.

## Loss, evaluation and GPU configuration

`task/loss.py` computes unreduced classification loss with safe labels, then divides the FP32 masked sum by the FP32 valid-target count. Valid targets satisfy `target_mask` and `label != -100`. Empty supervision produces zero loss and gradient. Keep the count explicit across SDK conversion; a Python mean with ignored labels alone is insufficient evidence of the lowered denominator.

`tests/test_masked_loss.py` compares loss and gradients to an unpadded PyTorch oracle. For an SDK upgrade, additionally convert the equivalent small forward/backward graph from the repository root:

    .venv/lib/python3.11/site-packages/cerebras/pytorch/lib/torch-cirh-opt \
      --torch-to-cirh-pipeline \
      agent/reference/fixtures/masked-loss.mlir \
      -o /tmp/gnn-masked-loss.cirh.mlir

Check that both loss and gradient divide by the runtime mask count, clamped to at least 1. The fixture is authored canonicalized arithmetic, not a full Python export. A new CSX execution must still verify the real model, compiler transformations and microbatch behavior. Local conversion and host tests do not establish device numerical equivalence.

The [compiler reference](compiler-artifacts.md) retains the SDK mean-loss counterexamples and archived full-graph evidence. Use those when deciding whether a newer SDK permits removal of the explicit-count workaround.

All evaluation paths exclude ignored labels from accuracy. PyG training uses `fit.val_dataloader`; standalone validation uses `validate.val_dataloader`. Explicit null splits fall back to the relevant train/validation split.

`gpu_policy.py` translates precision and AdamW settings for the native GPU runners. PyG's legacy `task.to_float16` applies only when precision is absent. BF16 does not use FP16 gradient scaling. Explicit optimizer kwargs are preserved; implicit defaults can still differ between Cerebras and PyTorch, so matched experiments should specify eps/betas as well as learning rate and weight decay.

PyG GraphSAGE honors mean/sum/max at every layer. CAGNET currently accepts mean only. DDP uses local CUDA device IDs, not global ranks. `tests/test_pyg_aggregator.py` checks real aggregations against manual and fixed-shape oracles; `test_pipeline_pyg.py` covers configuration and evaluation.

## Preparation and unsupported extensions

Homogeneous partition training requires the global label sidecar and the partitioner's owner map. arxiv partitions must be generated with the same undirected transformation as the ordinary loader; old directed partitions need regeneration. MAG partition generation uses `node_map/paper.pt`, but heterogeneous partition training is rejected until a compatible model and sampler contract exist.

Dataset readiness checks required nonempty raw/split/processed files, including MAG's typed label layout. It does not validate all file contents or large-file checksums. papers100M raw preparation is distinct from full PyG processing. MAG240M streaming is not implemented. The registry explicitly rejects GCN SparseMatMul: SDK expert-selection sparse kernels are not a supported full-graph adjacency primitive for that model.

## Measurement and operational references

Nominal fixed-shape slots include padding; valid seed counts do not. Keep the numerator definition in every throughput result. `tools/measure_window.py` rejects logged evaluation/checkpoint activity within a CSX training window. CPU diagnostics, compiler estimates and wall-clock training rates describe different observations.

Current operational guides are in [GNN docs](../../src/cerebras/modelzoo/models/gnn/docs/): `autotune.md`, `worker_campaign.md`, `worker_diagnostics.md`, `worker_sensitivity.md`, `fixed_shape_gpu.md` and `throughput.md`. Use fresh experiment output directories when source identity or conditions change.

For autotuner extensions, keep search, budgets, resume and ranking in the shared `Study`. The backend in `tools/autotune_backends.py` supplies `prepare_config(base, knobs, model_dir, steps, job_time_sec, warmup_steps)`, `command(config, model_dir)` and `measure(log, start, end, tolerance)`. Measurements carry `throughput`, its `metric` definition and `half_window_check`; a common numeric field does not make nominal slots and actual seeds equivalent.

The worker campaign also locks its output directory. Keep the initial plan/configuration distinct from the mutable execution journal, and retain the lock when changing resume behavior so concurrent campaigns cannot write the same output.

Run focused tests for the changed contract. The complete local suite, from the repository root, is:

    PYTHONPATH=src .venv/bin/python -m unittest discover \
      -s src/cerebras/modelzoo/models/gnn/tests -v

Multiprocessing tests require local IPC; CUDA coverage requires a visible GPU. Passing this suite does not replace CSX execution, large-dataset validation or a multi-node DDP/RPC run.
