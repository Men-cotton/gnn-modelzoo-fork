"""Use the existing single-CSX ALCF settings and pinned worker source."""

from pathlib import Path

from cerebras.modelzoo.common.utils.run.config_loader import load_params_file
from cerebras.modelzoo.tools.job_source import pin_cluster_config, require_pinned

WARMUP_STEPS = 40


def load_config(dataset="arxiv"):
    name = dataset.removeprefix("ogbn-")
    return load_params_file(
        Path(__file__).resolve().parents[1] / f"configs/autotune/{name}_w40.yaml"
    )


def configure_optimizer(model, dataset, lr=None):
    import cerebras.pytorch as cstorch
    from cerebras.modelzoo.common.optim_utils import configure_param_groups
    from cerebras.modelzoo.models.gnn.gpu_policy import adamw_kwargs

    options = adamw_kwargs(load_config(dataset)["trainer"]["init"])
    if lr is not None:
        options["lr"] = lr
    return cstorch.optim.AdamW(configure_param_groups(model, options), **options)


def loader_options(backend, dataset):
    config = load_config(dataset)["trainer"]["fit"]["train_dataloader"]
    if backend == "CPU":
        return dict(num_workers=0)
    return {
        key: config[key]
        for key in (
            "num_workers",
            "prefetch_factor",
            "persistent_workers",
            "pin_memory",
        )
    }


def configure_precision(backend, dataset):
    import cerebras.pytorch as cstorch

    if backend == "CPU":
        cstorch.amp.enable_mixed_precision(False)
        return {"enabled": False, "fp16_type": "float32"}
    precision = load_config(dataset)["trainer"]["init"]["precision"]
    cstorch.amp.enable_mixed_precision(precision["enabled"])
    cstorch.amp.set_half_dtype(precision["fp16_type"])
    return precision


def backend_options(
    output, compile_dir, inputs, compile_only=False, *, dataset="arxiv"
):
    import cerebras.pytorch as cstorch

    require_pinned()
    config = load_config(dataset)
    pin_cluster_config(config)
    cluster = config["trainer"]["init"]["backend"]["cluster_config"]
    compile_dir = (compile_dir or output / "compile").resolve()
    compile_dir.mkdir(parents=True, exist_ok=True)
    # Client and input Workers use the same ALCF shared filesystem. The SDK
    # discovers the management endpoint/authentication just as in ordinary GNN jobs.
    cluster["mount_dirs"] = list(
        dict.fromkeys(
            [
                *cluster["mount_dirs"],
                str(output.resolve()),
                str(compile_dir),
                *(
                    str(Path(path).resolve().parent)
                    for path in inputs
                    if path is not None
                ),
            ]
        )
    )
    # A dictionary or local neighbor index addresses the whole logical batch.
    cstorch.backends.csx.performance.micro_batch_size = None
    return dict(
        cluster_config=cstorch.distributed.ClusterConfig(**cluster),
        compile_dir=str(compile_dir),
        compile_only=compile_only,
    )
