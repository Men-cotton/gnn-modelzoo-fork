"""Use the existing single-CSX ALCF settings and pinned worker source."""

from pathlib import Path

from cerebras.modelzoo.common.utils.run.config_loader import load_params_file
from cerebras.modelzoo.models.gnn.research import GNN
from cerebras.modelzoo.tools.job_source import pin_cluster_config, require_pinned


def backend_options(output, compile_dir, inputs, compile_only=False):
    import cerebras.pytorch as cstorch

    require_pinned()
    config = load_params_file(GNN / "configs/autotune/arxiv_w40.yaml")
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
