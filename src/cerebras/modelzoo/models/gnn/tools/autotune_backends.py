"""Backend-specific configuration, launch and measurement for the shared tuner."""

from copy import deepcopy
from pathlib import Path
import sys

import yaml

if __package__:
    from . import measure_pyg, measure_window
else:
    import measure_pyg
    import measure_window

GNN = Path(__file__).resolve().parents[1]


class CSXBackend:
    name = "csx"
    remote = True
    persistent_workers = False
    measure_steps = 200
    confirm_steps = 400

    @staticmethod
    def prepare_config(
        base: dict,
        knobs: dict,
        model_dir: Path,
        steps: int,
        job_time_sec: int,
        warmup_steps: int = 40,
    ) -> dict:
        config = deepcopy(base)
        trainer = config["trainer"]
        init = trainer["init"]
        # A custom callback can change the data stream or global-step origin.
        # Preserve the ordinary ModelZoo observers, while requiring an explicit
        # audit before admitting another callback to schedule-based accounting.
        observers = {
            "checkloss",
            "logoptimizerparamgroup",
            "computenorm",
            "modelevalmetrics",
            "rateprofiler",
            "durationprofiler",
            "floputilization",
            "saveperformancedata",
            "dumpavailabletensornames",
            "modelzooparamsmetadata",
            "samplesstreamedinfo",
            "countparams",
            "loginputsummaries",
            "keepncheckpoints",
        }
        for callback in init.get("callbacks") or []:
            if not isinstance(callback, dict) or len(callback) != 1:
                raise ValueError(
                    "Expected one named callback per benchmark callback entry"
                )
            name, settings = next(iter(callback.items()))
            if settings is None:
                continue
            name = name.lower()
            if name in {"globalflags", "scopedtrainflags"}:
                # Debug flags can bypass the real dataloader. Microbatch tiling
                # preserves input ordering, so retain this compiler setting.
                if not isinstance(settings, dict) or any(
                    key != "csx.performance.micro_batch_size" for key in settings
                ):
                    raise ValueError(
                        "Input accounting permits only explicit micro_batch_size flags"
                    )
            elif name not in observers:
                raise ValueError(
                    f"Callback {name!r} has not been audited for fresh continuous input accounting"
                )
        init["model_dir"] = str(model_dir)
        init["loop"].update(
            num_steps=None,
            max_steps=steps,
            num_epochs=None,
            steps_per_epoch=None,
            eval_frequency=None,
            grad_accum_steps=1,
        )
        init["autorestart"] = {"max_num_restarts": 0}
        init["checkpoint"].update(
            steps=None, autoload_last_checkpoint=False, save_initial_checkpoint=False
        )
        init["logging"]["log_steps"] = 10
        init.setdefault("backend", {}).setdefault("cluster_config", {}).update(
            num_csx=1, num_workers_per_csx=1, job_time_sec=job_time_sec
        )
        loader = trainer["fit"]["train_dataloader"]
        if (
            loader.get("data_processor") != "GNNDataProcessor"
            or loader.get("sampling_mode") != "neighbor"
            or loader.get("split") not in (None, "train")
        ):
            raise ValueError(
                "Input tuning requires the real GNN neighbor training loader"
            )
        loader.update(knobs)
        loader.update(
            batch_size=4096,
            cache_fraction=None,
            static_batch_cache_size=0,
            use_fake_data=False,
            measure_batch_accounting=True,
        )
        trainer["fit"].update(ckpt_path=None, val_dataloader=None)
        trainer.update(validate=None, validate_all=None)
        return config

    @staticmethod
    def command(config: Path, model_dir: Path) -> list[str]:
        return [
            sys.executable,
            "-u",
            "-m",
            "cerebras.modelzoo.cli.main",
            "fit",
            str(config),
            "--target_device",
            "CSX",
            "--model_dir",
            str(model_dir),
        ]

    @staticmethod
    def measure(log: Path, start: int, end: int, tolerance: float) -> dict:
        result = measure_window.summarize(log, start, end, 4096, tolerance)
        result.update(
            throughput=result["seed_nodes_per_second"],
            metric="seed_nodes_per_second",
        )
        return result


class PyGBackend:
    name = "pyg"
    remote = False
    persistent_workers = True
    measure_steps = 400
    confirm_steps = 800

    @staticmethod
    def prepare_config(
        base: dict,
        knobs: dict,
        model_dir: Path,
        steps: int,
        job_time_sec: int,
        warmup_steps: int = 40,
    ) -> dict:
        config = CSXBackend.prepare_config(
            base, knobs, model_dir, steps, job_time_sec, warmup_steps
        )
        init = config["trainer"]["init"]
        init.pop("backend", None)
        init["benchmark"] = {"warmup_steps": warmup_steps}
        init["model"]["task"]["compute_eval_metrics"] = False
        # None means automatic GPU caching on the existing PyG feature-fetch path.
        config["trainer"]["fit"]["train_dataloader"]["cache_fraction"] = 0.0
        return config

    @staticmethod
    def command(config: Path, model_dir: Path) -> list[str]:
        return [
            sys.executable,
            "-u",
            str(GNN / "pyg_graphsage.py"),
            "--config",
            str(config),
        ]

    @staticmethod
    def measure(log: Path, start: int, end: int, tolerance: float) -> dict:
        return measure_pyg.summarize(log, start, end, tolerance)


class FixedShapeBackend:
    name = "fixed_shape"
    remote = False
    persistent_workers = True
    measure_steps = 400
    confirm_steps = 400

    @staticmethod
    def prepare_config(base, knobs, model_dir, steps, job_time_sec, warmup_steps=40):
        # GPU handoffs intentionally omit the SDK backend configuration.
        base = deepcopy(base)
        base["trainer"]["init"].setdefault("backend", {}).setdefault(
            "cluster_config", {}
        )
        config = CSXBackend.prepare_config(
            base, knobs, model_dir, steps, job_time_sec, warmup_steps
        )
        init = config["trainer"]["init"]
        init.pop("backend", None)
        init["benchmark"] = {"warmup_steps": warmup_steps}
        init["model"]["task"]["compute_eval_metrics"] = False
        return config

    @staticmethod
    def command(config: Path, model_dir: Path) -> list[str]:
        benchmark = yaml.safe_load(config.read_text())["trainer"]["init"]["benchmark"]
        command = [
            sys.executable,
            "-u",
            str(GNN / "fixed_shape_gpu.py"),
            "--config",
            str(config),
            "--output-dir",
            str(model_dir),
            "--warmup-steps",
            str(benchmark["warmup_steps"]),
        ]
        if benchmark.get("compile", False):
            command.append("--compile")
        return command

    @staticmethod
    def measure(log: Path, start: int, end: int, tolerance: float) -> dict:
        if __package__:
            from . import measure_fixed_shape
        else:
            import measure_fixed_shape
        result = measure_fixed_shape.summarize(log, start, end, tolerance)
        if result["skipped_optimizer_steps"]:
            raise ValueError("AMP skipped optimizer updates in the throughput window")
        return result


def get_backend(name: str) -> CSXBackend | PyGBackend | FixedShapeBackend:
    return {"csx": CSXBackend, "pyg": PyGBackend, "fixed_shape": FixedShapeBackend}[
        name
    ]()
