"""Backend-specific configuration, launch and measurement for the shared tuner."""

from copy import deepcopy
from pathlib import Path

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
        init["model_dir"] = str(model_dir)
        init["loop"].update(
            num_steps=None,
            max_steps=steps,
            num_epochs=None,
            steps_per_epoch=None,
            eval_frequency=None,
        )
        init["checkpoint"].update(
            steps=None, autoload_last_checkpoint=False, save_initial_checkpoint=False
        )
        init["logging"]["log_steps"] = 10
        init["backend"]["cluster_config"].update(num_csx=1, job_time_sec=job_time_sec)
        loader = trainer["fit"]["train_dataloader"]
        loader.update(knobs)
        loader.update(
            batch_size=4096,
            cache_fraction=None,
            static_batch_cache_size=0,
            use_fake_data=False,
        )
        trainer["fit"].update(ckpt_path=None, val_dataloader=None)
        trainer.update(validate=None, validate_all=None)
        return config

    @staticmethod
    def command(config: Path, model_dir: Path) -> list[str]:
        return [
            "uv",
            "run",
            "--no-sync",
            "--",
            "cszoo",
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
            throughput=result["nominal_slots_per_second"],
            metric="nominal_slots_per_second",
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
            "uv",
            "run",
            "--no-sync",
            "--",
            "python",
            "-u",
            str(GNN / "pyg_graphsage.py"),
            "--config",
            str(config),
        ]

    @staticmethod
    def measure(log: Path, start: int, end: int, tolerance: float) -> dict:
        return measure_pyg.summarize(log, start, end, tolerance)


def get_backend(name: str) -> CSXBackend | PyGBackend:
    return {"csx": CSXBackend, "pyg": PyGBackend}[name]()
