#!/usr/bin/env python3
"""Run trace-external correctness checks for SDK benchmark analogues."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from cerebras.modelzoo.registry import registry


DEFAULT_TOLERANCE = 1.0e-4


@dataclass(frozen=True)
class BenchmarkCheck:
    name: str
    config_path: Path
    output_reference_error: float | None
    metrics: dict[str, float]


def _iter_config_paths(root: Path) -> Iterable[Path]:
    benchmarks_root = root / "src/cerebras/modelzoo/models/sdk-benchmark"
    return sorted(benchmarks_root.glob("*/configs/*.yaml"))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def _assert_finite_tensor(name: str, key: str, value: torch.Tensor) -> None:
    if value.is_floating_point() or value.is_complex():
        finite = torch.isfinite(value)
        if not bool(finite.all().item()):
            raise AssertionError(f"{name}: output field {key!r} is not finite")


def _max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach().float()
    expected = expected.detach().float()
    return _as_float((actual - expected).abs().amax())


def _get_data_processor(name: str, data_processor_name: str):
    processors = {
        processor.__name__: processor
        for processor in registry.get_data_processors(name)
    }
    try:
        return processors[data_processor_name]
    except KeyError as exc:
        available = ", ".join(sorted(processors))
        raise AssertionError(
            f"{name}: data processor {data_processor_name!r} is not registered "
            f"(available: {available})"
        ) from exc


def _run_one(
    config_path: Path,
    tolerance: float,
    steps: int,
) -> BenchmarkCheck:
    params = _load_yaml(config_path)
    trainer = params["trainer"]
    model_config = trainer["init"]["model"]
    dataloader_config = trainer["validate"]["val_dataloader"]
    name = model_config["name"]

    model_cls = registry.get_model_class(name)
    data_processor_cls = _get_data_processor(
        name, dataloader_config["data_processor"]
    )

    model = model_cls(model_config)
    model.eval()
    dataloader = data_processor_cls(dataloader_config).create_dataloader()

    metrics: dict[str, float] = {}
    output_reference_error: float | None = None

    for step, batch in enumerate(dataloader, start=1):
        if step > steps:
            break
        with torch.no_grad():
            outputs = model(batch)
        if not isinstance(outputs, dict):
            raise AssertionError(f"{name}: forward did not return a dict")
        if "loss" not in outputs:
            raise AssertionError(f"{name}: forward output is missing 'loss'")

        for key, value in outputs.items():
            if torch.is_tensor(value):
                _assert_finite_tensor(name, key, value)

        scalar_metrics = {
            key: _as_float(value)
            for key, value in outputs.items()
            if torch.is_tensor(value)
            and value.numel() == 1
            and (
                key == "loss"
                or "error" in key.lower()
                or "residual" in key.lower()
            )
        }
        metrics.update(scalar_metrics)

        error_values = {
            key: value
            for key, value in scalar_metrics.items()
            if key == "loss" or "error" in key.lower()
        }
        for key, value in error_values.items():
            if not math.isfinite(value) or value > tolerance:
                raise AssertionError(
                    f"{name}: {key}={value:g} exceeds tolerance {tolerance:g}"
                )

        if "output" in outputs and "reference" in outputs:
            output_reference_error = _max_abs_diff(
                outputs["output"], outputs["reference"]
            )
        elif "output_real" in outputs and "reference_real" in outputs:
            real_error = _max_abs_diff(
                outputs["output_real"], outputs["reference_real"]
            )
            imag_error = _max_abs_diff(
                outputs["output_imag"], outputs["reference_imag"]
            )
            output_reference_error = max(real_error, imag_error)
            metrics["real_output_reference_max_abs"] = real_error
            metrics["imag_output_reference_max_abs"] = imag_error
        else:
            raise AssertionError(
                f"{name}: forward output has no output/reference pair"
            )

        metrics["output_reference_max_abs"] = output_reference_error
        if output_reference_error > tolerance:
            raise AssertionError(
                f"{name}: output/reference max error "
                f"{output_reference_error:g} exceeds tolerance {tolerance:g}"
            )

    if not metrics:
        raise AssertionError(f"{name}: dataloader produced no samples")

    return BenchmarkCheck(
        name=name,
        config_path=config_path,
        output_reference_error=output_reference_error,
        metrics=metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Instantiate every sdk_benchmark registry/config pair, run CPU "
            "forward passes, and assert returned correctness metrics outside "
            "the traced model path."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of this script directory.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        help=(
            "Model name to check, for example "
            "sdk_benchmark/gemv-checkerboard-pattern. May be repeated."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of dataloader samples to check per benchmark.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Maximum allowed loss/error/output-reference absolute error.",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")

    selected = set(args.benchmark or [])
    checked: list[BenchmarkCheck] = []
    for config_path in _iter_config_paths(args.root):
        params = _load_yaml(config_path)
        name = params["trainer"]["init"]["model"]["name"]
        if selected and name not in selected:
            continue
        checked.append(_run_one(config_path, args.tolerance, args.steps))

    if selected:
        missing = selected - {check.name for check in checked}
        if missing:
            raise AssertionError(
                "Requested benchmark(s) not found: " + ", ".join(sorted(missing))
            )

    print(
        f"verified {len(checked)} sdk benchmark analogue(s) "
        f"with tolerance {args.tolerance:g}"
    )
    for check in checked:
        metric_text = ", ".join(
            f"{key}={value:.6g}" for key, value in sorted(check.metrics.items())
        )
        print(f"{check.name}: {metric_text}")


if __name__ == "__main__":
    main()
