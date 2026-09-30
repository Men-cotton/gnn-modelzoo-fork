"""Data-driven plans used by benchmark_launcher and legacy submit wrappers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
CATALOG = ROOT / "benchmark_scripts/jobs.json"
CONFIGS = ROOT / "src/cerebras/modelzoo/models/gnn/configs"


def load_catalog(path=CATALOG):
    value = json.loads(Path(path).read_text())
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "jobs", "sets", "presets"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or not isinstance(value["jobs"], dict)
        or not value["jobs"]
        or not isinstance(value["sets"], dict)
    ):
        raise ValueError("Invalid job catalog schema/version")
    presets = value["presets"]
    if (
        not isinstance(presets, dict)
        or set(presets) != {"worker_diagnostics"}
        or not isinstance(presets["worker_diagnostics"], dict)
        or set(presets["worker_diagnostics"]) != {"workers"}
    ):
        raise ValueError("Invalid diagnostic presets")
    workers = presets["worker_diagnostics"]["workers"]
    if (
        not isinstance(workers, list)
        or not workers
        or any(type(x) is not int or x < 0 for x in workers)
    ):
        raise ValueError("Invalid diagnostic workers")
    fields = {
        "fixed": {"script"},
        "csx": {"configs"},
        "single": {"script", "config_variable"},
        "hpcasia": {"script", "datasets", "seeds", "runs"},
        "sweep": {
            "script",
            "datasets",
            "backends",
            "phases",
            "control_phases",
            "control_worker",
            "workers",
        },
        "pyg": {"script", "hardware_script", "profiles"},
        "driver": {"module", "arguments"},
    }
    for name, job in value["jobs"].items():
        if not re.fullmatch(r"[a-z0-9-]+", name) or not isinstance(job, dict):
            raise ValueError("Invalid job name/definition")
        kind = job.get("kind")
        if (
            not isinstance(kind, str)
            or kind not in fields
            or set(job) != fields[kind] | {"kind"}
        ):
            raise ValueError(f"Invalid fields for job {name}")
        if kind == "csx":
            if not isinstance(job["configs"], list) or not job["configs"]:
                raise ValueError("Empty CSX configs")
            for config in job["configs"]:
                if (
                    not isinstance(config, dict)
                    or set(config) != {"label", "path"}
                    or not all(isinstance(x, str) and x for x in config.values())
                    or not re.fullmatch(r"params_[a-z0-9_]+\.yaml", config["path"])
                ):
                    raise ValueError("Invalid CSX config entry")
        for key in ("script", "hardware_script"):
            if key in job:
                script_path(job[key])
        for key in (
            "datasets",
            "backends",
            "phases",
            "control_phases",
            "runs",
            "arguments",
        ):
            if key in job and (
                not isinstance(job[key], list)
                or not job[key]
                or not all(
                    isinstance(x, str) and x and "\x00" not in x and "\n" not in x
                    for x in job[key]
                )
            ):
                raise ValueError(f"Invalid {key} for job {name}")
        for key in ("seeds", "workers"):
            if key in job and (
                not isinstance(job[key], list)
                or not job[key]
                or not all(type(x) is int and x >= 0 for x in job[key])
            ):
                raise ValueError(f"Invalid {key} for job {name}")
        if kind == "hpcasia" and (
            len(set(job["seeds"])) != len(job["seeds"])
            or any(x >= 2**32 for x in job["seeds"])
            or len(set(job["runs"])) != len(job["runs"])
            or any(
                not re.fullmatch(r"(learning|throughput|cache)_r[1-9][0-9]*", x)
                for x in job["runs"]
            )
        ):
            raise ValueError("Invalid campaign seeds/runs")
        if kind == "sweep" and (
            type(job["control_worker"]) is not int
            or job["control_worker"] < 0
            or not set(job["control_phases"]) <= set(job["phases"])
        ):
            raise ValueError("Invalid sweep controls")
        if kind == "driver" and (
            not isinstance(job["module"], str)
            or not re.fullmatch(r"cerebras\.modelzoo\.[a-zA-Z0-9_.]+", job["module"])
        ):
            raise ValueError("Invalid driver module")
        if kind == "single" and (
            not isinstance(job["config_variable"], str)
            or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", job["config_variable"])
        ):
            raise ValueError("Invalid config variable")
        if kind == "pyg":
            if not isinstance(job["profiles"], list) or not job["profiles"]:
                raise ValueError("Empty PyG profiles")
            for profile in job["profiles"]:
                if (
                    not isinstance(profile, dict)
                    or set(profile) != {"benchmark", "profile", "config_prefix"}
                    or not all(
                        isinstance(x, str) and re.fullmatch(r"[a-z0-9_]+", x)
                        for x in profile.values()
                    )
                ):
                    raise ValueError("Invalid PyG profile")
    for name, members in value["sets"].items():
        if (
            not re.fullmatch(r"[a-z0-9-]+", name)
            or not isinstance(members, list)
            or not members
            or any(not isinstance(x, str) or x not in value["jobs"] for x in members)
        ):
            raise ValueError(f"Invalid set {name}")
    return value


def script_path(relative):
    if not isinstance(relative, str) or "\x00" in relative:
        raise ValueError("Invalid script path")
    path = (ROOT / relative).resolve()
    if (
        Path(relative).is_absolute()
        or not path.is_relative_to(ROOT / "benchmark_scripts")
        or path.suffix != ".pbs"
        or not path.is_file()
    ):
        raise ValueError(f"Invalid PBS script: {relative}")
    return path


def qsub_command(script, variables=None, resources=()):
    """One argv builder for every scheduler entrypoint; never a shell string."""
    script = Path(script).resolve()
    if (
        not script.is_file()
        or not script.is_relative_to(ROOT / "benchmark_scripts")
        or script.suffix != ".pbs"
    ):
        raise ValueError(f"Invalid PBS script: {script}")
    command = ["qsub"]
    for resource in resources:
        if not re.fullmatch(r"elapstim_req=\d{2}:\d{2}:\d{2}", resource):
            raise ValueError("Invalid scheduler resource")
        command += ["-l", resource]
    values = []
    for key, value in (variables or {}).items():
        value = str(value)
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) or any(
            c in value for c in ",\n\r\x00"
        ):
            raise ValueError("Scheduler values must not contain commas or newlines/NUL")
        values.append(key + "=" + value)
    if values:
        command += ["-v", ",".join(values)]
    return command + [str(script)]


def execute_command(command, *, cwd=ROOT, **kwargs):
    if not command or not all(isinstance(x, str) and "\x00" not in x for x in command):
        raise ValueError("Expected a nonempty argv without NUL")
    return subprocess.run(command, cwd=cwd, shell=False, **kwargs)


def config_path(value, *, planned=False):
    if not value:
        raise ValueError("--config is required")
    path = Path(value).resolve()
    if not planned:
        if not path.is_file():
            raise ValueError(f"Config not found: {path}")
        import yaml
        from cerebras.modelzoo.common.utils.run.config_loader import load_params_file

        try:
            config = load_params_file(path)
        except (yaml.YAMLError, TypeError) as exc:
            raise ValueError(f"Invalid YAML config: {path}") from exc
        if not isinstance(config, dict) or not isinstance(config.get("trainer"), dict):
            raise ValueError(f"Config requires a trainer mapping: {path}")
    return str(path)


def positive(value):
    if not re.fullmatch(r"[1-9][0-9]{0,5}", value):
        raise argparse.ArgumentTypeError(
            "Expected positive integer without leading zeros"
        )
    return int(value)


def options(kind):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--json", action="store_true", help="print validated plan without executing"
    )
    if kind == "csx":
        p.add_argument("--config-index", type=positive)
        p.add_argument("--list-configs", action="store_true")
    if kind in ("fixed", "single"):
        p.add_argument("--config", required=True)
    if kind in ("fixed", "sweep", "hpcasia"):
        p.add_argument("--compile", dest="compile", action="store_true")
        p.add_argument("--no-compile", dest="compile", action="store_false")
        p.set_defaults(compile=kind == "hpcasia")
    if kind == "fixed":
        p.add_argument("--measure-input", action="store_true")
        p.add_argument("--measure-neighbor-padding", action="store_true")
    if kind in ("hpcasia", "sweep"):
        p.add_argument("--output", required=True)
        p.add_argument(
            "--dataset",
            choices=("all", "arxiv", "products"),
            default="all" if kind == "hpcasia" else "arxiv",
        )
    if kind == "hpcasia":
        p.add_argument("--hours", choices=(3, 4), type=int, default=4)
    if kind == "sweep":
        p.add_argument(
            "--backend", choices=("both", "fixed_shape", "pyg"), default="both"
        )
        p.add_argument("--phase", default="all")
        p.add_argument("--workers")
        p.add_argument("--pyg-base-config")
        p.add_argument("--budget-sec", type=positive, default=10800)
        p.add_argument("--trial-timeout-sec", type=positive, default=1800)
        p.add_argument("--walltime-hours", type=positive, default=24)
        p.add_argument("--record-resources", action="store_true")
    if kind == "pyg":
        p.add_argument("--only", action="append", default=[])
        p.add_argument(
            "--no-compile",
            action="store_true",
            default=os.environ.get("PYG_NO_COMPILE", "0") == "1",
        )
    return p


def build_plan(job, args):
    """Expand and validate everything before the first subprocess or directory write."""
    kind = job["kind"]
    if kind == "csx":
        if args.config_index is None or not 1 <= args.config_index <= len(
            job["configs"]
        ):
            raise ValueError("--config-index must select one listed CSX config")
        selected = job["configs"][args.config_index - 1]
        config = config_path(CONFIGS / selected["path"])
        if "graphsage" in selected["path"]:
            text = (
                CONFIGS / "components/architectures/input_pipelines/neighbor.yaml"
            ).read_text()
            if re.search(r"^\s*num_workers:\s*(\d+)\s*(?:#.*)?$", text, re.M) is None:
                raise ValueError("Invalid shared num_workers")
        return [
            [str(ROOT / ".venv/bin/cszoo"), "fit", config, "--target_device", "CSX"]
        ]
    script = script_path(job["script"])
    commands = []

    def add(variables=None, hours=None, selected_script=script):
        commands.append(
            qsub_command(
                selected_script,
                variables,
                () if hours is None else (f"elapstim_req={hours:02d}:00:00",),
            )
        )

    if kind in ("fixed", "single"):
        config = config_path(
            args.config, planned=kind == "single" and (args.dry_run or args.json)
        )
        if kind == "single":
            add({job["config_variable"]: config})
        else:
            add(
                dict(
                    FIXED_SHAPE_CONFIG=config,
                    FIXED_SHAPE_COMPILE=int(args.compile),
                    FIXED_SHAPE_MEASURE_NEIGHBOR_PADDING=int(
                        args.measure_neighbor_padding
                    ),
                    FIXED_SHAPE_MEASURE_INPUT=int(args.measure_input),
                )
            )
    elif kind == "pyg":
        known = {x["benchmark"] for x in job["profiles"]} | {
            x["benchmark"] + "/" + x["profile"] for x in job["profiles"]
        }
        if any(x not in known for x in args.only):
            raise ValueError("Unknown PyG --only job/profile")
        add(selected_script=script_path(job["hardware_script"]))
        import yaml

        worker_config = yaml.safe_load(
            (
                CONFIGS / "components/architectures/input_pipelines/neighbor.yaml"
            ).read_text()
        )
        # Existing guard checks the first num_workers declaration, independent of nesting.
        match = re.search(
            r"^\s*num_workers:\s*(\d+)\s*(?:#.*)?$",
            (
                CONFIGS / "components/architectures/input_pipelines/neighbor.yaml"
            ).read_text(),
            re.M,
        )
        if not isinstance(worker_config, dict) or match is None:
            raise ValueError("Invalid shared num_workers")
        for profile in job["profiles"]:
            bench, name = profile["benchmark"], profile["profile"]
            if (
                args.only
                and bench not in args.only
                and bench + "/" + name not in args.only
            ):
                continue
            config_path(CONFIGS / (profile["config_prefix"] + "_" + name + ".yaml"))
            add(
                dict(
                    PYG_BENCHMARK=bench,
                    PYG_RUN_PROFILE=name,
                    PYG_NO_COMPILE=int(args.no_compile),
                )
            )
    else:
        output = str(Path(args.output).absolute())
        if "," in output or any(c.isspace() for c in output) or "\x00" in output:
            raise ValueError("Paths must not contain commas or whitespace/NUL")
        datasets = job["datasets"] if args.dataset == "all" else [args.dataset]
        if kind == "hpcasia":
            for dataset in datasets:
                for seed in job["seeds"]:
                    for run in job["runs"]:
                        add(
                            dict(
                                HPCASIA_DATASET=dataset,
                                HPCASIA_RUN_ID=f"seed_{seed}/{run}",
                                HPCASIA_OUTPUT=output,
                                HPCASIA_TIMEOUT=args.hours * 3600 - 300,
                                HPCASIA_COMPILE=int(args.compile),
                            ),
                            args.hours,
                        )
        else:
            if not (
                args.walltime_hours <= 24
                and args.trial_timeout_sec + 10
                <= args.budget_sec
                <= args.walltime_hours * 3600 - 300
            ):
                raise ValueError(
                    "Require timeout + 10 <= budget <= PBS walltime - 300; walltime 1..24"
                )
            if args.dataset == "all" and args.pyg_base_config:
                raise ValueError(
                    "--pyg-base-config cannot be combined with --dataset all"
                )
            if args.phase != "all" and args.phase not in job["phases"]:
                raise ValueError("Unknown sweep phase")
            if args.workers is not None and not re.fullmatch(
                r"[0-9]+(?:[ \t]+[0-9]+)*", args.workers
            ):
                raise ValueError("Invalid --workers list")
            workers = (
                args.workers.split() if args.workers is not None else job["workers"]
            )
            backends = job["backends"] if args.backend == "both" else [args.backend]
            phases = job["phases"] if args.phase == "all" else [args.phase]
            for dataset in datasets:
                pyg_config = ""
                if args.backend != "fixed_shape":
                    pyg_config = config_path(
                        args.pyg_base_config or CONFIGS / f"autotune/{dataset}_w40.yaml"
                    )
                    if "," in pyg_config or any(c.isspace() for c in pyg_config):
                        raise ValueError("Paths must not contain commas or whitespace")
                if args.backend != "pyg":
                    config_path(CONFIGS / f"fixed_shape_gpu/{dataset}.yaml")
                for backend in backends:
                    for phase in phases:
                        for worker in (
                            [job["control_worker"]]
                            if phase in job["control_phases"]
                            else workers
                        ):
                            add(
                                dict(
                                    GPU_SWEEP_DATASET=dataset,
                                    GPU_SWEEP_OUTPUT=output,
                                    GPU_SWEEP_COMPILE=int(args.compile),
                                    GPU_SWEEP_WORKER=worker,
                                    GPU_SWEEP_PHASE=phase,
                                    GPU_SWEEP_BACKEND=backend,
                                    GPU_SWEEP_PYG_CONFIG=pyg_config,
                                    GPU_SWEEP_BUDGET_SEC=args.budget_sec,
                                    GPU_SWEEP_TRIAL_TIMEOUT_SEC=args.trial_timeout_sec,
                                    GPU_SWEEP_WALLTIME_HOURS=args.walltime_hours,
                                    GPU_SWEEP_RECORD_RESOURCES=int(
                                        args.record_resources
                                    ),
                                ),
                                args.walltime_hours,
                            )
    if not commands:
        raise ValueError("Empty job plan")
    return commands


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = any(
        x in ("--job", "--set") or x.startswith(("--job=", "--set=")) for x in argv
    )
    p = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, add_help=not selected
    )
    select = p.add_mutually_exclusive_group()
    select.add_argument("--job")
    select.add_argument("--set", dest="job_set")
    p.add_argument("--catalog", type=Path, default=CATALOG)
    p.add_argument("--list", action="store_true")
    # Selection and per-job options are parsed separately; each strict parser rejects unknown flags.
    args, remaining = p.parse_known_args(argv)
    catalog = load_catalog(args.catalog)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    if args.list:
        if remaining:
            p.error("Unexpected list arguments")
        print(json.dumps(catalog, indent=2))
        return 0
    if args.job_set:
        if args.job_set not in catalog["sets"]:
            p.error("Unknown set: " + args.job_set)
        names = catalog["sets"][args.job_set]
    elif args.job in catalog["jobs"]:
        names = [args.job]
    else:
        p.error("Unknown or missing job; use --list")
    plans = []
    for name in names:
        job = catalog["jobs"][name]
        if job["kind"] == "driver":
            forwarded = remaining[1:] if remaining[:1] == ["--"] else remaining
            command = [
                sys.executable,
                "-u",
                "-m",
                job["module"],
                *job["arguments"],
                *forwarded,
            ]
            # The driver owns its scientific validation/dry-run, locks and bounded trials.
            plans.append((job, None, [command]))
        else:
            parsed = options(job["kind"]).parse_args(remaining)
            if job["kind"] == "csx" and parsed.list_configs:
                print(json.dumps(job["configs"], indent=2))
                return 0
            plans.append((job, parsed, build_plan(job, parsed)))
    if (
        any(
            parsed is not None
            and job["kind"] != "csx"
            and not (parsed.json or parsed.dry_run)
            for job, parsed, _ in plans
        )
        and shutil.which("qsub") is None
    ):
        raise ValueError("qsub not found; use --dry-run to preview")
    for job, parsed, commands in plans:
        cwd = (
            Path.cwd()
            if job["kind"] == "driver"
            else (CONFIGS.parent if job["kind"] == "csx" else ROOT)
        )
        if parsed is not None and parsed.json:
            print(json.dumps({"cwd": str(cwd), "commands": commands}, indent=2))
            continue
        if parsed is not None and parsed.dry_run:
            for command in commands:
                print("cd " + shlex.quote(str(cwd)) + " && " + shlex.join(command))
            continue
        if job["kind"] == "hpcasia":
            Path(parsed.output).mkdir(parents=True, exist_ok=False)
        for command in commands:
            result = execute_command(command, cwd=cwd)
            if result.returncode:
                return result.returncode
    return 0
