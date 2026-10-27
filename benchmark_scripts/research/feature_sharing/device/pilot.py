"""Sequential dense/dictionary compile, numerical admission, or bounded timing."""
import argparse
import json
from pathlib import Path
import subprocess
from benchmark_scripts.research.runtime import module_command

import torch
import cerebras.pytorch as cstorch
from benchmark_scripts.research.feature_sharing.device.core import FrozenInputs, file_hash


def compare(output, steps, rtol, atol):
    dirs = [output/m for m in ("dense", "dictionary")]
    records = [[json.loads(line) for line in (d/"steps.jsonl").read_text().splitlines()] for d in dirs]
    if any(len(rows) != steps for rows in records):
        raise ValueError("incomplete activation capture")
    loss_diff = logit_diff = 0.0
    for index, (a,b) in enumerate(zip(*records)):
        if (a["step"],a["sample_id"],a["valid_targets"]) != (b["step"],b["sample_id"],b["valid_targets"]):
            raise ValueError("paired input order mismatch")
        torch.testing.assert_close(torch.tensor(a["loss"]), torch.tensor(b["loss"]), rtol=rtol, atol=atol)
        loss_diff = max(loss_diff, abs(a["loss"]-b["loss"]))
        logits = [torch.load(d/f"logits-{index+1:05d}.pt", weights_only=True).float() for d in dirs]
        torch.testing.assert_close(*logits, rtol=rtol, atol=atol)
        logit_diff = max(logit_diff, float((logits[0]-logits[1]).abs().max()))
    checkpoints = [cstorch.load(str(d/"checkpoint.mdl")) for d in dirs]
    if any(c["steps"] != steps for c in checkpoints):
        raise ValueError("checkpoint step mismatch")
    torch.testing.assert_close(checkpoints[0]["model"], checkpoints[1]["model"], rtol=rtol, atol=atol)
    torch.testing.assert_close(checkpoints[0]["optimizer"], checkpoints[1]["optimizer"], rtol=rtol, atol=atol)
    parameter_diff = max(float((checkpoints[0]["model"][n]-checkpoints[1]["model"][n]).abs().max())
                         for n in checkpoints[0]["model"])
    return dict(rtol=rtol, atol=atol, max_loss_absolute_difference=loss_diff,
                max_logit_absolute_difference=logit_diff, max_parameter_absolute_difference=parameter_diff,
                optimizer_state_compared=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=["compile", "verify", "measure"], default="verify")
    p.add_argument("--backend", choices=["CPU", "CSX"], default="CSX")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--cluster-config", type=Path)
    p.add_argument("--compile-dir", type=Path)
    p.add_argument("--steps", type=int)
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--verification", type=Path, help="passed same-input/backend/precision pilot.json required for measure")
    p.add_argument("--rtol", type=float)
    p.add_argument("--atol", type=float)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    dataset = FrozenInputs(args.input, "dictionary")
    if dataset.metadata["model_config"]["dropout"] != 0:
        p.error("paired pilot requires dropout=0; compiler RNG parity is not established")
    if args.backend == "CSX" and (not args.cluster_config or not args.compile_dir):
        p.error("CSX requires site --cluster-config and --compile-dir")
    if args.phase == "compile" and args.backend != "CSX":
        p.error("compile phase is CSX only")
    steps = args.steps or (30 if args.phase == "measure" else len(dataset))
    if steps < len(dataset) or not 0 <= args.warmup_steps < steps:
        p.error("steps must cover the frozen window, with warmup < steps")
    rtol = args.rtol if args.rtol is not None else (0.0 if args.precision == "fp32" else 0.01)
    atol = args.atol if args.atol is not None else (0.0 if args.precision == "fp32" else 0.01)
    if min(rtol,atol) < 0:
        p.error("negative comparison tolerance")
    result = dict(status="starting", phase=args.phase, backend=args.backend, precision=args.precision,
                  input_sha256=file_hash(args.input/"input.json"), planned_steps=steps, cases={})
    result["runtime_sha256"] = {n:file_hash(Path(__file__).with_name(n))
                                for n in ("core.py", "run.py", "pilot.py")}
    if args.phase == "measure":
        if not args.verification:
            p.error("measure requires --verification from a passed paired numerical pilot")
        verified = json.loads(args.verification.read_text())
        if verified.get("status") != "passed" or verified.get("phase") != "verify" or any(
                verified.get(k) != result[k] for k in ("backend", "precision", "input_sha256", "runtime_sha256")):
            p.error("verification does not match this input/backend/precision")
        result["verification_sha256"] = file_hash(args.verification)
    commands = {}
    for mode in ("dense", "dictionary"):
        cmd = module_command("benchmark_scripts.research.feature_sharing.device.run") + [
               "--input", str(args.input.resolve()), "--output", str((args.output/mode).resolve()),
               "--mode", mode, "--backend", args.backend, "--precision", args.precision,
               "--steps", str(steps), "--warmup-steps", str(args.warmup_steps)]
        if args.backend == "CSX":
            cmd += ["--cluster-config", str(args.cluster_config.resolve()),
                    "--compile-dir", str((args.compile_dir/mode).resolve())]
        if args.phase == "compile":
            cmd += ["--compile-only"]
        if args.phase == "verify":
            cmd += ["--capture-logits"]
        commands[mode] = cmd
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output/"pilot.json"
    try:
        for mode, cmd in commands.items():
            result["active_mode"] = mode
            status_path.write_text(json.dumps(result, indent=2)+"\n")
            print(f"Starting {args.phase}: {mode}", flush=True)
            with (args.output/f"{mode}.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            case = json.loads((args.output/mode/"result.json").read_text())
            expected = "compiled_not_executed" if args.phase == "compile" else "executed"
            if case["status"] != expected or case["input_sha256"] != result["input_sha256"]:
                raise ValueError("case completion/input mismatch")
            result["cases"][mode] = case
        if args.phase == "verify":
            result["comparison"] = compare(args.output, steps, rtol, atol)
        result["status"] = "compiled_not_executed" if args.phase == "compile" else "passed"
        result.pop("active_mode", None)
    except BaseException as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        status_path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
