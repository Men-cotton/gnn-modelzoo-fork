"""One dense or shared-feature SDK job. Use pilot.py for paired admission."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
import cerebras.pytorch as cstorch

from cerebras.modelzoo.models.gnn.gpu_measurements import logical_tensor_bytes

from cerebras.modelzoo.models.gnn.tools.feature_sharing.core import (
    FrozenInputs,
    SharedTreeModel,
    file_hash,
    input_fn,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["dense", "dictionary"], required=True)
    p.add_argument("--backend", choices=["CPU", "CSX"], default="CSX")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--compile-dir", type=Path)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--steps", type=int)
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--capture-logits", action="store_true")
    args = p.parse_args()
    if args.backend == "CSX":
        from cerebras.modelzoo.tools.job_source import require_pinned

        require_pinned()
    dataset = FrozenInputs(args.input.resolve(), args.mode)
    meta = dataset.metadata
    steps = args.steps or len(dataset)
    if steps < len(dataset) or not 0 <= args.warmup_steps < steps:
        p.error("run must cover the complete frozen window, with warmup < steps")
    args.output.mkdir(parents=True, exist_ok=False)
    options = dict(artifact_dir=str(args.output.resolve() / "runtime"))
    if args.backend == "CSX":
        from cerebras.modelzoo.models.gnn.tools.csx_backend import backend_options

        options.update(
            backend_options(
                args.output,
                args.compile_dir,
                [args.input / "input.json"],
                args.compile_only,
            )
        )
    elif args.compile_only:
        p.error("compile-only requires CSX")
    torch.set_num_threads(4)
    cstorch.amp.enable_mixed_precision(args.precision == "bf16")
    if args.precision == "bf16":
        cstorch.amp.set_half_dtype(torch.bfloat16)
    torch.manual_seed(meta["seed"])
    backend = cstorch.backend(args.backend, **options)
    with backend.device:
        model = SharedTreeModel(
            meta["model_config"], meta["logical_batch_size"], args.mode
        )
    model.load_state_dict(
        torch.load(args.input / "initial.pt", map_location="cpu", weights_only=True)
    )
    model.train()
    compiled = cstorch.compile(model, backend)
    optimizer = cstorch.optim.AdamW(
        model.parameters(), lr=0.003, eps=1e-6, weight_decay=0.0
    )
    loader = cstorch.utils.data.DataLoader(
        input_fn, str(args.input.resolve()), args.mode
    )
    torch.manual_seed(meta["seed"])
    records = []

    @cstorch.step_closure
    def capture(loss, logits, sample_id, valid):
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        index = len(records)
        expected = index % len(dataset)
        if int(sample_id.item()) != expected:
            raise ValueError("input window order changed")
        row = dict(
            step=index + 1,
            sample_id=expected,
            loss=float(loss),
            valid_targets=int(valid.sum()),
            elapsed_s=time.perf_counter() - start,
        )
        if args.capture_logits:
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite logits")
            torch.save(logits.cpu(), args.output / f"logits-{index + 1:05d}.pt")
        records.append(row)
        with (args.output / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    @cstorch.checkpoint_closure
    def checkpoint():
        cstorch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps": steps,
            },
            str(args.output / "checkpoint.mdl"),
        )

    @cstorch.trace
    def step(batch):
        optimizer.zero_grad()
        with cstorch.amp.autocast() if args.precision == "bf16" else nullcontext():
            loss, logits = compiled(batch)
        loss.backward()
        optimizer.step()
        capture(
            loss,
            logits if args.capture_logits else (),
            batch["sample_id"],
            batch["target_mask"],
        )
        checkpoint()

    start = time.perf_counter()
    executor = cstorch.utils.data.DataExecutor(
        loader, num_steps=steps, checkpoint_steps=steps, activation_steps=1
    )
    for batch in executor:
        step(batch)
    if not args.compile_only:
        if len(records) != steps or not (args.output / "checkpoint.mdl").is_file():
            raise RuntimeError("missing activations or final checkpoint")
    rate = None
    if not args.compile_only and args.warmup_steps > 0:
        interval = (
            records[-1]["elapsed_s"] - records[args.warmup_steps - 1]["elapsed_s"]
        )
        rate = sum(r["valid_targets"] for r in records[args.warmup_steps :]) / interval
    result = dict(
        status="compiled_not_executed" if args.compile_only else "executed",
        backend=args.backend,
        mode=args.mode,
        precision=args.precision,
        sdk=cstorch.__version__,
        torch=torch.__version__,
        input_sha256=file_hash(args.input / "input.json"),
        planned_steps=steps,
        captured_steps=len(records),
        capture_logits=args.capture_logits,
        full_batch_input_bytes=logical_tensor_bytes(dataset[0]),
        logical_batch_size=meta["logical_batch_size"],
        sdk_batch_size=1,
        capacity=meta["capacity"],
        warmup_steps=args.warmup_steps,
        client_observed_targets_per_s_after_warmup=rate,
        timing_scope="client activation receipts including capture overhead; not isolated device time",
        scope=meta["scope"],
        optimizer="cstorch.optim.AdamW lr=0.003 eps=1e-6 weight_decay=0",
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
