"""Replay a real-data SDK run with the same model and SDK optimizer on CPU.

Reads batch size, seed, step count, mode and learning rate from the run receipt.
Partial windows are checked as partial windows, never as completed epochs.
"""

import argparse
import json
from pathlib import Path

import cerebras.pytorch as cstorch
import torch

from cerebras.modelzoo.models.gnn.tools.history_embedding.core import (
    masked_loss,
    refresh,
)
from cerebras.modelzoo.models.gnn.tools.history_embedding.hardware import (
    SnapshotDataset,
    StaticModel,
)
from cerebras.modelzoo.models.gnn.tools.csx_backend import configure_optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--epoch0", type=Path, required=True)
    parser.add_argument("--epoch1", type=Path)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cstorch.backend("CPU", artifact_dir=str(args.output.parent / "sdk-eager-reference"))
    b = torch.load(args.input / "bundle.pt", weights_only=True)
    if b["model_config"]["dropout"]:
        raise ValueError("Deterministic numerical comparison requires dropout=0")
    model = StaticModel(**b["model_config"])
    model.load_state_dict(b["initial_model"])
    first = json.loads((args.epoch0 / "result.json").read_text())
    optimizer = configure_optimizer(
        model, b["source"]["dataset"], first["learning_rate"]
    )
    mixed = first["precision"]["enabled"]
    if args.atol is None:
        args.atol = 0.01 if mixed else 1e-6
    if args.rtol is None:
        args.rtol = 0.01 if mixed else 1e-5
    skipped_steps = 0
    result, total_steps = {}, 0
    previous_complete = False
    for epoch, output in enumerate(
        p for p in (args.epoch0, args.epoch1) if p is not None
    ):
        receipt = json.loads((output / "result.json").read_text())
        if receipt["status"] != "executed" or receipt["epoch"] != epoch:
            raise ValueError(
                "Expected consecutively executed epochs from initial weights"
            )
        if (
            receipt["learning_rate"] != first["learning_rate"]
            or receipt["mode"] != first["mode"]
        ):
            raise ValueError("Resume must preserve mode and optimizer configuration")
        if receipt["captured_steps"] != receipt["planned_steps"]:
            raise ValueError(
                "Replay requires activation-steps=1 and every step captured"
            )
        if epoch and not previous_complete:
            raise ValueError(
                "Cannot treat a partial preceding window as a complete epoch"
            )
        snapshot = receipt["snapshot"]
        refresh_error = None
        if receipt["mode"] == "history":
            stored = torch.load(snapshot, weights_only=True)
            exact = refresh(model, b["features"], b["neighbors"], b["mask"])
            refresh_error = max(
                (
                    (a - c).abs().max().item()
                    for a, c in zip(stored["histories"], exact)
                ),
                default=0.0,
            )
            for a, c in zip(stored["histories"], exact):
                torch.testing.assert_close(a, c, atol=args.atol, rtol=args.rtol)
        dataset = SnapshotDataset(
            str(args.input / "bundle.pt"),
            snapshot,
            receipt["mode"],
            receipt["batch_size"],
            epoch,
            receipt["seed"],
        )
        rows = [
            json.loads(line)
            for line in (output / "steps.jsonl").read_text().splitlines()
        ]
        steps = receipt["planned_steps"]
        if len(rows) != steps:
            raise ValueError("Missing step records")
        ids, history_error = [], None
        for i in range(steps):
            batch = dataset[(i // len(dataset), i % len(dataset))]
            if i % len(dataset) == 0:
                ids = []
            valid, labels = batch["valid"][0], batch["labels"][0]
            optimizer.zero_grad()
            logits, hidden = model(batch)
            loss = masked_loss(logits, labels, valid)
            torch.testing.assert_close(
                loss.detach(),
                torch.tensor(rows[i]["loss"]),
                atol=args.atol,
                rtol=args.rtol,
            )
            if rows[i]["supervised"] != int((valid & (labels >= 0)).sum()):
                raise AssertionError("Observed supervised-target count mismatch")
            roots = batch["root_ids"][0][valid]
            ids.extend(roots.tolist())
            if len(set(ids)) != len(ids):
                raise AssertionError("Duplicate roots within one input pass")
            if (i + 1) % len(dataset) == 0 and sorted(ids) != sorted(
                b["train_ids"].tolist()
            ):
                raise AssertionError("Complete input pass did not cover training split")
            if receipt["export_history"]:
                capture = torch.load(output / f"capture-{i:05d}.pt", weights_only=True)
                torch.testing.assert_close(roots, capture["root_ids"])
                for actual, expected in zip(capture["histories"], hidden):
                    expected = expected.detach()[valid]
                    torch.testing.assert_close(
                        actual, expected, atol=args.atol, rtol=args.rtol
                    )
                    history_error = max(
                        history_error or 0.0, (actual - expected).abs().max().item()
                    )
            loss.backward()
            skipped = rows[i]["skipped_optimizer_steps"]
            if skipped == skipped_steps:
                optimizer.step()
            elif skipped != skipped_steps + 1:
                raise AssertionError("Invalid skipped optimizer step count")
            skipped_steps = skipped
        previous_complete = steps == len(dataset)
        if previous_complete and sorted(ids) != sorted(b["train_ids"].tolist()):
            raise AssertionError("Complete epoch did not cover the training split")
        checkpoint = cstorch.load(str(output / "checkpoint.mdl"))
        weight_error = 0.0
        for name, value in model.state_dict().items():
            actual = checkpoint["model"][name]
            torch.testing.assert_close(value, actual, atol=args.atol, rtol=args.rtol)
            weight_error = max(weight_error, (value - actual).abs().max().item())
        total_steps += steps
        if checkpoint["weight_step"] != total_steps:
            raise AssertionError("Checkpoint weight version mismatch")
        result[f"epoch{epoch}"] = dict(
            max_weight_abs_error=weight_error,
            max_history_abs_error=history_error,
            max_refresh_abs_error=refresh_error,
            checked_root_ids=sum(row["valid_roots"] for row in rows),
            optimizer_steps=steps,
            complete_epoch=previous_complete,
        )
    result.update(
        status="PASS_against_eager_reference",
        compared_backend=first["backend"],
        atol=args.atol,
        rtol=args.rtol,
        source=b["source"],
        reference="eager CPU model with cstorch.optim.AdamW",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
