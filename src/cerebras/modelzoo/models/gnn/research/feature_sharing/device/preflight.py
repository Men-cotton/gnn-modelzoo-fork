"""CPU numerical and PyTorch export checks; no CSX compilation or submission."""

import argparse
import json
from pathlib import Path

import torch
from cerebras.pytorch.utils.data.utils import infer_batch_size
from cerebras.modelzoo.models.gnn.research.feature_sharing.device.core import (
    FrozenInputs,
    SharedTreeModel,
    file_hash,
    restore,
    tensor_bytes,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("output already exists")
    torch.set_num_threads(4)
    datasets = {m: FrozenInputs(args.input, m) for m in ("dense", "dictionary")}
    meta = datasets["dense"].metadata
    initial = torch.load(args.input / "initial.pt", weights_only=True)
    models = {
        m: SharedTreeModel(meta["model_config"], meta["logical_batch_size"], m)
        for m in datasets
    }
    for model in models.values():
        model.load_state_dict(initial)
    rows = []
    for index in range(len(datasets["dense"])):
        batches = {m: ds[index] for m, ds in datasets.items()}
        assert all(infer_batch_size(b) == 1 for b in batches.values())
        restored = restore(
            batches["dictionary"]["table"],
            batches["dictionary"]["refs"],
            meta["logical_batch_size"],
        )
        for dense, shared in zip(batches["dense"]["node_features"], restored):
            torch.testing.assert_close(dense[0], shared, rtol=0, atol=0)
        outputs, gradients = {}, {}
        for mode, model in models.items():
            model.train()
            model.zero_grad(set_to_none=True)
            torch.manual_seed(meta["seed"] + index)
            loss, logits = model(batches[mode])
            loss.backward()
            outputs[mode] = (loss, logits)
            gradients[mode] = {n: p.grad for n, p in model.named_parameters()}
        torch.testing.assert_close(
            outputs["dense"], outputs["dictionary"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            gradients["dense"], gradients["dictionary"], rtol=0, atol=0
        )
        rows.append(
            dict(
                sample_id=index,
                valid_targets=meta["batches"][index]["valid_targets"],
                input_bytes={m: tensor_bytes(b) for m, b in batches.items()},
                exact=True,
            )
        )
    # Reuse one exported graph on every changed dictionary, reference array and tail mask.
    # This verifies dynamic tensor data within fixed shapes, not the CSX export bridge.
    for mode, model in models.items():
        model.eval()
        exported = torch.export.export(model, (datasets[mode][0],)).module()
        for batch in datasets[mode]:
            torch.testing.assert_close(exported(batch), model(batch), rtol=0, atol=0)
    result = dict(
        status="passed",
        backend="CPU",
        input_sha256=file_hash(args.input / "input.json"),
        batches=rows,
        exact_loss_logits_parameter_gradients=True,
        pytorch_export_changed_inputs=True,
        csx_compilation_tested=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
