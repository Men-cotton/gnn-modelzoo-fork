"""Local tensor/export checks. This never connects to or compiles for CSX."""

import argparse
import json
from pathlib import Path

import cerebras.pytorch as cstorch
from cerebras.pytorch.utils.data.utils import infer_batch_size
import torch

from cerebras.modelzoo.models.gnn.gpu_measurements import logical_tensor_bytes

from cerebras.modelzoo.models.gnn.tools.history_embedding.hardware import (
    SnapshotDataset,
    StaticModel,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--export",
        action="store_true",
        help="Also torch.export the CPU graph and execute the padded tail",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    bundle = args.input / "bundle.pt"
    snapshot = args.input / "snapshot-0.pt"
    results = dict(
        sdk=cstorch.__version__,
        torch=torch.__version__,
        csx_compile="NOT_RUN",
        csx_execution="NOT_RUN",
    )
    for mode in ("tree", "tree_mean", "history"):
        dataset = SnapshotDataset(
            str(bundle), str(snapshot), mode, args.batch_size, 0, 42
        )
        b = dataset.bundle
        first, last = dataset[0], dataset[len(dataset) - 1]
        if infer_batch_size(first) != 1 or infer_batch_size(last) != 1:
            raise AssertionError("Inconsistent leading batch dimension")
        if (
            int(last["valid"].sum())
            != len(b["train_ids"]) - (len(dataset) - 1) * args.batch_size
        ):
            raise AssertionError("Tail count mismatch")
        result = dict(
            batches=len(dataset),
            input_bytes=logical_tensor_bytes(first),
            tail_targets=int(last["valid"].sum()),
            uniform_batch_axis=True,
            sdk_batch_size=1,
            logical_batch_size=args.batch_size,
            torch_export="NOT_RUN",
        )
        if args.export:
            model = StaticModel(**b["model_config"])
            model.load_state_dict(b["initial_model"])
            model.eval()
            exported = torch.export.export(model, (first,))
            torch.testing.assert_close(exported.module()(last)[0], model(last)[0])
            result["torch_export"] = "PASS"
        results[mode] = result
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
