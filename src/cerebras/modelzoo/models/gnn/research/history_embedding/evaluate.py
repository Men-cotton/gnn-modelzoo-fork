"""Evaluate current checkpoint weights on the full real graph and official validation split."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch
import cerebras.pytorch as cstorch

from cerebras.modelzoo.models.gnn.research.history_embedding.core import HistorySAGE
from cerebras.modelzoo.models.gnn.research.history_embedding.data import (
    full_neighbor_infer,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--split", choices=["valid", "test"], default="valid")
    args = p.parse_args()
    torch.set_num_threads(4)
    b = torch.load(args.bundle, map_location="cpu", weights_only=True)
    ckpt = cstorch.load(str(args.checkpoint))
    model = HistorySAGE(**b["model_config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    n = len(b["features"])
    adjacency = sparse.csr_matrix(
        (
            np.ones(len(b["adjacency_indices"]), dtype=np.bool_),
            b["adjacency_indices"].numpy(),
            b["adjacency_indptr"].numpy(),
        ),
        shape=(n, n),
    )
    start = time.perf_counter()
    logits = full_neighbor_infer(model, b["features"], adjacency, args.device)
    ids = b["splits"][args.split]
    ids = ids[b["labels"][ids] >= 0]
    if not len(ids):
        raise ValueError("No labeled evaluation nodes")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite evaluation logits")
    result = dict(
        split=args.split,
        targets=len(ids),
        accuracy=float((logits[ids].argmax(1) == b["labels"][ids]).float().mean()),
        seconds=time.perf_counter() - start,
        checkpoint=str(args.checkpoint.resolve()),
        weight_step=ckpt["weight_step"],
        device=args.device,
        scope="full original undirected graph; fresh checkpoint weights; no history",
        source=b["source"],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
