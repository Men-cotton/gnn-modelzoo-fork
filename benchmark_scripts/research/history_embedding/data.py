"""Local OGB source and a frozen uniform-without-replacement sampled graph."""

import numpy as np
import torch


def load_graph(data_dir, name="ogbn-arxiv", fanout=10, seed=20261027):
    from benchmark_scripts.research.logs.local_ogb_graph import (
        canonical_undirected_csr,
        local_ogb_paths,
    )
    from torch_geometric.data import Data

    path, train_path = local_ogb_paths(name, data_dir)
    payload = torch.load(path, mmap=True, weights_only=False)
    graph = payload[0]
    if isinstance(graph, dict):
        graph = (payload[2] if len(payload) == 3 else Data).from_dict(graph)
    adjacency = canonical_undirected_csr(graph.edge_index, len(graph.x))
    neighbors = np.zeros((len(graph.x), fanout), dtype=np.int64)
    mask = np.zeros_like(neighbors, dtype=np.bool_)
    rng = np.random.default_rng(seed)
    for node in range(len(graph.x)):
        row = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]
        selected = rng.choice(row, fanout, replace=False) if len(row) > fanout else row
        neighbors[node, : len(selected)] = selected
        mask[node, : len(selected)] = True
    splits = {
        split: torch.from_numpy(
            np.loadtxt(train_path.with_name(split + ".csv.gz"), dtype=np.int64, ndmin=1)
        )
        for split in ("train", "valid", "test")
    }
    return {
        "features": graph.x.float().contiguous(),
        "labels": graph.y.flatten().long(),
        "neighbors": torch.from_numpy(neighbors),
        "mask": torch.from_numpy(mask),
        "splits": splits,
        "adjacency": adjacency,
        "source": {
            "dataset": name,
            "path": str(path),
            "num_nodes": len(graph.x),
            "undirected_edges": adjacency.nnz,
            "fanout": fanout,
            "sample_seed": seed,
            "sampling": "frozen_uniform_without_replacement",
            "input_dim": graph.x.shape[1],
            "classes": int(graph.y.max()) + 1,
            "split_counts": {k: len(v) for k, v in splits.items()},
        },
    }


@torch.no_grad()
def full_neighbor_infer(model, features, adjacency, device, chunk=2048):
    """Full original graph evaluation; neither history nor sampled edges used."""
    deg = np.diff(adjacency.indptr).clip(min=1)
    values = torch.from_numpy(
        np.repeat(1.0 / deg, np.diff(adjacency.indptr)).astype(np.float32)
    )
    operator = torch.sparse_csr_tensor(
        torch.from_numpy(adjacency.indptr.astype(np.int64)),
        torch.from_numpy(adjacency.indices.astype(np.int64)),
        values,
        size=adjacency.shape,
        device=device,
    )
    training = model.training
    model.eval()
    x = features.to(device)
    for layer in model.layers:
        mean = torch.sparse.mm(operator, x)
        x = torch.cat(
            [
                layer(x[s : s + chunk], mean[s : s + chunk])
                for s in range(0, len(x), chunk)
            ]
        )
    logits = torch.cat(
        [model.classifier(x[s : s + chunk]) for s in range(0, len(x), chunk)]
    )
    model.train(training)
    return logits.cpu()
