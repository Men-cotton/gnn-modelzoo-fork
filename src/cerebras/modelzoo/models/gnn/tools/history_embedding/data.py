"""Build frozen sampled neighborhoods from the native GNN graph source."""

import numpy as np
import torch


def load_graph(data_dir, name="ogbn-arxiv", fanout=10, seed=20261027):
    from cerebras.modelzoo.models.gnn.data_processing.sources import BaseGraphDataSource
    from torch_geometric.utils import to_scipy_sparse_matrix

    source = BaseGraphDataSource(
        dataset_name=name,
        data_dir=str(data_dir),
        current_split="train",
        float_dtype=torch.float32,
        label_dtype=torch.int64,
        adj_normalization_fn=None,
    )
    features, edge_index, labels, split_masks = source.prepare_graph_components()
    # The native source has already canonicalized and symmetrized the edges.
    adjacency = to_scipy_sparse_matrix(edge_index, num_nodes=len(features)).tocsr()
    neighbors = np.zeros((len(features), fanout), dtype=np.int64)
    mask = np.zeros_like(neighbors, dtype=np.bool_)
    rng = np.random.default_rng(seed)
    for node in range(len(features)):
        row = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]
        selected = rng.choice(row, fanout, replace=False) if len(row) > fanout else row
        neighbors[node, : len(selected)] = selected
        mask[node, : len(selected)] = True
    splits = {
        split: split_masks[split].nonzero(as_tuple=True)[0]
        for split in ("train", "valid", "test")
    }
    return {
        "features": features,
        "labels": labels,
        "neighbors": torch.from_numpy(neighbors),
        "mask": torch.from_numpy(mask),
        "splits": splits,
        "adjacency": adjacency,
        "source": {
            "dataset": name,
            "data_dir": str(data_dir),
            "loader": "BaseGraphDataSource.prepare_graph_components",
            "num_nodes": len(features),
            "undirected_edges": adjacency.nnz,
            "fanout": fanout,
            "sample_seed": seed,
            "sampling": "frozen_uniform_without_replacement",
            "input_dim": features.shape[1],
            "classes": int(labels.max()) + 1,
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
