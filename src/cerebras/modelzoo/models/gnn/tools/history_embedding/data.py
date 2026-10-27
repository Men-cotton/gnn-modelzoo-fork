"""Build frozen sampled neighborhoods from the native GNN graph source."""

import numpy as np
import torch


def load_graph(data_dir, name="ogbn-arxiv", fanout=15, seed=42):
    from cerebras.modelzoo.models.gnn.data_processing.sources import BaseGraphDataSource
    from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
        GraphSAGENeighborSamplerDataset,
    )
    from torch_geometric.utils import to_scipy_sparse_matrix

    source = BaseGraphDataSource(
        dataset_name=name,
        data_dir=str(data_dir),
        current_split="train",
        float_dtype=torch.float16,
        label_dtype=torch.int64,
        adj_normalization_fn=None,
    )
    features, edge_index, labels, split_masks = source.prepare_graph_components()
    # Match native input quantization; accumulate host means in FP32.
    features = features.float()
    # The native source has already canonicalized and symmetrized the edges.
    adjacency = to_scipy_sparse_matrix(edge_index, num_nodes=len(features)).tocsr()
    sampler = GraphSAGENeighborSamplerDataset(
        features=features,
        edge_index=edge_index,
        labels=labels,
        mask=split_masks["train"],
        edge_index_is_undirected=True,
        fanouts=[fanout],
        batch_size=1,
        shuffle=False,
        pad_id=0,
        seed=seed,
    )
    neighbors, mask = sampler._sample_neighbors(
        np.arange(len(features)), np.ones(len(features), dtype=bool), fanout, 0
    )
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
            "sampling": "frozen_native_root_neighbors",
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
