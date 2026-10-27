"""Read existing OGB graphs for preliminary local, single-process probes.

The Model Zoo source materializes both edge directions and sorts a large int64
edge tensor.  This optional source uses a memory-mapped processed graph and
SciPy's boolean CSR union instead.  Sorted, unique neighbor lists match PyG's
``to_undirected`` result; target nodes and sampler behavior are unchanged.
No dataset is downloaded, modified, or subsampled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse


def canonical_undirected_csr(
    edge_index: torch.Tensor, num_nodes: int
) -> sparse.csr_matrix:
    """Return PyG-compatible sorted unique neighbors without int64 edge sorting.

    Boolean storage prevents duplicate edge counts from overflowing and keeps
    only presence/absence, as required by the unweighted GraphSAGE sampler.
    This helper is deliberately limited to graphs whose CSR fits int32.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.device.type != "cpu":
        raise ValueError("The local graph source requires CPU edges")
    int32_max = np.iinfo(np.int32).max
    if num_nodes <= 0 or num_nodes > int32_max:
        raise ValueError("The local graph source requires 0 < num_nodes <= int32 max")
    # The union cannot contain more than twice the input edge count.
    if edge_index.shape[1] > int32_max // 2:
        raise ValueError("The graph is too large for this int32 local source")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("edge_index must contain int32 or int64 node IDs")
    edge_array = edge_index.numpy()
    if edge_array.size and (
        int(edge_array.min()) < 0 or int(edge_array.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains a node ID outside the graph")
    rows = edge_array[0].astype(np.int32, copy=True)
    cols = edge_array[1].astype(np.int32, copy=True)
    adjacency = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.bool_), (rows, cols)),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    del rows, cols
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    if adjacency.indices.dtype != np.int32 or adjacency.indptr.dtype != np.int32:
        raise ValueError("The resulting CSR does not fit int32")
    return adjacency


def local_ogb_paths(dataset_name: str, data_dir: str | Path) -> tuple[Path, Path]:
    """Resolve only existing native Model Zoo / OGB directories."""
    split_names = {"ogbn-arxiv": "time", "ogbn-products": "sales_ranking"}
    if dataset_name not in split_names:
        raise ValueError("Only ogbn-arxiv and ogbn-products are supported")
    root = Path(data_dir).expanduser().resolve()
    underscore_name = dataset_name.replace("-", "_")
    candidates = (
        root / dataset_name / underscore_name,
        root / underscore_name,
        root,
    )
    for directory in candidates:
        processed = directory / "processed/geometric_data_processed.pt"
        train_split = directory / "split" / split_names[dataset_name] / "train.csv.gz"
        if processed.is_file() and train_split.is_file():
            return processed, train_split
    raise FileNotFoundError(
        f"Existing processed graph and training split for {dataset_name} "
        f"were not found under {root}; this helper never downloads data"
    )


def load_local_ogb_graph(dataset_name: str, data_dir: str | Path) -> dict[str, Any]:
    """Return native sampler keyword arguments plus a ``metadata`` dictionary.

    Pop ``metadata`` before passing the result to
    ``GraphSAGENeighborSamplerDataset``.  Its own fanouts, batch size, seed,
    shuffling and padding options must still be supplied by the caller.
    ``neighbor_indexes`` supersedes the deliberately empty ``edge_index``.
    """
    from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
        NeighborSliceIndex,
    )
    from torch_geometric.data import Data

    processed, train_split = local_ogb_paths(dataset_name, data_dir)
    # These are the caller's already-trusted local PyG processed files, which
    # contain Data objects rather than a weights-only state dictionary.
    payload = torch.load(processed, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(payload, tuple) or len(payload) not in (2, 3):
        raise ValueError("Expected the existing PyG (data, slices[, data_class]) tuple")
    graph = payload[0]
    if isinstance(graph, dict):
        data_class = payload[2] if len(payload) == 3 else Data
        graph = data_class.from_dict(graph)
    if not isinstance(graph, Data):
        raise ValueError("Expected a homogeneous PyG Data graph")
    if graph.x is None or graph.y is None or graph.edge_index is None:
        raise ValueError("Processed graph must contain features, labels and edges")
    num_nodes = int(graph.x.shape[0])
    if graph.num_nodes != num_nodes:
        raise ValueError("Processed graph node count does not match its features")
    features = graph.x.to(dtype=torch.float16, copy=True)
    labels = graph.y.to(dtype=torch.int64, copy=True)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    if labels.ndim != 1 or labels.shape[0] != num_nodes:
        raise ValueError("Expected one class label per node")
    training_ids = np.loadtxt(train_split, dtype=np.int64, delimiter=",", ndmin=1)
    if training_ids.size == 0 or training_ids.ndim != 1:
        raise ValueError("Expected a nonempty one-column training split")
    if int(training_ids.min()) < 0 or int(training_ids.max()) >= num_nodes:
        raise ValueError("Training split contains a node ID outside the graph")
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[torch.from_numpy(training_ids)] = True
    if int(mask.sum()) != training_ids.size:
        raise ValueError("Training split contains duplicate node IDs")
    source_edges = int(graph.edge_index.shape[1])
    adjacency = canonical_undirected_csr(graph.edge_index, num_nodes)
    forward = NeighborSliceIndex(
        row_offsets=adjacency.indptr, neighbors=adjacency.indices
    )
    metadata = {
        "source": "mmap_processed_ogb_scipy_canonical_undirected_csr",
        "processed_path": str(processed),
        "train_split_path": str(train_split),
        "num_nodes": num_nodes,
        "feature_dim": int(features.shape[1]),
        "source_edges": source_edges,
        "undirected_edges": int(adjacency.nnz),
        "training_nodes": int(training_ids.size),
        "feature_dtype": str(features.dtype),
        "neighbor_index_dtype": str(forward.neighbors.dtype),
        "node_subset": False,
    }
    return {
        "features": features,
        "labels": labels,
        "mask": mask,
        "edge_index": torch.empty((2, 0), dtype=torch.int32),
        "edge_index_is_undirected": True,
        "neighbor_indexes": (forward, None),
        "metadata": metadata,
    }
