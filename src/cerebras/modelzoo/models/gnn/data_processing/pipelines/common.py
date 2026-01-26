from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch_geometric.data import Data as PyGData, HeteroData
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected

import cerebras.pytorch as cstorch

try:
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:  # pragma: no cover - optional dependency
    PygNodePropPredDataset = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdgeIndexAdjacency:
    """Container for edge_index / edge_weight tensors."""

    edge_index: torch.Tensor
    edge_weight: torch.Tensor

    def to(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "EdgeIndexAdjacency":
        target_dtype = torch.int32 if cstorch.use_cs() else torch.long
        edge_index = self.edge_index
        if edge_index.dtype != target_dtype or (
            device is not None and edge_index.device != device
        ):
            edge_index = edge_index.to(
                device=device or edge_index.device, dtype=target_dtype
            )
        edge_weight = self.edge_weight
        if device is not None or dtype is not None:
            edge_weight = edge_weight.to(device=device, dtype=dtype or edge_weight.dtype)
        return EdgeIndexAdjacency(edge_index=edge_index, edge_weight=edge_weight)


def normalize_adj_gcn(adj: sp.spmatrix, add_self_loops: bool = True) -> sp.spmatrix:
    """
    Symmetrically normalize adjacency matrix for GCN.
    (D^-0.5 * A_tilde * D^-0.5), where A_tilde = A + I.
    Other GNNs might require different or no normalization.
    """
    if add_self_loops:
        adj_ = adj + sp.eye(adj.shape[0], dtype=adj.dtype)
        adj_.data = np.clip(adj_.data, 0, 1)
    else:
        adj_ = adj

    adj_ = adj_.tocoo()
    rowsum = np.array(adj_.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(
        rowsum, -0.5, where=rowsum > 0, out=np.zeros_like(rowsum, dtype=float)
    )

    # Efficiently compute D^-0.5 * A * D^-0.5 with element-wise multiplication,
    # which is significantly faster than sparse matrix multiplication.
    normalized_data = adj_.data * d_inv_sqrt[adj_.row] * d_inv_sqrt[adj_.col]
    return sp.coo_matrix((normalized_data, (adj_.row, adj_.col)), shape=adj_.shape)


def sparse_scipy_to_torch_coo(
    sp_matrix: sp.spmatrix,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Converts a SciPy COO sparse matrix to a PyTorch COO sparse tensor."""
    sp_matrix = sp_matrix.tocoo()

    indices = torch.from_numpy(np.vstack((sp_matrix.row, sp_matrix.col))).long()
    values = torch.from_numpy(sp_matrix.data)

    # Determine the target dtype, converting float64 to float32 by default
    # as float64 is rarely needed in deep learning.
    final_dtype = dtype or values.dtype
    if final_dtype == torch.float64:
        final_dtype = torch.float32

    return torch.sparse_coo_tensor(
        indices,
        values.to(final_dtype),
        torch.Size(sp_matrix.shape),
        device=device,
    ).coalesce()


def sparse_scipy_to_edge_tensors(
    sp_matrix: sp.spmatrix,
    *,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts a SciPy COO matrix into edge_index / edge_weight tensors."""
    sp_coo = sp_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((sp_coo.row, sp_coo.col))).long()
    values = torch.from_numpy(sp_coo.data).to(dtype=dtype)
    return indices, values


class BaseGraphDataSource:
    """Common utilities for loading and transforming graph datasets."""

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        current_split: str,
        float_dtype: torch.dtype,
        label_dtype: torch.dtype,
        adj_normalization_fn: Optional[Callable[[sp.spmatrix, bool], sp.spmatrix]],
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.current_split = current_split
        self.float_dtype = float_dtype
        self.label_dtype = label_dtype
        self.adj_normalization_fn = adj_normalization_fn
        self._graph_data_cache: Optional[PyGData] = None

    def _get_planetoid_raw_file_names(self, dataset_name_lower: str) -> List[str]:
        names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [f"ind.{dataset_name_lower}.{name}" for name in names]

    def _load_planetoid_data(self) -> PyGData:
        dataset_lower = self.dataset_name.lower()
        if dataset_lower not in ["cora", "citeseer", "pubmed"]:
            raise ValueError(
                f"Dataset '{self.dataset_name}' is not a supported Planetoid dataset."
            )

        dataset_root = os.path.join(self.data_dir, self.dataset_name)
        raw_dir = os.path.join(dataset_root, "raw")

        missing_files = []
        for fname in self._get_planetoid_raw_file_names(dataset_lower):
            path = os.path.join(raw_dir, fname)
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            error_message = (
                f"Required raw data file(s) not found for Planetoid dataset '{self.dataset_name}'.\n"
                f"The GNN data pipeline does not auto-download datasets.\n"
                f"Ensure the dataset is located at '{raw_dir}'. Missing files:\n"
            )
            error_message += "".join(f"  - {path}\n" for path in missing_files)
            raise FileNotFoundError(error_message)

        logger.info(
            "All raw files for Planetoid dataset '%s' found in '%s'. Proceeding to load.",
            self.dataset_name,
            raw_dir,
        )

        dataset = Planetoid(root=self.data_dir, name=self.dataset_name, split="public")
        processed_path = os.path.join(
            dataset_root, "processed", dataset.processed_file_names[0]
        )
        if not os.path.exists(processed_path):
            logger.info(
                "Processed file not found at '%s'. PyG will generate it from raw data.",
                processed_path,
            )
        return cast(PyGData, dataset[0])

    def _load_ogb_data(self) -> PyGData:
        dataset_lower = self.dataset_name.lower()
        if dataset_lower in {"mag240m", "mag240m-lsc", "ogbn-mag240m"}:
            # MAG240M is extremely large and requires a custom streaming pipeline.
            raise NotImplementedError(
                "MAG240M requires a specialized data loader that streams from disk. "
                "Consider implementing a custom pipeline before enabling this dataset."
            )

        if PygNodePropPredDataset is None:
            raise ImportError(
                "ogb is required to load datasets whose names start with 'ogbn-'. "
                "Install it via `uv pip install ogb` before using these configs."
            )

        dataset_root = os.path.join(self.data_dir, self.dataset_name)
        try:
            dataset = PygNodePropPredDataset(
                name=self.dataset_name,
                root=dataset_root,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize PygNodePropPredDataset for '{self.dataset_name}'."
            ) from exc

        graph = dataset[0]
        split_idx = dataset.get_idx_split()
        if isinstance(graph, HeteroData) or getattr(graph, "x_dict", None) is not None:
            graph = self._convert_hetero_ogb_graph(
                hetero_graph=graph,
                dataset_lower=dataset_lower,
                split_idx=split_idx,
            )
        else:
            graph = cast(PyGData, graph)
            num_nodes = graph.num_nodes
            if num_nodes is None:
                if hasattr(graph, "x") and graph.x is not None:
                    num_nodes = graph.x.size(0)
                else:
                    raise ValueError(
                        f"Unable to determine number of nodes for dataset '{self.dataset_name}'. "
                        "Ensure graph.x is populated before loading this dataset."
                    )
            self._assign_split_masks(graph, split_idx, int(num_nodes))
            graph.y = self._sanitize_labels(graph.y)
        return graph

    @staticmethod
    def _sanitize_labels(labels: Tensor) -> Tensor:
        sanitized = labels
        if not torch.is_tensor(sanitized):
            sanitized = torch.as_tensor(sanitized)
        if sanitized.ndim > 1 and sanitized.shape[1] == 1:
            sanitized = sanitized.squeeze(1)
        if sanitized.dtype != torch.long:
            sanitized = sanitized.to(torch.long)
        return sanitized

    @staticmethod
    def _normalize_split_indices(indices) -> torch.Tensor:
        if isinstance(indices, dict):
            for key in ("paper", "node", "target"):
                if key in indices:
                    return BaseGraphDataSource._normalize_split_indices(indices[key])
            raise ValueError(
                "Unsupported split index format encountered while processing OGB dataset."
            )
        tensor = torch.as_tensor(indices, dtype=torch.long)
        if tensor.dim() != 1:
            tensor = tensor.view(-1)
        return tensor

    def _assign_split_masks(
        self,
        graph: PyGData,
        split_idx,
        num_nodes: int,
    ) -> None:
        name_mapping = {
            "train": "train",
            "training": "train",
            "valid": "val",
            "validation": "val",
            "val": "val",
            "test": "test",
            "test-dev": "test",
            "test-dev-shuffled": "test",
        }
        for split_name, indices in split_idx.items():
            mapped_name = name_mapping.get(split_name.lower())
            if not mapped_name:
                continue
            idx_tensor = self._normalize_split_indices(indices)
            if idx_tensor.numel() == 0:
                continue
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            mask[idx_tensor] = True
            graph[f"{mapped_name}_mask"] = mask

    def _convert_hetero_ogb_graph(
        self,
        hetero_graph,
        *,
        dataset_lower: str,
        split_idx,
    ) -> PyGData:
        if dataset_lower != "ogbn-mag":
            raise NotImplementedError(
                f"Heterogeneous OGB dataset '{self.dataset_name}' is not supported."
            )
        if "paper" not in hetero_graph.x_dict:
            raise ValueError(
                "Expected 'paper' node features in ogbn-mag dataset; none were found."
            )
        paper_features = hetero_graph.x_dict["paper"]
        if paper_features is None:
            raise ValueError(
                "ogbn-mag 'paper' features are missing. "
                "Ensure the dataset was processed with node embeddings enabled."
            )
        paper_labels = hetero_graph.y_dict.get("paper")
        if paper_labels is None:
            raise ValueError(
                "Expected 'paper' node labels in ogbn-mag dataset; none were found."
            )
        paper_labels = self._sanitize_labels(paper_labels)
        edge_dict = getattr(hetero_graph, "edge_index_dict", {})
        if ("paper", "cites", "paper") in edge_dict:
            edge_index = edge_dict[("paper", "cites", "paper")]
        elif ("paper", "paper") in edge_dict:
            edge_index = edge_dict[("paper", "paper")]
        else:
            candidate = next(
                (
                    edge_dict[key]
                    for key in edge_dict
                    if key[0] == "paper" and key[-1] == "paper"
                ),
                None,
            )
            if candidate is None:
                raise ValueError(
                    "Expected a paper-to-paper edge relation in ogbn-mag dataset."
                )
            edge_index = candidate
        edge_index = to_undirected(edge_index, num_nodes=paper_features.size(0))
        graph = PyGData(
            x=paper_features,
            edge_index=edge_index,
            y=paper_labels,
        )
        self._assign_split_masks(graph, split_idx, paper_features.size(0))
        return graph

    def _load_reddit_data(self) -> PyGData:
        dataset_root = os.path.join(self.data_dir, "Reddit")
        if not os.path.isdir(dataset_root):
            logger.info(
                "Reddit dataset directory '%s' not found. PyG will download it automatically.",
                dataset_root,
            )
        dataset = Reddit(root=dataset_root)
        return cast(PyGData, dataset[0])

    def load_graph(self) -> PyGData:
        if self._graph_data_cache is not None:
            return self._graph_data_cache

        dataset_lower = self.dataset_name.lower()
        if dataset_lower in ["cora", "citeseer", "pubmed"]:
            graph = self._load_planetoid_data()
        elif dataset_lower == "reddit":
            graph = self._load_reddit_data()
        elif dataset_lower.startswith("ogbn-"):
            graph = self._load_ogb_data()
        else:
            raise ValueError(
                f"Unsupported dataset_name: {self.dataset_name}. "
                "Extend BaseGraphDataSource to handle this dataset."
            )
        self._graph_data_cache = graph
        return graph

    def _get_common_graph_elements(self) -> Tuple[PyGData, Tensor, Tensor]:
        """Loads graph and prepares common feature and label tensors."""
        graph = self.load_graph()
        features = cast(Tensor, graph.x).to(dtype=self.float_dtype)
        labels = cast(Tensor, graph.y).to(dtype=self.label_dtype)
        if labels.ndim > 1 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        return graph, features, labels

    def load_full_graph(
        self,
    ) -> Tuple[Tensor, Union[EdgeIndexAdjacency, Tensor], Tensor, Tensor]:
        graph, features, labels = self._get_common_graph_elements()

        num_nodes = graph.num_nodes
        edge_index = graph.edge_index
        if edge_index is None:
            logger.warning(
                "Dataset %s has no edges; using an empty edge set for %d nodes.",
                self.dataset_name,
                num_nodes,
            )
            edge_index = torch.empty((2, 0), dtype=torch.long)

        edge_index_undirected = to_undirected(edge_index, num_nodes=num_nodes)
        adj_sp = to_scipy_sparse_matrix(edge_index_undirected, num_nodes=num_nodes)

        # Apply normalization if a function is provided.
        adj_processed_sp = (
            self.adj_normalization_fn(adj_sp, True)
            if self.adj_normalization_fn
            else adj_sp
        )

        edge_index, edge_weight = sparse_scipy_to_edge_tensors(
            adj_processed_sp, dtype=self.float_dtype
        )
        mask_attr_name = f"{self.current_split}_mask"
        if not hasattr(graph, mask_attr_name):
            raise ValueError(
                f"Split mask '{mask_attr_name}' not found for {self.dataset_name}."
            )
        mask = getattr(graph, mask_attr_name).bool()

        logger.info(
            "Loaded %s split '%s': %d nodes, %d features, %d edges, %d masked nodes.",
            self.dataset_name,
            self.current_split,
            features.shape[0],
            features.shape[1],
            edge_weight.numel(),
            mask.sum().item(),
        )
        if cstorch.use_cs():
            adjacency_dense = torch.zeros(
                (num_nodes, num_nodes), dtype=self.float_dtype
            )
            if edge_index.numel() > 0:
                edge_index_long = edge_index.to(dtype=torch.long)
                adjacency_dense.index_put_(
                    (edge_index_long[0], edge_index_long[1]),
                    edge_weight.to(dtype=self.float_dtype),
                    accumulate=True,
                )
            adjacency = adjacency_dense.unsqueeze(0)
            features = features.unsqueeze(0)
            labels = labels.unsqueeze(0)
            mask = mask.unsqueeze(0)
        else:
            adjacency = EdgeIndexAdjacency(
                edge_index=edge_index, edge_weight=edge_weight
            )

        return features, adjacency, labels, mask

    def prepare_graph_components(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        graph, features, labels = self._get_common_graph_elements()

        edge_index = graph.edge_index
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)

        split_masks: Dict[str, Tensor] = {}
        for split in ("train", "val", "test"):
            attr = f"{split}_mask"
            if not hasattr(graph, attr):
                raise ValueError(
                    f"Expected mask '{attr}' on dataset '{self.dataset_name}'."
                )
            split_masks[split] = getattr(graph, attr).bool()
        return features, edge_index, labels, split_masks


__all__ = [
    "BaseGraphDataSource",
    "EdgeIndexAdjacency",
    "normalize_adj_gcn",
    "sparse_scipy_to_torch_coo",
    "sparse_scipy_to_edge_tensors",
]
