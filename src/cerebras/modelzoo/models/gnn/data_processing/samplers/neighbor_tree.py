from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

import cerebras.pytorch as cstorch
from cerebras.modelzoo.models.gnn.worker_validation import validate_num_workers

from .caching import GraphCache
from ..sources.base import BaseGraphDataSource


@dataclass(frozen=True)
class NeighborSliceIndex:
    row_offsets: np.ndarray
    neighbors: np.ndarray

    @classmethod
    def from_edge_index(
        cls,
        rows: Tensor,
        cols: Tensor,
        num_nodes: int,
    ) -> "NeighborSliceIndex":
        if rows.numel() == 0:
            return cls(
                row_offsets=np.zeros(num_nodes + 1, dtype=np.int32),
                neighbors=np.empty(0, dtype=np.int32),
            )

        row_cpu = rows.to(device="cpu", dtype=torch.int32).contiguous()
        col_cpu = cols.to(device="cpu", dtype=torch.int32).contiguous()

        counts = torch.bincount(row_cpu, minlength=num_nodes)
        row_offsets = torch.empty(num_nodes + 1, dtype=torch.long)
        row_offsets[0] = 0
        row_offsets[1:] = torch.cumsum(counts, dim=0)

        sort_order = torch.argsort(row_cpu, stable=True)
        sorted_neighbors = col_cpu.index_select(0, sort_order)

        row_offset_dtype = (
            torch.int32
            if int(row_offsets[-1].item()) <= np.iinfo(np.int32).max
            else torch.int64
        )
        return cls(
            row_offsets=row_offsets.to(dtype=row_offset_dtype).cpu().numpy(),
            neighbors=sorted_neighbors.cpu().numpy(),
        )


class GraphSAGENeighborSamplerDataset(Dataset):
    """Deterministic neighbor sampler yielding static-shape batches."""

    def __init__(
        self,
        features: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        mask: Tensor,
        fanouts: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool,
        pad_id: int,
        seed: int,
        edge_index_is_undirected: bool,
        graph_cache: Optional[GraphCache] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0 for GraphSAGE neighbor sampling.")
        if not fanouts:
            raise ValueError(
                "fanouts must be provided for GraphSAGE neighbor sampling."
            )

        self.features = features
        self.labels = labels
        self.mask = mask.bool()
        self.fanouts = list(fanouts)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_id = pad_id
        self.seed = seed
        self.num_nodes = features.size(0)
        self.graph_cache = graph_cache

        self._forward_index = NeighborSliceIndex.from_edge_index(
            edge_index[0],
            edge_index[1],
            self.num_nodes,
        )
        self._reverse_index = None
        if not edge_index_is_undirected:
            self._reverse_index = NeighborSliceIndex.from_edge_index(
                edge_index[1],
                edge_index[0],
                self.num_nodes,
            )
        self._target_nodes = (
            torch.nonzero(self.mask, as_tuple=False).squeeze(1).cpu().numpy()
        )
        if self._target_nodes.size == 0:
            raise ValueError("Split has no target nodes; cannot construct batches.")

        self._ordered_targets = self._order_targets(self._target_nodes)
        self._num_batches = math.ceil(self._ordered_targets.size / self.batch_size)

    def __len__(self) -> int:
        return self._num_batches

    def __getitem__(self, index: int) -> Dict[str, List[Tensor] | Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Batch index {index} out of range for dataset length {len(self)}."
            )

        start = index * self.batch_size
        end = min(start + self.batch_size, self._ordered_targets.size)
        real_count = end - start

        target_nodes = np.full(self.batch_size, self.pad_id, dtype=np.int64)
        target_mask = np.zeros(self.batch_size, dtype=bool)
        if real_count > 0:
            target_nodes[:real_count] = self._ordered_targets[start:end]
            target_mask[:real_count] = True

        layer_nodes, layer_masks, neighbor_masks = self._sample_layers(
            target_nodes, target_mask
        )
        node_features = self._gather_features(layer_nodes, layer_masks)

        target_indices = torch.from_numpy(target_nodes)
        labels = self.labels[target_indices].clone()
        target_mask_tensor = torch.from_numpy(target_mask).bool()
        if labels.ndim > 1 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels[~target_mask_tensor] = 0

        node_masks_tensors = [
            torch.from_numpy(mask.astype(bool)).bool() for mask in layer_masks
        ]
        neighbor_masks_tensors = [
            torch.from_numpy(mask.astype(bool)).bool() for mask in neighbor_masks
        ]

        node_features = [
            feats.view(self.batch_size, -1, feats.size(-1)) for feats in node_features
        ]
        node_masks_tensors = [
            mask.view(self.batch_size, -1) for mask in node_masks_tensors
        ]
        neighbor_masks_tensors = [
            mask.view(self.batch_size, -1, mask.size(-1))
            for mask in neighbor_masks_tensors
        ]

        return {
            "node_features": node_features,
            "node_masks": node_masks_tensors,
            "neighbor_masks": neighbor_masks_tensors,
            "labels": labels,
            "target_mask": target_mask_tensor,
        }

    def _order_targets(self, targets: np.ndarray) -> np.ndarray:
        ordered = targets.copy()
        if self.shuffle and ordered.size > 0:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(ordered)
        return ordered

    def _neighbor_slices(self, node_id: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        out_start = int(self._forward_index.row_offsets[node_id])
        out_end = int(self._forward_index.row_offsets[node_id + 1])
        out_neighbors = self._forward_index.neighbors[out_start:out_end]

        if self._reverse_index is None:
            return out_neighbors, None

        in_start = int(self._reverse_index.row_offsets[node_id])
        in_end = int(self._reverse_index.row_offsets[node_id + 1])
        in_neighbors = self._reverse_index.neighbors[in_start:in_end]
        return out_neighbors, in_neighbors

    def _sample_layers(
        self, target_nodes: np.ndarray, target_mask: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        layer_nodes: List[np.ndarray] = [target_nodes.astype(np.int64)]
        layer_masks: List[np.ndarray] = [target_mask.astype(bool)]
        neighbor_masks: List[np.ndarray] = []

        current_nodes = target_nodes
        current_mask = target_mask

        for hop_idx, fanout in enumerate(self.fanouts):
            next_nodes, next_mask = self._sample_neighbors(
                current_nodes, current_mask, fanout, hop_idx
            )
            neighbor_masks.append(next_mask)
            layer_nodes.append(next_nodes.reshape(-1))
            layer_masks.append(next_mask.reshape(-1))

            current_nodes = next_nodes.reshape(-1)
            current_mask = next_mask.reshape(-1)

        return layer_nodes, layer_masks, neighbor_masks

    def _sample_neighbors(
        self,
        nodes: np.ndarray,
        nodes_mask: np.ndarray,
        fanout: int,
        hop_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_parents = nodes.shape[0]
        next_nodes = np.full((num_parents, fanout), self.pad_id, dtype=np.int64)
        next_mask = np.zeros((num_parents, fanout), dtype=bool)

        for idx in range(num_parents):
            if not nodes_mask[idx]:
                continue

            node_id = int(nodes[idx])
            out_neighbors, in_neighbors = self._neighbor_slices(node_id)
            num_out = out_neighbors.size
            num_in = 0 if in_neighbors is None else in_neighbors.size
            num_neighbors = num_out + num_in
            if num_neighbors == 0:
                continue

            if num_neighbors >= fanout:
                selected = self._deterministic_choice(
                    out_neighbors,
                    in_neighbors,
                    fanout,
                    node_id,
                    hop_idx,
                )
                next_nodes[idx, :] = selected
                next_mask[idx, :] = True
            else:
                if num_out > 0:
                    next_nodes[idx, :num_out] = out_neighbors
                if num_in > 0 and in_neighbors is not None:
                    next_nodes[idx, num_out:num_neighbors] = in_neighbors
                next_mask[idx, :num_neighbors] = True

        return next_nodes, next_mask

    def _deterministic_choice(
        self,
        out_neighbors: np.ndarray,
        in_neighbors: Optional[np.ndarray],
        fanout: int,
        node_id: int,
        hop_idx: int,
    ) -> np.ndarray:
        total_neighbors = out_neighbors.size + (
            0 if in_neighbors is None else in_neighbors.size
        )
        if total_neighbors == 0:
            return np.empty(0, dtype=np.int64)
        rotation = (self.seed + node_id * 131 + hop_idx * 17) % total_neighbors
        positions = (rotation + np.arange(fanout, dtype=np.int64)) % total_neighbors
        selected = np.empty(fanout, dtype=np.int64)
        out_mask = positions < out_neighbors.size
        if out_mask.any():
            selected[out_mask] = out_neighbors[positions[out_mask]]
        if (~out_mask).any():
            if in_neighbors is None:
                selected[~out_mask] = out_neighbors[
                    positions[~out_mask] - out_neighbors.size
                ]
            else:
                selected[~out_mask] = in_neighbors[
                    positions[~out_mask] - out_neighbors.size
                ]
        return selected

    def _gather_features(
        self, layer_nodes: List[np.ndarray], layer_masks: List[np.ndarray]
    ) -> List[Tensor]:
        features: List[Tensor] = []
        for nodes, mask in zip(layer_nodes, layer_masks):
            node_tensor = torch.from_numpy(nodes)
            mask_tensor = torch.from_numpy(mask.astype(bool)).bool()

            if self.graph_cache is not None:
                feats = self.graph_cache.fetch(node_tensor)
            else:
                feats = self.features[node_tensor]

            mask_float = mask_tensor.view(-1, 1).to(
                device=feats.device, dtype=feats.dtype
            )
            feats = feats * mask_float
            features.append(feats)
        return features


class NeighborSamplingDataProcessor(BaseGraphDataSource):
    """Prepares deterministic neighbor-sampled batches for GraphSAGE."""

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        current_split: str,
        float_dtype: torch.dtype,
        label_dtype: torch.dtype,
        adj_normalization_fn,
        *,
        fanouts: Sequence[int],
        batch_size: int,
        shuffle: bool,
        sampler_seed: int,
        num_workers: int,
        pad_id: int,
        caching_percent: Optional[float] = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            data_dir=data_dir,
            current_split=current_split,
            float_dtype=float_dtype,
            label_dtype=label_dtype,
            adj_normalization_fn=adj_normalization_fn,
        )
        self.fanouts = list(fanouts)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler_seed = sampler_seed
        self.num_workers = validate_num_workers(
            num_workers,
            context=f"{self.__class__.__name__}.num_workers",
        )
        self.pad_id = pad_id
        self.caching_percent = caching_percent
        self.graph_cache = None

    def create_dataloader(self) -> DataLoader:
        features, edge_index, labels, split_masks = self.prepare_graph_components()
        split_key = self.current_split or "train"
        if split_key not in split_masks:
            raise ValueError(f"Split '{split_key}' not available in dataset masks.")
        split_mask = split_masks[split_key]

        # Initialize GraphCache if enabled
        if self.caching_percent is not None and self.graph_cache is None:
            # Determine target caching device
            if cstorch.use_cs():
                cache_device = torch.device("cpu")
            else:
                # Try to get device from cstorch backend, fallback to auto-detect
                cache_device = getattr(cstorch.backend(), "device", None)

                # Ensure compatibility if backend returns a custom device object (e.g. CPUDevice)
                if cache_device is not None and not isinstance(
                    cache_device, (torch.device, str, int)
                ):
                    cache_device = str(cache_device)

                cache_device = cache_device or torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            # Construct a minimal Data object for initialization
            data = Data(x=features, edge_index=edge_index, num_nodes=features.size(0))
            self.graph_cache = GraphCache(
                data, cache_device, percent=self.caching_percent
            )

        dataset = GraphSAGENeighborSamplerDataset(
            features=features,
            edge_index=edge_index,
            labels=labels,
            mask=split_mask,
            fanouts=self.fanouts,
            batch_size=self.batch_size,
            shuffle=self.shuffle and split_key == "train",
            pad_id=self.pad_id,
            seed=self.sampler_seed,
            edge_index_is_undirected=self._requires_materialized_undirected_edges(),
            graph_cache=self.graph_cache,
        )
        if (
            not self._requires_materialized_undirected_edges()
            and self._graph_data_cache is not None
        ):
            self._graph_data_cache.edge_index = None
        del edge_index

        def _build_torch_dataloader() -> DataLoader:
            return DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=(self.num_workers > 0 and torch.cuda.is_available()),
                collate_fn=lambda batch: batch[0],
            )

        return cstorch.utils.data.DataLoader(_build_torch_dataloader)


__all__ = ["GraphSAGENeighborSamplerDataset", "NeighborSamplingDataProcessor"]
