from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator

from .common import (
    BaseGraphDataSource,
    EdgeIndexAdjacency,
    sparse_scipy_to_edge_tensors,
)

logger = logging.getLogger(__name__)


class _SingleGraphDataset(Dataset):
    """Dataset wrapper for full-graph training (batch size 1)."""

    def __init__(
        self,
        features: Tensor,
        adjacency: Union[EdgeIndexAdjacency, Tensor],
        labels: Tensor,
        mask: Tensor,
    ):
        super().__init__()
        self._data = ((features, adjacency), (labels, mask))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        if idx != 0:
            raise IndexError("Dataset contains only one item indexed at 0.")
        return self._data


def _collate_full_graph(batch):
    if len(batch) != 1:
        raise ValueError(f"Full graph collate expects len=1; got {len(batch)}.")
    return batch[0]


class FullGraphDataProcessor(BaseGraphDataSource):
    """Handles full-graph loading used by the GCN baseline."""

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        current_split: str,
        float_dtype: torch.dtype,
        label_dtype: torch.dtype,
        adj_normalization_fn,
        *,
        drop_last: bool,
        num_workers: int,
    ):
        super().__init__(
            dataset_name=dataset_name,
            data_dir=data_dir,
            current_split=current_split,
            float_dtype=float_dtype,
            label_dtype=label_dtype,
            adj_normalization_fn=adj_normalization_fn,
        )
        self.drop_last = drop_last
        self.num_workers = num_workers

    def create_dataloader(self) -> DataLoader:
        features, adjacency, labels, mask = self.load_full_graph()
        dataset = _SingleGraphDataset(features, adjacency, labels, mask)

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=(self.num_workers > 0 and torch.cuda.is_available()),
            collate_fn=_collate_full_graph,
        )

    def create_fake_dataloader(
        self,
        *,
        num_nodes: int,
        n_feat: int,
        n_class: int,
        seed: Optional[int] = None,
    ) -> SampleGenerator:
        logger.info("Using fake data generator (seed: %s).", seed)
        if seed is not None:
            torch.manual_seed(seed)

        fake_features = torch.rand(num_nodes, n_feat, dtype=self.float_dtype)

        num_edges = num_nodes * 5
        row = torch.randint(0, num_nodes, (num_edges,))
        col = torch.randint(0, num_nodes, (num_edges,))
        adj_sp = sp.coo_matrix(
            (np.ones(num_edges, dtype=np.float32), (row.numpy(), col.numpy())),
            shape=(num_nodes, num_nodes),
        )
        adj_sp.eliminate_zeros()
        adj_sp = adj_sp + adj_sp.T
        adj_sp.data = np.clip(adj_sp.data, 0, 1)
        adj_sp = adj_sp.tocoo()

        if self.adj_normalization_fn:
            adj_sp = self.adj_normalization_fn(adj_sp, True)

        fake_edge_index, fake_edge_weight = sparse_scipy_to_edge_tensors(
            adj_sp, dtype=self.float_dtype
        )
        fake_labels = torch.randint(0, n_class, (num_nodes,), dtype=self.label_dtype)
        fake_mask = (torch.rand(num_nodes) > 0.5).bool()

        get_streaming_batch_size(1)
        sample_count = 10
        logger.info(
            "Fake data generator: %d nodes, %d features, %d classes.",
            num_nodes,
            n_feat,
            n_class,
        )
        if cstorch.use_cs():
            fake_adj = torch.zeros(
                (num_nodes, num_nodes), dtype=self.float_dtype
            )
            if fake_edge_index.numel() > 0:
                fake_idx_long = fake_edge_index.to(dtype=torch.long)
                fake_adj.index_put_(
                    (fake_idx_long[0], fake_idx_long[1]),
                    fake_edge_weight.to(dtype=self.float_dtype),
                    accumulate=True,
                )
            fake_adj = fake_adj.unsqueeze(0)
            fake_features = fake_features.unsqueeze(0)
            fake_labels = fake_labels.unsqueeze(0)
            fake_mask = fake_mask.unsqueeze(0)
        else:
            fake_adj = EdgeIndexAdjacency(
                edge_index=fake_edge_index, edge_weight=fake_edge_weight
            )

        return SampleGenerator(
            data=(
                (
                    fake_features,
                    fake_adj,
                ),
                (fake_labels, fake_mask),
            ),
            sample_count=sample_count,
        )


__all__ = ["FullGraphDataProcessor"]
