"""Local, bounded-memory neighbor resampling for the tail-loss mechanism probe.

The caller owns the target permutation. This provider replaces only the native
sampler's fixed cyclic neighbor window with a fresh, uniform subset without
replacement for each parent occurrence. Features, labels, padding and feature
gathering retain the original Model Zoo contract. No graph or epoch trees are
cached. This is a local GPU experiment, not a production CS-3 loader patch.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations, product
from pathlib import Path
import sys

import numpy as np
import torch


def uniform_positions(degrees, fanout, rng):
    """Vectorized Floyd sampling: one uniform size-k subset per CSR row.

    All degrees must exceed k. At iteration j, choose t uniformly from [0, j];
    insert j if t was already selected, otherwise insert t. The result is an
    unordered uniform subset. Each row has a separate draw at every iteration.
    Scratch storage scales with sampled parent occurrences times fanout, never
    with their degrees or with all graph edges.
    """
    degrees = np.asarray(degrees, dtype=np.int64)
    if fanout <= 0 or np.any(degrees <= fanout):
        raise ValueError("Floyd sampling requires positive fanout < every degree")
    selected = np.empty((degrees.size, fanout), dtype=np.int64)
    for column in range(fanout):
        upper = degrees - fanout + column
        draw = rng.integers(0, upper + 1, dtype=np.int64)
        duplicate = np.any(selected[:, :column] == draw[:, None], axis=1)
        selected[:, column] = np.where(duplicate, upper, draw)
    return selected


def sample_neighbor_rows(index, nodes, nodes_mask, fanout, rng, pad_id=0):
    """Sample a canonical, duplicate-free CSR; return original int64/bool shapes."""
    if fanout <= 0:
        raise ValueError("fanout must be positive")
    nodes = np.asarray(nodes)
    nodes_mask = np.asarray(nodes_mask, dtype=bool)
    if nodes.ndim != 1 or nodes_mask.shape != nodes.shape:
        raise ValueError("nodes and nodes_mask must be equally sized vectors")
    result = np.full((nodes.size, fanout), pad_id, dtype=np.int64)
    valid = np.zeros((nodes.size, fanout), dtype=bool)
    rows = np.flatnonzero(nodes_mask)
    if rows.size == 0:
        return result, valid
    active_nodes = nodes[rows]
    starts = index.row_offsets[active_nodes].astype(np.int64, copy=False)
    degrees = (index.row_offsets[active_nodes + 1].astype(np.int64, copy=False)
               - starts)
    positions = np.broadcast_to(np.arange(fanout), (rows.size, fanout)).copy()
    sampled = degrees > fanout
    if sampled.any():
        positions[sampled] = uniform_positions(degrees[sampled], fanout, rng)
    row_valid = np.arange(fanout)[None, :] < degrees[:, None]
    # Gather valid slots only, so degree-zero and padded parents never index
    # outside the CSR or silently reuse the padding node's real neighbors.
    row_indices, columns = np.nonzero(row_valid)
    result[rows[row_indices], columns] = index.neighbors[
        starts[row_indices] + positions[row_indices, columns]]
    valid[rows] = row_valid
    return result, valid


class ResampledTreeProvider:
    """TreeCache-compatible on-demand trees with an independent epoch RNG.

    The dataset must provide canonical unique CSR rows and no reverse index,
    as produced by ``local_ogb_graph.load_local_ogb_graph``. Calling ``tree``
    advances the RNG. Pass the returned tree to ``payload`` to gather the same
    draw. ``set_epoch`` deterministically resets the RNG for replay; repeated
    occurrences within an epoch are independently sampled. Optional batch_sizes
    partitions the shuffled targets while retaining the native number of batches
    and padded batch shape; every training target occurs exactly once per epoch.
    """

    def __init__(self, dataset, *, seed, batch_sizes=None):
        if dataset._reverse_index is not None:
            raise ValueError("Resampling requires a canonical undirected CSR")
        if any(fanout <= 0 for fanout in dataset.fanouts):
            raise ValueError("All fanouts must be positive")
        self.dataset = dataset
        self.original_order = dataset._ordered_targets.copy()
        self.seed = int(seed)
        self.bytes = 0
        if batch_sizes is None:
            batch_sizes = [min(dataset.batch_size,
                               len(self.original_order) - index * dataset.batch_size)
                           for index in range(len(dataset))]
        raw_sizes = np.asarray(batch_sizes)
        if (raw_sizes.ndim != 1 or not np.issubdtype(raw_sizes.dtype, np.integer)
                or raw_sizes.size != len(dataset)):
            raise ValueError("batch_sizes must be one integer count per dataset batch")
        self.batch_sizes = raw_sizes.astype(np.int64, copy=True)
        if (np.any(self.batch_sizes <= 0)
                or np.any(self.batch_sizes > dataset.batch_size)
                or int(self.batch_sizes.sum()) != self.original_order.size):
            raise ValueError("batch_sizes must cover every target once and fit the padded batch")
        self.batch_offsets = np.concatenate(([0], np.cumsum(self.batch_sizes)))
        self.target_counts = self.batch_sizes.tolist()
        self.set_epoch(0)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)
        self.rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, self.epoch, 0x4E454947]))

    def tree(self, index):
        dataset = self.dataset
        if index < 0 or index >= len(dataset):
            raise IndexError(f"Batch index {index} outside dataset length {len(dataset)}")
        targets = dataset._ordered_targets[self.batch_offsets[index]:
                                           self.batch_offsets[index + 1]]
        root_ids = np.full(dataset.batch_size, dataset.pad_id, dtype=np.int64)
        root_mask = np.zeros(dataset.batch_size, dtype=bool)
        root_ids[:targets.size] = targets
        root_mask[:targets.size] = True
        nodes, masks, neighbors = [root_ids], [root_mask], []
        for fanout in dataset.fanouts:
            next_ids, next_mask = sample_neighbor_rows(
                dataset._forward_index, nodes[-1], masks[-1], fanout,
                self.rng, dataset.pad_id)
            nodes.append(next_ids.reshape(-1))
            masks.append(next_mask.reshape(-1))
            neighbors.append(next_mask)
        return nodes, masks, neighbors

    def payload(self, index, tree=None):
        dataset = self.dataset
        nodes, masks, neighbors = self.tree(index) if tree is None else tree
        features = dataset._gather_features(nodes, masks)
        labels = dataset.labels[torch.from_numpy(nodes[0])].clone().reshape(-1)
        target_mask = torch.from_numpy(masks[0]).bool()
        labels[~target_mask] = 0
        return {
            "node_features": [x.view(dataset.batch_size, -1, x.shape[-1])
                              for x in features],
            "node_masks": [torch.from_numpy(x).view(dataset.batch_size, -1)
                           for x in masks],
            "neighbor_masks": [torch.from_numpy(x).view(dataset.batch_size, -1,
                                                        x.shape[-1])
                               for x in neighbors],
            "labels": labels,
            "target_mask": target_mask,
        }


def self_test(modelzoo_root):
    """Check sampling distribution, original payload contracts and repeatability."""
    sys.path.insert(0, str(Path(modelzoo_root) / "src"))
    from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
        GraphSAGENeighborSamplerDataset, NeighborSliceIndex,
    )

    # Exhaust every possible random draw of Floyd(6, 3), an exact reference
    # check that covers all 20 subsets equally (six paths each).
    draws = np.array(list(product(range(4), range(5), range(6))))

    class ExhaustiveRng:
        column = 0

        def integers(self, low, high, dtype):
            assert low == 0 and np.all(high == 4 + self.column)
            values = draws[:, self.column].astype(dtype)
            self.column += 1
            return values

    selected = uniform_positions(np.full(draws.shape[0], 6), 3, ExhaustiveRng())
    subsets = Counter(tuple(sorted(row)) for row in selected)
    assert subsets == Counter({subset: 6 for subset in combinations(range(6), 3)})
    # Mixed degree and masked-parent contract, including a valid isolated root.
    index = NeighborSliceIndex(np.array([0, 0, 1, 4, 10], dtype=np.int32),
                               np.array([2, 0, 1, 3, 0, 1, 2, 3, 4, 5], dtype=np.int32))
    nodes = np.array([0, 1, 2, 3, 3])
    output, valid = sample_neighbor_rows(index, nodes, [1, 1, 1, 1, 0], 3,
                                         np.random.default_rng(42))
    np.testing.assert_array_equal(valid.sum(axis=1), [0, 1, 3, 3, 0])
    for row in range(4):
        values = output[row, valid[row]]
        assert len(set(values)) == values.size
        start, end = index.row_offsets[row:row + 2]
        assert set(values).issubset(index.neighbors[start:end])
    assert np.all(output[~valid] == 0)

    def dataset_for(edges, fanouts, num_nodes=7):
        return GraphSAGENeighborSamplerDataset(
            features=torch.arange(num_nodes * 3, dtype=torch.float32).reshape(num_nodes, 3),
            edge_index=torch.tensor(edges, dtype=torch.int64).t().contiguous(),
            labels=torch.arange(num_nodes), mask=torch.arange(num_nodes) < 5,
            fanouts=fanouts, batch_size=4, shuffle=True, pad_id=0,
            seed=42, edge_index_is_undirected=True)

    # All degrees below fanout: the complete payload must be bit-exact with
    # the native implementation, including tail targets and every padding mask.
    sparse_dataset = dataset_for([(0, 1), (1, 0), (1, 2), (2, 1)], [4, 3])
    sparse_provider = ResampledTreeProvider(sparse_dataset, seed=42)
    for batch in range(len(sparse_dataset)):
        expected, actual = sparse_dataset[batch], sparse_provider.payload(batch)
        for key, values in actual.items():
            expected_values = expected[key]
            if torch.is_tensor(values):
                values, expected_values = [values], [expected_values]
            for left, right in zip(values, expected_values):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    dense_dataset = dataset_for([(u, v) for u in range(7) for v in range(7) if u != v],
                                [3, 2])
    provider = ResampledTreeProvider(dense_dataset, seed=42)
    first, second = provider.tree(0), provider.tree(0)
    assert any(not np.array_equal(a, b) for a, b in zip(first[0], second[0]))
    provider.set_epoch(0)
    replay = provider.tree(0)
    for left_group, right_group in zip(first, replay):
        for left, right in zip(left_group, right_group):
            np.testing.assert_array_equal(left, right)
    provider.set_epoch(1)
    changed = provider.tree(0)
    assert any(not np.array_equal(a, b) for a, b in zip(first[0], changed[0]))
    for depth, (ids, mask) in enumerate(zip(changed[0], changed[1])):
        assert ids.dtype == np.int64 and mask.dtype == np.bool_
        if depth:
            np.testing.assert_array_equal(mask, changed[2][depth - 1].reshape(-1))
    payload = provider.payload(0, changed)
    for ids, mask, features in zip(changed[0], changed[1], payload["node_features"]):
        expected = dense_dataset.features[torch.from_numpy(ids)].clone()
        expected[~torch.from_numpy(mask)] = 0
        torch.testing.assert_close(features.reshape(-1, 3), expected, rtol=0, atol=0)
    balanced = ResampledTreeProvider(dense_dataset, seed=42, batch_sizes=[3, 2])
    seen = []
    for batch, count in enumerate(balanced.target_counts):
        tree = balanced.tree(batch)
        assert int(tree[1][0].sum()) == count and tree[0][0].size == 4
        seen.extend(tree[0][0][tree[1][0]].tolist())
        assert int(balanced.payload(batch, tree)["target_mask"].sum()) == count
    np.testing.assert_array_equal(seen, dense_dataset._ordered_targets)
    for bad_counts in ([4, 4], [5, 0], [4], [3.0, 2.0]):
        try:
            ResampledTreeProvider(dense_dataset, seed=42, batch_sizes=bad_counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid partition accepted: {bad_counts}")
    print("PASS: exact 20-subset distribution, mixed degrees, padding, original payload, "
          "fresh draws, epoch replay, original feature gathering, and balanced partitions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.add_argument("--modelzoo-root", type=Path,
                        default=Path(__file__).resolve().parents[3])
    arguments = parser.parse_args()
    self_test(arguments.modelzoo_root)
