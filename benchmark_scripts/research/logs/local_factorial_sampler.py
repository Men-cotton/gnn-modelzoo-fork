"""Bounded input providers for the local GraphSAGE sampling-factor probe.

FixedCyclicTreeProvider is an exact vectorization of the native deterministic
sampler on canonical CSR rows. FrozenUniformTreeProvider replays the earlier
uniform provider's epoch-zero random stream; its fixed-neighborhood guarantee
requires the same target order, batch partition and sequential batch traversal.
Neither provider caches features or full-epoch trees.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from benchmark_scripts.research.logs.local_resampled_sampler import ResampledTreeProvider


def sample_fixed_rows(index, nodes, nodes_mask, fanout, hop, seed, pad_id=0):
    """Match native cyclic CSR selection, including rotation at degree == fanout."""
    if fanout <= 0:
        raise ValueError("fanout must be positive")
    nodes = np.asarray(nodes)
    nodes_mask = np.asarray(nodes_mask, dtype=bool)
    if nodes.ndim != 1 or nodes_mask.shape != nodes.shape:
        raise ValueError("nodes and nodes_mask must be equally sized vectors")
    result = np.full((nodes.size, fanout), pad_id, dtype=np.int64)
    valid = np.zeros((nodes.size, fanout), dtype=bool)
    rows = np.flatnonzero(nodes_mask)
    if not rows.size:
        return result, valid
    active = nodes[rows].astype(np.int64, copy=False)
    starts = index.row_offsets[active].astype(np.int64, copy=False)
    degrees = index.row_offsets[active + 1].astype(np.int64, copy=False) - starts
    positions = np.broadcast_to(np.arange(fanout), (rows.size, fanout)).copy()
    rotated = degrees >= fanout
    if rotated.any():
        rotation = (int(seed) + active[rotated] * 131 + int(hop) * 17) % degrees[rotated]
        positions[rotated] = ((rotation[:, None] + np.arange(fanout))
                              % degrees[rotated, None])
    row_valid = np.arange(fanout)[None, :] < degrees[:, None]
    selected_rows, columns = np.nonzero(row_valid)
    result[rows[selected_rows], columns] = index.neighbors[
        starts[selected_rows] + positions[selected_rows, columns]]
    valid[rows] = row_valid
    return result, valid


class FixedCyclicTreeProvider(ResampledTreeProvider):
    """Exact native per-root trees under either original or balanced partitions.

    Target ordering remains the caller's responsibility. The seed used for
    cyclic selection is dataset.seed, as in the native implementation; require
    the public seed argument to agree so the logged policy cannot be ambiguous.
    """

    def __init__(self, dataset, *, seed, batch_sizes=None):
        if int(seed) != int(dataset.seed):
            raise ValueError("Fixed cyclic seed must match the native dataset seed")
        super().__init__(dataset, seed=seed, batch_sizes=batch_sizes)

    def tree(self, index):
        dataset = self.dataset
        if index < 0 or index >= len(dataset):
            raise IndexError(f"Batch index {index} outside dataset length {len(dataset)}")
        targets = dataset._ordered_targets[self.batch_offsets[index]:
                                           self.batch_offsets[index + 1]]
        roots = np.full(dataset.batch_size, dataset.pad_id, dtype=np.int64)
        root_mask = np.zeros(dataset.batch_size, dtype=bool)
        roots[:targets.size] = targets
        root_mask[:targets.size] = True
        nodes, masks, neighbors = [roots], [root_mask], []
        for hop, fanout in enumerate(dataset.fanouts):
            next_ids, next_mask = sample_fixed_rows(
                dataset._forward_index, nodes[-1], masks[-1], fanout,
                hop, dataset.seed, dataset.pad_id)
            nodes.append(next_ids.reshape(-1))
            masks.append(next_mask.reshape(-1))
            neighbors.append(next_mask)
        return nodes, masks, neighbors


class FrozenUniformTreeProvider(ResampledTreeProvider):
    """Replay uniform epoch-zero draws for an unchanged order and partition.

    This is the frozen-uniform control, not a node-keyed sampling cache. A
    different target order, partition, or traversal would associate its random
    draws with different nodes. The runner must keep these fixed for this arm.
    """

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        super().set_epoch(0)
        self.replay_epoch = int(epoch)


def assert_tree_equal(left, right):
    for left_group, right_group in zip(left, right):
        assert len(left_group) == len(right_group)
        for actual, expected in zip(left_group, right_group):
            np.testing.assert_array_equal(actual, expected)


def self_test(modelzoo_root):
    """Check exact native trees and payloads without loading graph datasets."""
    sys.path.insert(0, str(Path(modelzoo_root) / "src"))
    from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
        GraphSAGENeighborSamplerDataset, NeighborSliceIndex,
    )

    adjacency = [[], [2], [1, 3, 4], [1, 2, 4, 5],
                 [x for x in range(12) if x != 4], [1, 2], [0, 2, 3, 5],
                 [1, 2, 3], [2], [0, 1, 2, 3, 4, 5], [3, 7], [4]]
    offsets = np.concatenate(([0], np.cumsum([len(row) for row in adjacency])))
    index = NeighborSliceIndex(offsets.astype(np.int32),
                               np.array(sum(adjacency, []), dtype=np.int32))
    dataset = GraphSAGENeighborSamplerDataset(
        features=torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
        labels=torch.arange(12), mask=torch.arange(12) < 9,
        fanouts=[3, 2], batch_size=4, shuffle=True, pad_id=0,
        seed=42, edge_index_is_undirected=True, neighbor_indexes=(index, None))
    original = dataset._ordered_targets.copy()
    before_rng = torch.get_rng_state().clone()
    # A degree equal to fanout must retain the native rotated ordering.
    ids, mask = sample_fixed_rows(index, np.array([2]), [True], 3, 0, 42)
    np.testing.assert_array_equal(ids, [[3, 4, 1]])
    assert mask.all()
    # Explicitly cover isolated and masked parents, low degree and rotated rows.
    parents = np.array([0, 1, 2, 3, 4, 4])
    active = np.array([True, True, True, True, True, False])
    for fanout in (2, 3, 4):
        for hop in (0, 1, 2):
            expected = dataset._sample_neighbors(parents, active, fanout, hop)
            actual = sample_fixed_rows(index, parents, active, fanout, hop, 42)
            for left, right in zip(actual, expected):
                np.testing.assert_array_equal(left, right)
    per_root_trees = {}
    for sizes in (None, [3, 3, 3]):
        dataset._ordered_targets = original.copy()
        provider = FixedCyclicTreeProvider(dataset, seed=42, batch_sizes=sizes)
        for epoch in (0, 1):
            dataset._ordered_targets = (original.copy() if epoch == 0 else
                np.random.default_rng(10043).permutation(original))
            provider.set_epoch(epoch)
            seen = []
            for batch in range(len(dataset)):
                actual = provider.tree(batch)
                nodes, masks, neighbors = actual
                expected = dataset._sample_layers(nodes[0], masks[0])
                assert_tree_equal(actual, expected)
                seen.extend(nodes[0][masks[0]].tolist())
                # Prove repartitioning and reshuffling preserve every root's
                # complete sampled tree, including all downstream padding.
                for row in np.flatnonzero(masks[0]):
                    node = int(nodes[0][row])
                    signature = [value.reshape(dataset.batch_size, -1)[row].copy()
                                 for value in nodes + masks]
                    if node in per_root_trees:
                        for left, right in zip(signature, per_root_trees[node]):
                            np.testing.assert_array_equal(left, right)
                    else:
                        per_root_trees[node] = signature
                payload = provider.payload(batch, actual)
                if sizes is None:
                    native_payload = dataset[batch]
                    for key, values in payload.items():
                        expected_values = native_payload[key]
                        if torch.is_tensor(values):
                            values, expected_values = [values], [expected_values]
                        for left, right in zip(values, expected_values):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                else:
                    assert int(payload["target_mask"].sum()) == 3
                    for layer_ids, layer_mask, features in zip(
                            nodes, masks, payload["node_features"]):
                        expected_features = dataset.features[torch.from_numpy(layer_ids)].clone()
                        expected_features[~torch.from_numpy(layer_mask)] = 0
                        torch.testing.assert_close(features.reshape(-1, 3),
                                                   expected_features, rtol=0, atol=0)
            np.testing.assert_array_equal(seen, dataset._ordered_targets)
    dataset._ordered_targets = original.copy()
    fresh = ResampledTreeProvider(dataset, seed=42)
    frozen = FrozenUniformTreeProvider(dataset, seed=42)
    first_epoch = []
    for batch in range(len(dataset)):
        original_tree = fresh.tree(batch)
        assert_tree_equal(original_tree, frozen.tree(batch))
        first_epoch.append(original_tree)
    fresh.set_epoch(1)
    frozen.set_epoch(1)
    changed = False
    for batch, expected in enumerate(first_epoch):
        assert_tree_equal(frozen.tree(batch), expected)
        actual = fresh.tree(batch)
        changed |= any(not np.array_equal(left, right)
                       for left, right in zip(actual[0], expected[0]))
    assert changed, "Fresh epoch-one sampling unexpectedly repeated epoch zero"
    frozen.set_epoch(7)
    for batch, expected in enumerate(first_epoch):
        assert_tree_equal(frozen.tree(batch), expected)
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    print("PASS: native cyclic trees and payloads, degree == fanout rotation, "
          "padding, balanced and shuffled per-root invariance, frozen uniform "
          "epoch-zero equality and replay, fresh epoch-one changes, Torch RNG isolation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.add_argument("--modelzoo-root", type=Path,
                        default=Path(__file__).resolve().parents[3])
    arguments = parser.parse_args()
    self_test(arguments.modelzoo_root)
