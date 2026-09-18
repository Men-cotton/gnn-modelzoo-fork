"""Check schedule accounting against emitted payloads and the SDK stream pipe."""

import contextlib
import io
import itertools
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cerebras.modelzoo.models.gnn import fixed_shape_gpu as runner
from cerebras.modelzoo.models.gnn.data_processing.processor import GNNDataProcessor
from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
    CachedStaticGraphSAGEDataset,
    GraphSAGENeighborSamplerDataset,
    batch_accounting_contract,
)
from cerebras.modelzoo.models.gnn.data_processing.sources.base import (
    BaseGraphDataSource,
)
from cerebras.modelzoo.models.gnn.tests.test_fixed_shape_gpu import (
    tiny_config,
    tiny_graph,
)
from cerebras.pytorch.utils.data.dataloader import RestartableDataLoader
from cerebras.pytorch.utils.data.streamer.data_pipe import MegaBatcher, Repeater
from cerebras.pytorch.utils.data.utils import infer_batch_size


def dataset(*, shuffle=False, drop_last=False):
    return GraphSAGENeighborSamplerDataset(
        features=torch.ones(5, 2),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        labels=torch.tensor([0, -100, 1, 0, -100]),
        mask=torch.ones(5, dtype=torch.bool),
        fanouts=[2],
        batch_size=2,
        shuffle=shuffle,
        pad_id=0,
        seed=42,
        edge_index_is_undirected=True,
        drop_last=drop_last,
    )


def contract(source, *, drop_last=False, static_cache=0):
    return batch_accounting_contract(
        source,
        dataset_name="fixture",
        split="train",
        drop_last=drop_last,
        static_batch_cache_size=static_cache,
    )


class BatchAccountingTests(unittest.TestCase):
    def test_accounting_matches_emitted_masks_with_shuffle_drop_and_replay(self):
        for shuffle, drop_last, static_cache in itertools.product(
            (False, True), (False, True), (0, 1, 2)
        ):
            with self.subTest(shuffle=shuffle, drop_last=drop_last, cache=static_cache):
                source = dataset(shuffle=shuffle, drop_last=drop_last)
                actual = contract(
                    source, drop_last=drop_last, static_cache=static_cache
                )
                emitted = (
                    CachedStaticGraphSAGEDataset(source, static_cache)
                    if static_cache
                    else source
                )
                payloads = [emitted[index] for index in range(len(emitted))]
                self.assertEqual(
                    actual["seed_nodes_by_batch"],
                    [int(batch["target_mask"].sum()) for batch in payloads],
                )
                self.assertEqual(
                    actual["supervised_targets_by_batch"],
                    [
                        int((batch["target_mask"] & (batch["labels"] != -100)).sum())
                        for batch in payloads
                    ],
                )
                self.assertEqual(actual["source_seed_nodes"], 5)
                self.assertEqual(
                    actual["seed_nodes_per_epoch"], sum(actual["seed_nodes_by_batch"])
                )

    def test_digest_ignores_label_storage_dtype_but_detects_order_and_labels(self):
        source = dataset()
        digest = contract(source)["ordered_targets_and_labels_sha256"]
        source.labels = source.labels.to(torch.int32)
        self.assertEqual(contract(source)["ordered_targets_and_labels_sha256"], digest)
        source.labels[0] = 1
        self.assertNotEqual(
            contract(source)["ordered_targets_and_labels_sha256"], digest
        )
        self.assertNotEqual(
            contract(dataset(shuffle=True))["ordered_targets_and_labels_sha256"], digest
        )

    def test_sdk_batch_size_and_microblock_stream_preserve_tail_counts(self):
        source = dataset()
        payloads = [source[index] for index in range(len(source))]
        self.assertEqual([infer_batch_size(batch) for batch in payloads], [2, 2, 2])
        # The SDK remote path repeats complete logical batches, then splits
        # them for transfer. Splitting must preserve the padded tail position.
        named = [
            {key: batch[key].numpy() for key in ("labels", "target_mask")}
            for batch in payloads
        ]
        blocks = iter(MegaBatcher(Repeater(named), [1, 1]))
        seed_counts = []
        supervised_counts = []
        for _ in range(6):
            one, two = next(blocks), next(blocks)
            mask = np.concatenate([one["target_mask"], two["target_mask"]])
            labels = np.concatenate([one["labels"], two["labels"]])
            seed_counts.append(int(mask.sum()))
            supervised_counts.append(int((mask & (labels != -100)).sum()))
        blocks.close()
        actual = contract(source)
        self.assertEqual(seed_counts, actual["seed_nodes_by_batch"] * 2)
        self.assertEqual(supervised_counts, actual["supervised_targets_by_batch"] * 2)

    def test_opt_in_is_forwarded_and_reports_real_loader_contract(self):
        loader_cfg = tiny_config()["trainer"]["fit"]["train_dataloader"]
        loader_cfg["cache_fraction"] = None
        loader_cfg["measure_batch_accounting"] = True
        for native in (False, True):
            with (
                self.subTest(native=native),
                patch.object(BaseGraphDataSource, "load_graph", side_effect=tiny_graph),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                loader = (
                    runner.make_loader(
                        loader_cfg, num_layers=2, float_dtype=torch.float32
                    )
                    if native
                    else GNNDataProcessor(loader_cfg).create_dataloader()
                )
                records = [
                    json.loads(line.split("GNN_INPUT_CONTRACT ", 1)[1])
                    for line in stdout.getvalue().splitlines()
                    if line.startswith("GNN_INPUT_CONTRACT ")
                ]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["seed_nodes_by_batch"], [2, 1])
                self.assertEqual(records[0]["supervised_targets_by_batch"], [2, 1])
                self.assertFalse(isinstance(loader, RestartableDataLoader))
                self.assertEqual(records[0]["batch_index_origin"], 0)
                self.assertEqual(records[0]["traversal_scope"], "single_data_executor")
                self.assertEqual(loader.gnn_input_contract, records[0])
        loader_cfg["measure_batch_accounting"] = False
        with (
            patch.object(BaseGraphDataSource, "load_graph", side_effect=tiny_graph),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            GNNDataProcessor(loader_cfg).create_dataloader()
        self.assertNotIn("GNN_INPUT_CONTRACT", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
