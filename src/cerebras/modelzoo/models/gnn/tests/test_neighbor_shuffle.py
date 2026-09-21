"""Target shuffling must reach later input passes and persistent workers."""

import contextlib
import io
import itertools
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cerebras.modelzoo.models.gnn.data_processing.samplers import neighbor_tree
from cerebras.modelzoo.models.gnn.data_processing.worker_diagnostics_config import (
    WorkerDiagnosticsConfig,
)
from cerebras.modelzoo.models.gnn.tools.measure_window import validate_input_contract
from cerebras.pytorch.utils.data.streamer.data_pipe import Repeater


def graph():
    return (
        torch.arange(46, dtype=torch.float32).view(23, 2),
        torch.tensor([list(range(23)), list(range(1, 23)) + [0]]),
        torch.arange(23),
        {key: torch.ones(23, dtype=torch.bool) for key in ("train", "valid")},
    )


def targets(batches):
    return [batch["labels"][batch["target_mask"]].tolist() for batch in batches]


class NeighborShuffleTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def loader(
        self,
        *,
        workers=0,
        persistent=False,
        shuffle=True,
        split="train",
        drop_last=False,
        static=0,
        diagnostic=False,
        seed=42,
        components=None
    ):
        processor = neighbor_tree.NeighborSamplingDataProcessor(
            dataset_name="fixture",
            data_dir=self.temp.name,
            current_split=split,
            float_dtype=torch.float32,
            label_dtype=torch.long,
            adj_normalization_fn=None,
            fanouts=[2],
            batch_size=8,
            shuffle=shuffle,
            sampler_seed=seed,
            num_workers=workers,
            pad_id=0,
            persistent_workers=persistent,
            drop_last=drop_last,
            static_batch_cache_size=static,
            pin_memory=False,
            measure_batch_accounting=True,
            worker_diagnostics=(
                WorkerDiagnosticsConfig(
                    enabled=True,
                    output_dir=self.temp.name,
                    max_batches=20,
                    max_snapshots=0,
                )
                if diagnostic
                else None
            ),
        )
        with (
            patch.object(
                processor,
                "prepare_graph_components",
                return_value=components or graph(),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            loader = processor.create_torch_dataloader()
        self.addCleanup(self.stop_workers, loader)
        return loader

    @staticmethod
    def stop_workers(loader):
        if getattr(loader, "_iterator", None) is not None:
            loader._iterator._shutdown_workers()

    def check_passes(self, loader):
        passes = []
        pids = None
        for epoch in range(3):
            batches = list(loader)
            actual = targets(batches)
            self.assertEqual([len(batch) for batch in actual], [8, 8, 7])
            order = list(itertools.chain.from_iterable(actual))
            self.assertEqual(sorted(order), list(range(23)))
            expected = np.arange(23)
            np.random.default_rng(42 + epoch).shuffle(expected)
            self.assertEqual(order, expected.tolist())
            for batch in batches:
                self.assertEqual(tuple(batch["node_features"][0].shape), (8, 1, 2))
                self.assertEqual(tuple(batch["target_mask"].shape), (8,))
                # Root features and labels must follow the same permutation.
                mask = batch["target_mask"]
                self.assertTrue(
                    torch.equal(
                        batch["node_features"][0][mask, 0, 0], batch["labels"][mask] * 2
                    )
                )
            if loader.persistent_workers:
                current = [p.pid for p in loader._iterator._workers]
                if pids is not None:
                    self.assertEqual(current, pids)
                pids = current
            passes.append(actual)
        self.assertNotEqual(set(passes[0][-1]), set(passes[1][-1]))
        self.assertNotEqual(set(passes[1][-1]), set(passes[2][-1]))
        return passes

    def test_worker_zero_reshuffles_targets_and_tail(self):
        self.check_passes(self.loader())

    def test_worker_copies_reshuffle_with_and_without_persistence(self):
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                self.check_passes(self.loader(workers=2, persistent=persistent))

    def test_seed_reproducibility_is_independent_of_worker_count(self):
        expected = self.check_passes(self.loader())
        self.assertEqual(
            expected, self.check_passes(self.loader(workers=2, persistent=True))
        )
        different = targets(list(self.loader(seed=43)))
        self.assertNotEqual(expected[0], different)

    def test_sdk_repeater_preserves_progress_between_consumer_chunks(self):
        loader = self.loader(workers=2, persistent=True)
        repeated = iter(Repeater(loader))
        self.addCleanup(repeated.close)
        batches = []
        # Consumer pauses need not coincide with the three-batch input pass.
        for steps in (2, 2, 5):
            batches.extend(itertools.islice(repeated, steps))
        expected_loader = self.loader()
        expected = [targets(list(expected_loader)) for _ in range(3)]
        self.assertEqual(
            targets(batches), list(itertools.chain.from_iterable(expected))
        )

    def test_disabled_shuffle_validation_and_static_replay_remain_fixed(self):
        for options in (dict(shuffle=False), dict(split="valid"), dict(static=1)):
            with self.subTest(options=options):
                loader = self.loader(workers=2, persistent=True, **options)
                first = targets(list(loader))
                self.assertEqual(first, targets(list(loader)))
                self.assertEqual(first, targets(list(loader)))
                self.assertEqual(loader.gnn_input_contract["target_order"], "fixed")

    def test_drop_last_changes_omitted_targets_between_passes(self):
        loader = self.loader(drop_last=True, workers=2, persistent=True)
        epochs = [targets(list(loader)) for _ in range(3)]
        for batches in epochs:
            self.assertEqual([len(batch) for batch in batches], [8, 8])
            self.assertEqual(len(set(itertools.chain.from_iterable(batches))), 16)
        self.assertNotEqual(
            set(itertools.chain.from_iterable(epochs[0])),
            set(itertools.chain.from_iterable(epochs[1])),
        )

    def test_diagnostics_accept_epoch_tagged_indices(self):
        import json
        from pathlib import Path

        self.check_passes(self.loader(workers=2, persistent=True, diagnostic=True))
        events = [
            json.loads(line)
            for path in Path(self.temp.name).rglob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        batches = [r for r in events if r["event"] == "batch_generated"]
        self.assertEqual({r["input_epoch"] for r in batches}, {0, 1, 2})
        self.assertEqual({r["batch_index"] for r in batches}, {0, 1, 2})

    def test_contract_scopes_first_pass_order_and_repeated_counts(self):
        loader = self.loader()
        contract = validate_input_contract(loader.gnn_input_contract)
        self.assertEqual(contract["version"], 2)
        self.assertEqual(contract["target_order"], "reshuffle_each_epoch")
        self.assertEqual(contract["ordered_targets_and_labels_epoch"], 0)
        self.assertEqual(contract["supervised_targets_by_batch_scope"], "all_epochs")
        for _ in range(3):
            batches = list(loader)
            self.assertEqual(
                [int(b["target_mask"].sum()) for b in batches],
                contract["seed_nodes_by_batch"],
            )

        components = graph()
        components[2][0] = -100
        contract = self.loader(components=components).gnn_input_contract
        self.assertEqual(contract["supervised_targets_by_batch_scope"], "first_epoch")
        validate_input_contract(contract)
        # Without reshuffling, ignored labels retain their positions.
        validate_input_contract(
            self.loader(components=components, shuffle=False).gnn_input_contract
        )


if __name__ == "__main__":
    unittest.main()
