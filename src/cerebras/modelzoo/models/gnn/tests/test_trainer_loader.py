"""Exercise the Trainer-owned SDK wrapper and its serialized remote factory."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cerebras.appliance.appliance_client import (
    fw_user_deserialize,
    fw_user_serialize,
)
from cerebras.modelzoo.config import create_config_class
from cerebras.modelzoo.models.gnn.data_processing import processor as facade
from cerebras.modelzoo.models.gnn.data_processing.samplers import neighbor_tree
from cerebras.modelzoo.trainer.utils import create_dataloader_from_config


def graph():
    return (
        torch.arange(24, dtype=torch.float32).view(12, 2),
        torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]),
        torch.arange(12) % 2,
        {"train": torch.ones(12, dtype=torch.bool)},
    )


def remote_factory(loader):
    def roundtrip(value):
        return fw_user_deserialize(
            fw_user_serialize(value, from_usr=True, recurse=True), from_usr=True
        )

    builder = roundtrip(loader.input_fn)
    args, kwargs = roundtrip(loader.input_fn_params)
    return builder(*args, **kwargs)


class TrainerLoaderTests(unittest.TestCase):
    def test_remote_workers_survive_local_inspection(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        for workers, cache, static, diagnostic in (
            (0, None, 0, False),
            (2, None, 0, False),
            (2, 1.0, 0, False),
            (2, None, 1, False),
            (2, None, 0, True),
        ):
            with self.subTest(
                workers=workers, cache=cache, static=static, diagnostic=diagnostic
            ):
                with (
                    tempfile.TemporaryDirectory() as directory,
                    patch.object(facade.cstorch, "use_cs", return_value=True),
                    patch.object(facade.dist, "get_ordinal", return_value=0),
                    patch.object(
                        facade.cstorch.amp,
                        "get_floating_point_dtype",
                        return_value=torch.float32,
                    ),
                    patch.object(
                        neighbor_tree.NeighborSamplingDataProcessor,
                        "prepare_graph_components",
                        side_effect=graph,
                    ),
                ):
                    config = create_config_class(facade.GNNDataProcessor)(
                        config=dict(
                            data_processor="GNNDataProcessor",
                            dataset_name="ogbn-arxiv",
                            data_dir=directory,
                            sampling_mode="neighbor",
                            fanouts=[2, 2],
                            batch_size=2,
                            split="train",
                            num_workers=workers,
                            persistent_workers=True,
                            prefetch_factor=3,
                            cache_fraction=cache,
                            static_batch_cache_size=static,
                            worker_diagnostics=dict(
                                enabled=diagnostic,
                                output_dir=directory,
                                max_batches=16,
                                max_snapshots=0,
                            ),
                        )
                    )
                    outer = create_dataloader_from_config(config)
                    local = outer.dataloader
                    self.assertIsInstance(local, torch.utils.data.DataLoader)
                    self.assertEqual(local.num_workers, 0)
                    expected = list(local)
                    remote = remote_factory(outer)
                    # The remote factory must return the torch loader directly;
                    # another SDK wrapper would reset its worker count to zero.
                    self.assertIsInstance(remote, torch.utils.data.DataLoader)
                    self.assertEqual(remote.num_workers, workers)
                    self.assertEqual(remote.prefetch_factor, 3 if workers else None)
                    self.assertEqual(remote.persistent_workers, bool(workers))
                    try:
                        actual = list(remote)
                        pids = (
                            [p.pid for p in remote._iterator._workers]
                            if workers
                            else []
                        )
                        self.assertEqual(len(set(pids)), workers)
                        for _ in range(2):
                            self.assert_batches_equal(expected, actual)
                            actual = list(remote)
                        if workers:
                            self.assertEqual(
                                pids, [p.pid for p in remote._iterator._workers]
                            )
                            self.assertTrue(
                                all(p.is_alive() for p in remote._iterator._workers)
                            )
                        if diagnostic:
                            events = [
                                json.loads(line)
                                for path in Path(directory).rglob("*.jsonl")
                                for line in path.read_text().splitlines()
                            ]
                            children = [
                                r for r in events if r["event"] == "worker_initialized"
                            ]
                            self.assertEqual({r["pid"] for r in children}, set(pids))
                            self.assertEqual(
                                {r["worker_id"] for r in children}, set(range(workers))
                            )
                    finally:
                        if getattr(remote, "_iterator", None) is not None:
                            remote._iterator._shutdown_workers()

    def assert_batches_equal(self, expected, actual):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(expected, actual):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                x = left[key] if isinstance(left[key], list) else [left[key]]
                y = right[key] if isinstance(right[key], list) else [right[key]]
                self.assertEqual(len(x), len(y))
                for a, b in zip(x, y):
                    self.assertTrue(torch.equal(a, b), key)


if __name__ == "__main__":
    unittest.main()
