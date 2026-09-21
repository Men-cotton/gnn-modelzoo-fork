"""Exercise GNN config -> neighbor processor -> real PyTorch DataLoader on a tiny graph."""

import tempfile
import unittest
from unittest.mock import patch

import torch

from cerebras.modelzoo.models.gnn.data_processing import processor as facade


class LoaderSettingsTests(unittest.TestCase):
    def test_forwarding_and_worker_zero(self):
        with tempfile.TemporaryDirectory() as data_dir:
            for workers, prefetch, persistent in [
                (0, 3, True),
                (1, 3, True),
                (1, 1, False),
            ]:
                with self.subTest(
                    workers=workers, prefetch=prefetch, persistent=persistent
                ):
                    with (
                        patch.object(facade.cstorch, "use_cs", return_value=False),
                        patch.object(
                            facade.cstorch.amp,
                            "get_floating_point_dtype",
                            return_value=torch.float32,
                        ),
                    ):
                        processor = facade.GNNDataProcessor(
                            dict(
                                data_processor="GNNDataProcessor",
                                dataset_name="ogbn-arxiv",
                                data_dir=data_dir,
                                sampling_mode="neighbor",
                                fanouts=[1],
                                batch_size=2,
                                split="train",
                                shuffle=False,
                                num_workers=workers,
                                prefetch_factor=prefetch,
                                persistent_workers=persistent,
                            )
                        )
                    graph = (
                        torch.ones(3, 2),
                        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                        torch.tensor([0, 1, 0]),
                        {"train": torch.ones(3, dtype=torch.bool)},
                    )
                    with (
                        patch.object(
                            processor._processor,
                            "prepare_graph_components",
                            return_value=graph,
                        ),
                    ):
                        loader = processor.create_dataloader()
                    self.assertEqual(loader.num_workers, workers)
                    self.assertEqual(
                        loader.prefetch_factor, prefetch if workers else None
                    )
                    self.assertEqual(
                        loader.persistent_workers,
                        persistent if workers else False,
                    )
                    first = list(loader)
                    worker_pids = (
                        [p.pid for p in loader._iterator._workers]
                        if loader.persistent_workers
                        else []
                    )
                    second = list(loader)
                    if loader.persistent_workers:
                        self.assertEqual(
                            worker_pids,
                            [p.pid for p in loader._iterator._workers],
                        )
                    self.assertEqual(len(first), 2)
                    # A padded tail still contains only one valid seed in the second batch.
                    self.assertEqual(int(first[-1]["target_mask"].sum()), 1)
                    for a, b in zip(first, second):
                        for key in a:
                            left = a[key] if isinstance(a[key], list) else [a[key]]
                            right = b[key] if isinstance(b[key], list) else [b[key]]
                            self.assertEqual(len(left), len(right))
                            for x, y in zip(left, right):
                                self.assertTrue(torch.equal(x, y), key)
                    del loader

    def test_invalid_values(self):
        config = dict(data_processor="GNNDataProcessor", dataset_name="ogbn-arxiv")
        for knobs in ({"num_workers": -1}, {"prefetch_factor": 0}):
            with self.assertRaises(ValueError):
                facade.GNNDataProcessorConfig(**config, **knobs)


if __name__ == "__main__":
    unittest.main()
