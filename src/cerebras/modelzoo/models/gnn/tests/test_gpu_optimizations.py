"""Sample-preserving native optimizations, including optimizer and dropout parity."""

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import numpy as np
import torch

from cerebras.modelzoo.models.gnn.gpu_policy import adamw_param_groups

from cerebras.modelzoo.models.gnn.architectures.graphsage import GraphSAGE
from cerebras.modelzoo.models.gnn.data_processing.batches import GraphSAGEBatch
from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
    GraphSAGENeighborSamplerDataset,
)
from cerebras.modelzoo.models.gnn.gpu_optimizations import (
    enable_topology_cache,
    fuse_depths,
)
from cerebras.modelzoo.models.gnn.task.loss import masked_classification_loss


def dataset(undirected=False):
    # Duplicate edges, a self loop, degree zero and a padded final target.
    return GraphSAGENeighborSamplerDataset(
        torch.arange(28, dtype=torch.float32).reshape(7, 4) / 20,
        torch.tensor([[0, 0, 1, 1, 2, 3], [1, 1, 0, 1, 3, 2]]),
        torch.tensor([0, 1, -100, 1, 0, 1, 0]),
        torch.ones(7, dtype=torch.bool),
        [3, 2, 2],
        4,
        shuffle=True,
        pad_id=0,
        seed=42,
        edge_index_is_undirected=undirected,
    )


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_topology_preserves_epochs_features_duplicates_and_padding(self):
        for undirected in (True, False):
            baseline, cached = dataset(undirected), dataset(undirected)
            cache = enable_topology_cache(cached)
            self.assertEqual(cache.nbytes, 7 * 7 * 9)
            for epoch in range(3):
                # Feature updates must be observed, unlike static batch replay.
                baseline.features.add_(1)
                cached.features.add_(1)
                for index in range(len(baseline)):
                    a, b = baseline[epoch, index], cached[epoch, index]
                    for name in a:
                        av = a[name] if isinstance(a[name], list) else [a[name]]
                        bv = b[name] if isinstance(b[name], list) else [b[name]]
                        for x, y in zip(av, bv):
                            torch.testing.assert_close(x, y, rtol=0, atol=0)
            ids, masks = cached._sample_neighbors(
                np.array([999, 6]), np.array([False, True]), 3, 0
            )
            self.assertFalse(masks.any())
            self.assertTrue((ids == 0).all())
            cached.seed += 1
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                cached[0]

    def test_cache_with_persistent_worker_and_epoch_reshuffle(self):
        from torch.utils.data import DataLoader
        from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
            EpochBatchSampler,
            _first_batch,
        )

        baseline, cached = dataset(), dataset()
        enable_topology_cache(cached)
        loaders = [
            DataLoader(
                source,
                batch_size=1,
                sampler=EpochBatchSampler(source),
                num_workers=1,
                persistent_workers=True,
                collate_fn=_first_batch,
            )
            for source in (baseline, cached)
        ]
        for _ in range(3):
            for a, b in zip(*loaders):
                for name in a:
                    av = a[name] if isinstance(a[name], list) else [a[name]]
                    bv = b[name] if isinstance(b[name], list) else [b[name]]
                    for x, y in zip(av, bv):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_cache_capacity_and_reenable(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            enable_topology_cache(dataset(), 1)
        source = dataset()
        enable_topology_cache(source)
        with self.assertRaisesRegex(ValueError, "already enabled"):
            enable_topology_cache(source)

    def test_logits_loss_grad_adamw_and_dropout(self):
        self.check_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_logits_loss_grad_adamw_and_dropout(self):
        self.check_parity("cuda")

    def check_parity(self, device):
        for aggregator in ("mean", "sum", "max"):
            for dropout in (0.0, 0.5):
                for training in (True, False):
                    with self.subTest(
                        aggregator=aggregator, dropout=dropout, training=training
                    ):
                        torch.manual_seed(1)
                        a = GraphSAGE(4, 8, 3, dropout, aggregator, 2).to(device)
                        b = fuse_depths(copy.deepcopy(a))
                        a.train(training)
                        b.train(training)
                        batch = GraphSAGEBatch.from_payload(dataset()[1]).to(
                            torch.device(device)
                        )
                        # A child node can be invalid even if its edge mask is true.
                        batch.node_masks[1][0, 0] = False
                        optimizers = [
                            torch.optim.AdamW(
                                adamw_param_groups(m),
                                lr=0.003,
                                eps=1e-6,
                                betas=(0.9, 0.999),
                                weight_decay=0.01,
                            )
                            for m in (a, b)
                        ]
                        values = []
                        for model, optimizer in zip((a, b), optimizers):
                            torch.manual_seed(99)
                            logits = model(batch)
                            loss = masked_classification_loss(
                                logits, batch.labels, batch.target_mask
                            )
                            loss.backward()
                            gradients = [p.grad.clone() for p in model.parameters()]
                            optimizer.step()
                            values.append((logits, loss, gradients))
                        for x, y in zip(values[0][:2], values[1][:2]):
                            torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)
                        mask = batch.target_mask & (batch.labels != -100)
                        self.assertEqual(
                            int(
                                ((values[0][0].argmax(-1) == batch.labels) & mask).sum()
                            ),
                            int(
                                ((values[1][0].argmax(-1) == batch.labels) & mask).sum()
                            ),
                        )
                        for x, y in zip(values[0][2], values[1][2]):
                            torch.testing.assert_close(x, y, rtol=3e-5, atol=3e-6)
                        for x, y in zip(a.parameters(), b.parameters()):
                            torch.testing.assert_close(x, y, rtol=3e-5, atol=3e-6)
                        for x, y in zip(
                            optimizers[0].state.values(), optimizers[1].state.values()
                        ):
                            for key in x:
                                torch.testing.assert_close(
                                    x[key], y[key], rtol=3e-5, atol=3e-6
                                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_native_cuda_switches_checkpoint_equivalence(self):
        from cerebras.modelzoo.models.gnn import fixed_shape_gpu as runner
        from cerebras.modelzoo.models.gnn.data_processing.sources.base import (
            BaseGraphDataSource,
        )
        from cerebras.modelzoo.models.gnn.tests.test_fixed_shape_gpu import (
            tiny_config,
            tiny_graph,
        )

        cfg = tiny_config()
        cfg["trainer"]["init"]["model"]["architecture"]["dropout"] = 0.5
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(BaseGraphDataSource, "load_graph", side_effect=tiny_graph),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            checkpoints = []
            for optimized in (False, True):
                output = Path(temp) / str(optimized)
                summary = runner.train(
                    cfg,
                    output,
                    device=torch.device("cuda"),
                    dtype=torch.float32,
                    warmup_steps=1,
                    fuse_depth_linear=optimized,
                    topology_cache_mb=int(optimized),
                )
                self.assertTrue(summary["completed"])
                self.assertEqual(summary["optimizer_steps"], 4)
                metadata = json.loads((output / "gpu_optimizations.json").read_text())
                self.assertEqual(metadata["fuse_depth_linear"], optimized)
                if optimized:
                    self.assertEqual(metadata["topology_cache_bytes"], 108)
                checkpoints.append(
                    torch.load(output / "checkpoint.pt", weights_only=False)
                )
            for key, value in checkpoints[0]["model"].items():
                torch.testing.assert_close(
                    value, checkpoints[1]["model"][key], rtol=3e-5, atol=3e-6
                )
            for key, state in checkpoints[0]["optimizer"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        value,
                        checkpoints[1]["optimizer"]["state"][key][name],
                        rtol=3e-5,
                        atol=3e-6,
                    )

    def test_empty_supervision(self):
        batch = GraphSAGEBatch.from_payload(dataset()[1])
        batch.target_mask.zero_()
        model = fuse_depths(GraphSAGE(4, 8, 3, 0.5, "mean", 2))
        loss = masked_classification_loss(model(batch), batch.labels, batch.target_mask)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(
            all(torch.count_nonzero(p.grad) == 0 for p in model.parameters())
        )


if __name__ == "__main__":
    unittest.main()
