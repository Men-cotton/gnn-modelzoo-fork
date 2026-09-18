"""Bounded host optimization check; this does not validate physical CSX training."""

import unittest
from unittest.mock import patch

import torch

from cerebras.modelzoo.models.gnn import fixed_shape_gpu as runner
from cerebras.modelzoo.models.gnn.data_processing.batches import GraphSAGEBatch
from cerebras.modelzoo.models.gnn.data_processing.sources.base import (
    BaseGraphDataSource,
)
from cerebras.modelzoo.models.gnn.model import GNNModel
from cerebras.modelzoo.models.gnn.tests.test_fixed_shape_gpu import (
    tiny_config,
    tiny_graph,
)


class HostLearnabilityTests(unittest.TestCase):
    def test_shared_sampler_and_model_fit_isolated_node_and_padded_tail(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            cfg = tiny_config()
            model_cfg = cfg["trainer"]["init"]["model"]
            model_cfg["task"]["compute_eval_metrics"] = False
            model_cfg["architecture"]["hidden_dim"] = 16
            model = GNNModel(model_cfg)
            with patch.object(
                BaseGraphDataSource, "load_graph", side_effect=tiny_graph
            ):
                loader = runner.make_loader(
                    cfg["trainer"]["fit"]["train_dataloader"],
                    num_layers=2,
                    float_dtype=torch.float32,
                )
            batches = [GraphSAGEBatch.from_payload(payload) for payload in loader]
            # The second batch contains the isolated target and one padded slot.
            self.assertEqual(
                [int(batch.target_mask.sum()) for batch in batches], [2, 1]
            )
            self.assertFalse(batches[-1].neighbor_masks[0].any())

            def objective():
                with torch.no_grad():
                    return (
                        sum(
                            float(model(batch)) * int(batch.target_mask.sum())
                            for batch in batches
                        )
                        / 3
                    )

            initial_loss = objective()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.02,
                weight_decay=0.0,
                eps=1e-6,
                betas=(0.9, 0.999),
            )
            for _ in range(100):
                for batch in batches:
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(batch)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    optimizer.step()

            self.assertLess(objective(), initial_loss * 0.01)
            actual = runner.evaluate(model, loader, torch.device("cpu"), torch.float32)
            self.assertEqual(actual, {"accuracy": 1.0, "targets": 3})


if __name__ == "__main__":
    unittest.main()
