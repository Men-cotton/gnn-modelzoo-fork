"""Compare configured PyG and fixed-shape aggregations against a small oracle."""

from types import SimpleNamespace
import unittest

import torch

from cerebras.modelzoo.models.gnn.architectures.graphsage import GraphSAGE
from cerebras.modelzoo.models.gnn.data_processing.batches import GraphSAGEBatch
from cerebras.modelzoo.models.gnn.data_processing.samplers.neighbor_tree import (
    GraphSAGENeighborSamplerDataset,
)
from cerebras.modelzoo.models.gnn.reference.pyg.model import get_model


def config(aggregator):
    return {
        "trainer": {
            "init": {
                "model": {
                    "architecture": {
                        "type": "graphsage",
                        "n_feat": 2,
                        "hidden_dim": 2,
                        "n_class": 2,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "aggregator": aggregator,
                    }
                }
            }
        }
    }


class PyGAggregatorTests(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)

    def test_configured_aggregation_matches_manual_and_fixed_shape_oracles(self):
        # Negative features distinguish a real max from an incorrectly padded
        # zero. The isolated node must receive zero aggregate for every mode.
        features = torch.tensor([[-4.0, 1.0], [-2.0, -3.0], [3.0, -5.0], [7.0, 4.0]])
        edges = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
        dataset = GraphSAGENeighborSamplerDataset(
            features=features,
            edge_index=edges,
            labels=torch.zeros(4, dtype=torch.long),
            mask=torch.ones(4, dtype=torch.bool),
            fanouts=[3],
            batch_size=4,
            shuffle=False,
            pad_id=0,
            seed=0,
            edge_index_is_undirected=True,
        )
        batch = GraphSAGEBatch.from_payload(dataset[0])
        for aggregator in ("mean", "sum", "max"):
            with self.subTest(aggregator=aggregator):
                pyg = get_model(config(aggregator))
                fixed = GraphSAGE(2, 2, 1, 0.0, aggregator, 2)
                with torch.no_grad():
                    pyg.gnn.convs[0].lin_l.weight.copy_(torch.eye(2))
                    pyg.gnn.convs[0].lin_l.bias.zero_()
                    pyg.gnn.convs[0].lin_r.weight.copy_(0.25 * torch.eye(2))
                    fixed.layers[0].neighbor_linear.weight.copy_(torch.eye(2))
                    fixed.layers[0].neighbor_linear.bias.zero_()
                    fixed.layers[0].self_linear.weight.copy_(0.25 * torch.eye(2))
                    fixed.layers[0].self_linear.bias.zero_()
                    for model in (pyg, fixed):
                        model.classifier.weight.copy_(torch.eye(2))
                        model.classifier.bias.zero_()
                expected = 0.25 * features
                for node in range(4):
                    neighbors = features[edges[0, edges[1] == node]]
                    if neighbors.numel() == 0:
                        continue
                    if aggregator == "mean":
                        pooled = neighbors.mean(0)
                    elif aggregator == "sum":
                        pooled = neighbors.sum(0)
                    else:
                        pooled = neighbors.max(0).values
                    expected[node] += pooled
                torch.testing.assert_close(pyg(features, edges), expected)
                torch.testing.assert_close(fixed(batch), expected)

    def test_every_layer_receives_aggregator_and_mean_remains_default(self):
        cfg = config("sum")
        architecture = cfg["trainer"]["init"]["model"]["architecture"]
        architecture["num_layers"] = 3
        self.assertEqual(
            [layer.aggr for layer in get_model(cfg).gnn.convs], ["sum"] * 3
        )
        architecture.pop("aggregator")
        self.assertEqual(
            [layer.aggr for layer in get_model(cfg).gnn.convs], ["mean"] * 3
        )

    def test_unsupported_aggregators_are_rejected_before_cagnet_construction(self):
        args = SimpleNamespace(
            cagnet_rows=1, cagnet_cols=1, cagnet_rep=1, force_cagnet=True
        )
        for aggregator in ("sum", "max"):
            with self.subTest(aggregator=aggregator):
                with self.assertRaisesRegex(ValueError, "mean aggregator only"):
                    get_model(config(aggregator), args=args, num_nodes=4)
        for args_value in (None, args):
            with self.assertRaisesRegex(ValueError, "Unsupported GraphSAGE aggregator"):
                get_model(config("min"), args=args_value, num_nodes=4)


if __name__ == "__main__":
    unittest.main()
