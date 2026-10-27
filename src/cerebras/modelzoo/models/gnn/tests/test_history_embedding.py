import copy
import unittest

import torch

from cerebras.modelzoo.models.gnn.tools.history_embedding.core import (
    HistorySAGE,
    HistoryStore,
    infer,
    masked_loss,
    neighbor_mean,
    prepare,
    prepare_tree,
    prepare_tree_mean,
    refresh,
)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)
        self.x = torch.randn(6, 4)
        self.n = torch.tensor(
            [[1, 2, 0], [0, 3, 0], [0, 3, 4], [1, 2, 5], [2, 0, 0], [0, 0, 0]]
        )
        self.mask = torch.tensor(
            [[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 0]],
            dtype=torch.bool,
        )
        self.labels = torch.tensor([0, 1, 0, 1, 1, -100])
        self.model = HistorySAGE(4, 7, 3, 2)
        self.mean = neighbor_mean(self.x, self.n, self.mask)
        self.store = HistoryStore(refresh(self.model, self.x, self.n, self.mask))

    def batch(self, roots, size=None):
        return prepare(
            self.x, self.mean, self.n, self.mask, self.labels, roots, self.store, size
        )

    def tree_batch(self, roots, layers=3):
        return prepare_tree(self.x, self.n, self.mask, self.labels, roots, layers)

    def test_fresh_partial_and_isolated_forward_match_recursive(self):
        roots = torch.tensor([3, 0, 5])
        actual, _ = self.model(self.batch(roots, 5))
        expected = self.model(self.tree_batch(roots))[0]
        torch.testing.assert_close(actual[:3], expected)
        torch.testing.assert_close(
            infer(self.model, self.x, self.n, self.mask)[roots], expected
        )

    def test_complete_batch_gradients_match_recursive(self):
        roots = torch.tensor([3, 0, 4, 2, 5, 1])
        baseline = copy.deepcopy(self.model)
        batch = self.batch(roots)
        actual, _ = self.model(batch)
        masked_loss(actual, batch["labels"], batch["valid"]).backward()
        expected = baseline(self.tree_batch(roots))[0]
        masked_loss(
            expected, self.labels[roots], torch.ones(6, dtype=torch.bool)
        ).backward()
        for a, b in zip(self.model.parameters(), baseline.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-5)

    def test_partial_fresh_forward_does_not_imply_exact_gradient(self):
        roots = torch.tensor([0])
        baseline = copy.deepcopy(self.model)
        batch = self.batch(roots)
        actual, _ = self.model(batch)
        expected = baseline(self.tree_batch(roots))[0]
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        self.assertGreater(
            (
                self.model.layers[0].self_linear.weight.grad
                - baseline.layers[0].self_linear.weight.grad
            )
            .abs()
            .max()
            .item(),
            1e-5,
        )
        self.assertTrue(all(not h.requires_grad for h in self.store.values))

    def test_cached_boundary_mean_matches_gather_and_invalidates_on_write(self):
        roots = torch.tensor([3, 1, 5])
        gathered = self.batch(roots, 7)
        self.store.means = [
            neighbor_mean(h, self.n, self.mask) for h in self.store.values
        ]
        cached = self.batch(roots, 7)
        for a, b in zip(gathered["boundary_mean"], cached["boundary_mean"]):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
        _, values = self.model(cached)
        self.store.write(roots, [h[:3] for h in values], 1)
        self.assertIsNone(self.store.means)
        self.assertEqual(self.store.version.tolist(), [0, 1, 0, 1, 0, 1])

    def test_padding_and_ignored_labels_do_not_change_loss_or_gradient(self):
        roots = torch.tensor([3, 5, 1])
        other = copy.deepcopy(self.model)
        a, b = self.batch(roots), self.batch(roots, 8)
        la = masked_loss(self.model(a)[0], a["labels"], a["valid"])
        lb = masked_loss(other(b)[0], b["labels"], b["valid"])
        torch.testing.assert_close(la, lb)
        la.backward()
        lb.backward()
        for p, q in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
        ignored = masked_loss(other(b)[0], torch.full((8,), -100), b["valid"])
        self.assertEqual(ignored.item(), 0)

    def test_invalid_write_is_atomic(self):
        roots = torch.tensor([0, 1])
        before = [h.clone() for h in self.store.values]
        with self.assertRaises(ValueError):
            self.store.write(roots, [torch.ones(2, 7), torch.ones(2, 6)], 3)
        with self.assertRaises(ValueError):
            self.store.write(torch.tensor([1, 1]), [torch.ones(2, 7)] * 2, 3)
        with self.assertRaises(ValueError):
            self.store.write(roots, [torch.ones(2, 7)] * 2, -1)
        for a, b in zip(before, self.store.values):
            torch.testing.assert_close(a, b)

    def test_one_layer_needs_no_history(self):
        model = HistorySAGE(4, 7, 1, 2)
        store = HistoryStore([])
        roots = torch.tensor([0, 3])
        batch = prepare(self.x, self.mean, self.n, self.mask, self.labels, roots, store)
        torch.testing.assert_close(
            model(batch)[0], model(self.tree_batch(roots, layers=1))[0]
        )

    def test_sdk_sample_survives_dataloader_tuple_to_list_conversion(self):
        from torch.utils.data import default_convert
        from cerebras.modelzoo.models.gnn.tools.history_embedding.hardware import (
            StaticModel,
            pack_batch,
        )

        model = StaticModel(4, 7, 3, 2)
        model.load_state_dict(self.model.state_dict())
        roots = torch.tensor([3, 0, 5])
        batches = [
            self.batch(roots, 5),
            prepare_tree(self.x, self.n, self.mask, self.labels, roots, 3, 5),
            prepare_tree_mean(
                self.x, self.mean, self.n, self.mask, self.labels, roots, 3, 5
            ),
        ]
        for batch in batches:
            converted = default_convert(pack_batch(batch))
            torch.testing.assert_close(model(converted)[0], self.model(batch)[0])

    def test_no_history_preaggregation_preserves_forward_and_gradients(self):
        roots = torch.tensor([3, 0, 5])
        other = copy.deepcopy(self.model)
        batch = prepare_tree_mean(
            self.x, self.mean, self.n, self.mask, self.labels, roots, 3
        )
        actual = self.model(batch)[0]
        expected = other(self.tree_batch(roots))[0]
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        for p, q in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

    def test_preaggregation_preserves_native_dropout_order_and_gradients(self):
        model = HistorySAGE(4, 7, 3, 2, dropout=0.35)
        other = copy.deepcopy(model)
        roots = torch.tensor([3, 0, 5])
        dense = prepare_tree(self.x, self.n, self.mask, self.labels, roots, 3, 5)
        means = prepare_tree_mean(
            self.x, self.mean, self.n, self.mask, self.labels, roots, 3, 5
        )
        torch.manual_seed(29)
        expected = model(dense)[0]
        torch.manual_seed(29)
        actual = other(means)[0]
        torch.testing.assert_close(actual, expected)
        masked_loss(expected, dense["labels"], dense["valid"]).backward()
        masked_loss(actual, means["labels"], means["valid"]).backward()
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad)

    def test_native_model_semantics(self):
        # Existing native model remains the semantic reference, without edits.
        from cerebras.modelzoo.models.gnn.architectures.graphsage import GraphSAGE
        from cerebras.modelzoo.models.gnn.data_processing.batches import GraphSAGEBatch

        native = GraphSAGE(4, 7, 3, 0.0, "mean", 2)
        native.load_state_dict(self.model.state_dict())
        roots = torch.tensor([0, 3, 5])
        payload = prepare_tree(self.x, self.n, self.mask, self.labels, roots, 3)
        batch = GraphSAGEBatch(
            list(payload["node_features"]),
            [torch.ones(len(x), dtype=torch.bool) for x in payload["node_features"]],
            list(payload["neighbor_masks"]),
            self.labels[roots],
            payload["valid"],
        )
        torch.testing.assert_close(native(batch), self.model(payload)[0])


if __name__ == "__main__":
    unittest.main()
