"""Check GNN loader configuration and repeated use at the real factory boundary."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from cerebras.appliance.appliance_client import fw_user_deserialize, fw_user_serialize
from cerebras.modelzoo.config import create_config_class
from cerebras.modelzoo.models.gnn.data_processing import processor as facade
from cerebras.modelzoo.models.gnn.data_processing.samplers import neighbor_tree
from cerebras.modelzoo.models.gnn.data_processing.sources.base import (
    BaseGraphDataSource,
)
from cerebras.modelzoo.models.gnn.fixed_shape_gpu import make_loader
from cerebras.modelzoo.trainer.utils import create_dataloader_from_config
from cerebras.pytorch.utils.nest import visit_torch_tensors


def tiny_graph():
    return Data(
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]]),
        y=torch.arange(4),
        train_mask=torch.tensor([True, True, True, False]),
        val_mask=torch.tensor([False, False, False, True]),
        test_mask=torch.ones(4, dtype=torch.bool),
    )


def remote_factory(loader):
    def roundtrip(value):
        return fw_user_deserialize(
            fw_user_serialize(value, from_usr=True, recurse=True), from_usr=True
        )

    builder = roundtrip(loader.input_fn)
    args, kwargs = roundtrip(loader.input_fn_params)
    return builder(*args, **kwargs)


class LoaderRegressionTests(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.enterContext(
            patch.object(
                facade.cstorch.amp,
                "get_floating_point_dtype",
                return_value=torch.float32,
            )
        )
        self.enterContext(patch.object(facade.cstorch, "use_cs", return_value=False))
        self.enterContext(
            patch.object(BaseGraphDataSource, "load_graph", side_effect=tiny_graph)
        )

    def config(self, **overrides):
        return dict(
            data_processor="GNNDataProcessor",
            dataset_name="ogbn-arxiv",
            data_dir=self.directory.name,
            sampling_mode="neighbor",
            fanouts=[2],
            batch_size=2,
            **overrides,
        )

    def test_full_graph_remote_worker_settings_and_default_split(self):
        for workers in (0, 1):
            with self.subTest(workers=workers):
                config = self.config(
                    num_workers=workers,
                    prefetch_factor=7,
                    persistent_workers=True,
                    pin_memory=False,
                )
                config.update(sampling_mode="full_graph", batch_size=1)
                outer = create_dataloader_from_config(
                    create_config_class(facade.GNNDataProcessor)(config=config)
                )
                local = next(iter(outer.dataloader))
                torch.testing.assert_close(
                    local["target_mask"], tiny_graph().train_mask
                )
                remote = remote_factory(outer)
                self.assertEqual(remote.num_workers, workers)
                self.assertEqual(remote.prefetch_factor, 7 if workers else None)
                self.assertEqual(remote.persistent_workers, bool(workers))
                self.assertFalse(remote.pin_memory)
                try:
                    first = next(iter(remote))
                    pids = (
                        [worker.pid for worker in remote._iterator._workers]
                        if workers
                        else []
                    )
                    second = next(iter(remote))
                    if workers:
                        self.assertEqual(
                            pids, [worker.pid for worker in remote._iterator._workers]
                        )
                    torch.testing.assert_close(first["features"], second["features"])
                    # The actual emitted dict must expose adjacency to the SDK.
                    scopes = {tuple(scope) for scope, _ in visit_torch_tensors(first)}
                    self.assertIn(("adjacency", "edge_index"), scopes)
                    self.assertIn(("adjacency", "edge_weight"), scopes)
                finally:
                    if getattr(remote, "_iterator", None) is not None:
                        remote._iterator._shutdown_workers()

    def test_logical_drop_last_applies_to_trainer_and_native_loader(self):
        for drop_last, expected in ((False, [2, 1]), (True, [2])):
            config = self.config(drop_last_batch=drop_last, pin_memory=False)
            with self.subTest(drop_last=drop_last):
                outer = create_dataloader_from_config(
                    create_config_class(facade.GNNDataProcessor)(config=config)
                )
                loaders = [
                    outer.dataloader,
                    remote_factory(outer),
                    make_loader(config, num_layers=1, float_dtype=torch.float32),
                ]
                for loader in loaders:
                    self.assertFalse(loader.pin_memory)
                    self.assertEqual(
                        [int(batch["target_mask"].sum()) for batch in loader], expected
                    )

    def test_drop_last_cannot_silently_empty_training_loader(self):
        config = self.config(drop_last_batch=True)
        config["batch_size"] = 4
        with self.assertRaisesRegex(ValueError, "discard all target nodes"):
            facade.GNNDataProcessor(config).create_dataloader()

    def test_multi_streamer_fails_before_graph_loading(self):
        # Serialize the same factory the SDK sends to its remote workers.
        for mode in ("neighbor", "full_graph"):
            with self.subTest(mode=mode):
                config = self.config()
                config.update(sampling_mode=mode, batch_size=1)
                outer = create_dataloader_from_config(
                    create_config_class(facade.GNNDataProcessor)(config=config)
                )
                with (
                    patch.object(facade.cstorch, "use_cs", return_value=True),
                    patch.object(facade.dist, "num_streamers", return_value=2),
                    patch.object(BaseGraphDataSource, "load_graph") as load_graph,
                    self.assertRaisesRegex(ValueError, "one CSX input streamer"),
                ):
                    remote_factory(outer)
                load_graph.assert_not_called()

    def test_repeated_papers_loader_reuses_indexes_after_releasing_edges(self):
        for cache in (None, 1.0):
            with self.subTest(cache_fraction=cache):
                config = self.config(cache_fraction=cache, pin_memory=False)
                config["dataset_name"] = "ogbn-papers100M"
                processor = facade.GNNDataProcessor(config)._processor
                processor._graph_data_cache = tiny_graph()
                # Restore the real cache-aware source method for this regression.
                with patch.object(
                    processor, "load_graph", return_value=processor._graph_data_cache
                ):
                    first = processor.create_torch_dataloader()
                    self.assertIsNone(processor._graph_data_cache.edge_index)
                    indexes = processor._neighbor_indexes
                    with patch.object(
                        neighbor_tree.NeighborSliceIndex,
                        "from_edge_index",
                        side_effect=AssertionError(
                            "rebuilding would lose released edges"
                        ),
                    ):
                        second = processor.create_torch_dataloader()
                    self.assertIs(
                        first.dataset._forward_index, second.dataset._forward_index
                    )
                    self.assertIs(indexes[1], second.dataset._reverse_index)
                    for a, b in zip(first, second):
                        self.assertEqual(
                            int(a["neighbor_masks"][0].sum()),
                            int(b["neighbor_masks"][0].sum()),
                        )
                        self.assertGreater(int(b["neighbor_masks"][0].sum()), 0)
                        for x, y in zip(a["node_features"], b["node_features"]):
                            torch.testing.assert_close(x, y)

    def test_implicit_undirected_neighbors_have_no_duplicates(self):
        graph = tiny_graph()
        # Include repeated outgoing edges, reciprocal edges, and a self-loop.
        edges = torch.tensor([[0, 0, 1, 0, 0], [2, 1, 0, 2, 0]])
        datasets = []
        for undirected in (False, True):
            datasets.append(
                neighbor_tree.GraphSAGENeighborSamplerDataset(
                    features=graph.x,
                    edge_index=(
                        to_undirected(edges, num_nodes=4) if undirected else edges
                    ),
                    labels=graph.y,
                    mask=torch.ones(4, dtype=torch.bool),
                    fanouts=[5],
                    batch_size=4,
                    shuffle=False,
                    pad_id=0,
                    seed=0,
                    edge_index_is_undirected=undirected,
                )
            )
        implicit, materialized = datasets
        first, reverse = implicit._neighbor_slices(0)
        self.assertEqual(first.tolist(), [2, 1, 0])
        self.assertIsNone(reverse)
        for node in range(4):
            a, _ = implicit._neighbor_slices(node)
            b, _ = materialized._neighbor_slices(node)
            self.assertEqual(sorted(a.tolist()), sorted(b.tolist()))
        a, b = implicit[0], materialized[0]
        torch.testing.assert_close(a["neighbor_masks"][0], b["neighbor_masks"][0])
        torch.testing.assert_close(
            a["node_features"][1].sum(dim=1), b["node_features"][1].sum(dim=1)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_gpu_cache_and_pinning_with_zero_or_multiple_workers(self):
        for workers in (0, 1):
            for fraction in (0.0, 1.0):
                for pin in (False, True):
                    with self.subTest(workers=workers, cache=fraction, pin=pin):
                        processor = facade.GNNDataProcessor(
                            self.config(
                                num_workers=workers,
                                cache_fraction=fraction,
                                pin_memory=pin,
                                persistent_workers=False,
                            )
                        )
                        with patch.object(
                            facade.cstorch,
                            "backend",
                            return_value=SimpleNamespace(device=torch.device("cuda:0")),
                        ):
                            loader = processor.create_dataloader()
                        batch = next(iter(loader))
                        features = batch["node_features"][0]
                        self.assertEqual(
                            features.device.type, "cpu" if workers else "cuda"
                        )
                        self.assertEqual(loader.pin_memory, pin and bool(workers))
                        if workers:
                            self.assertEqual(features.is_pinned(), pin)


if __name__ == "__main__":
    unittest.main()
