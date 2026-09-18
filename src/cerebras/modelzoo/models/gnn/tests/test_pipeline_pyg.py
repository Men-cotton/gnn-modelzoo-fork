"""Small CPU fixtures for PyG configuration, partition and measurement boundaries."""

import asyncio
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data
from torch_geometric.distributed import DistContext, DistNeighborSampler
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler import SamplerOutput
from torch_geometric.utils import to_undirected

from cerebras.modelzoo.models.gnn.gpu_policy import (
    adamw_kwargs,
    precision_dtype,
)
from cerebras.modelzoo.models.gnn.reference.pyg import data as loaders
from cerebras.modelzoo.models.gnn.reference.pyg import train
from cerebras.modelzoo.models.gnn.reference.pyg.eval import (
    evaluate,
    evaluate_full_batch,
)
from cerebras.modelzoo.models.gnn.reference.pyg.utils import wrap_ddp
from cerebras.modelzoo.models.gnn.tools import download_datasets as downloader
from cerebras.modelzoo.models.gnn.tools import partition_dataset as prep
from cerebras.modelzoo.models.gnn.tools.measure_window import summarize


class TinyDataset:
    def __init__(self):
        src = torch.arange(8)
        dst = (src + 1) % 8
        self.data = Data(
            x=torch.arange(16, dtype=torch.float).view(8, 2),
            y=torch.arange(8) % 2,
            edge_index=torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]),
        )

    def __getitem__(self, index):
        return self.data

    def get_idx_split(self):
        return {name: torch.arange(8) for name in ("train", "valid", "test")}


def gz(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        stream.write(value)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)

    def test_precision_and_optimizer_reach_training(self):
        class Captured(Exception):
            pass

        init = {
            "loop": {"max_steps": 1},
            "model": {"task": {"to_float16": True}},
            "model_dir": str(self.root),
            "logging": {"log_steps": 1},
            "optimizer": {
                "AdamW": {
                    "learning_rate": 0.01,
                    "betas": [0.6, 0.8],
                    "eps": 0.03,
                }
            },
        }
        for precision, expected_dtype, scale in (
            (None, torch.float16, True),
            ({"enabled": False}, torch.float32, False),
            ({"enabled": True, "fp16_type": "bfloat16"}, torch.bfloat16, False),
        ):
            with self.subTest(precision=precision):
                config = deepcopy(init)
                if precision is not None:
                    config["precision"] = precision
                with (
                    patch.object(train, "AdamW") as optimizer,
                    patch.object(
                        torch.amp, "GradScaler", side_effect=Captured
                    ) as scaler,
                    patch.object(
                        train, "precision_dtype", wraps=precision_dtype
                    ) as resolve,
                ):
                    with self.assertRaises(Captured):
                        train.train_model(
                            {"trainer": {"init": config}},
                            torch.nn.Linear(2, 2),
                            ([], None),
                            None,
                            None,
                            torch.device("cuda"),
                        )
                self.assertEqual(
                    optimizer.call_args.kwargs,
                    {"lr": 0.01, "betas": [0.6, 0.8], "eps": 0.03},
                )
                self.assertEqual(
                    precision_dtype(
                        *resolve.call_args.args, **resolve.call_args.kwargs
                    ),
                    expected_dtype,
                )
                self.assertEqual(scaler.call_args.kwargs["enabled"], scale)
        self.assertEqual(precision_dtype({}, default=torch.float32), torch.float32)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            precision_dtype({"precision": {"fp16_type": "cbfloat16"}})
        self.assertEqual(init["optimizer"]["AdamW"]["learning_rate"], 0.01)
        with self.assertRaisesRegex(ValueError, "parameter groups"):
            adamw_kwargs({"optimizer": {"AdamW": {"params": [{"lr": 0.2}]}}})

    def test_fit_and_standalone_validation_select_their_own_split(self):
        loader = {"dataset_name": "fixture", "sampling_mode": "full_graph"}
        cfg = {
            "trainer": {
                "fit": {
                    "train_dataloader": dict(loader, split="train"),
                    "val_dataloader": dict(loader, split="test"),
                },
                "validate": {"val_dataloader": dict(loader, split="valid")},
            }
        }
        graph = TinyDataset().data
        splits = {
            "train": torch.tensor([0]),
            "valid": torch.tensor([1]),
            "test": torch.tensor([2, 3]),
        }
        _, val = loaders.make_loaders(graph, splits, cfg)
        self.assertEqual(next(iter(val)).node_idx.tolist(), [2, 3])
        cfg["trainer"].pop("validate")
        _, val = loaders.make_loaders(graph, splits, cfg)
        self.assertEqual(next(iter(val)).node_idx.tolist(), [2, 3])
        cfg = {"trainer": {"validate": {"val_dataloader": dict(loader, split="valid")}}}
        val = loaders.make_validation_loader(graph, splits, cfg)
        self.assertEqual(next(iter(val)).node_idx.tolist(), [1])
        self.assertNotIn("fit", cfg["trainer"])

    def test_evaluation_uses_selected_autocast_dtype(self):
        class Model(torch.nn.Module):
            def forward(self, x, edge_index):
                out = x @ torch.eye(2)
                self.dtype = out.dtype
                return out

        model = Model()
        graph = Data(
            x=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            y=torch.tensor([0, 1]),
            node_idx=torch.arange(2),
        )
        self.assertEqual(
            evaluate(model, [graph], torch.device("cpu"), dtype=torch.bfloat16),
            1.0,
        )
        self.assertEqual(model.dtype, torch.bfloat16)

    def test_null_split_defaults_for_fit_and_standalone_validation(self):
        splits = {"train": torch.tensor([0, 2]), "valid": torch.tensor([1, 3])}
        for mode in ("full_graph", "neighbor"):
            with self.subTest(mode=mode):
                config = {
                    "dataset_name": "fixture",
                    "sampling_mode": mode,
                    "split": None,
                    "batch_size": 2,
                    "fanouts": [1],
                    "shuffle": False,
                    "drop_last_batch": False,
                    "sampler_seed": 42,
                    "pin_memory": False,
                    "num_workers": 0,
                }
                cfg = {
                    "trainer": {
                        "fit": {
                            "train_dataloader": dict(config),
                            "val_dataloader": dict(config),
                        },
                        "validate": {"val_dataloader": dict(config)},
                    }
                }
                graph = TinyDataset().data
                training, validation = loaders.make_loaders(graph, splits, cfg)
                standalone = loaders.make_validation_loader(graph, splits, cfg)
                for loader, expected in (
                    (training, [0, 2]),
                    (validation, [1, 3]),
                    (standalone, [1, 3]),
                ):
                    batch = next(iter(loader))
                    nodes = (
                        batch.node_idx
                        if mode == "full_graph"
                        else batch.n_id[: batch.batch_size]
                    )
                    self.assertEqual(nodes.tolist(), expected)
                self.assertIsNone(cfg["trainer"]["validate"]["val_dataloader"]["split"])

    def test_evaluation_excludes_ignored_labels_in_all_batch_paths(self):
        class Model(torch.nn.Module):
            def forward(self, x, edge_index, batch_size=None):
                return x

        model = Model()
        device = torch.device("cpu")
        for labels, expected in (
            ([0, -100, 1], 0.5),
            ([-100, -100, -100], 0.0),
        ):
            with self.subTest(labels=labels):
                graph = Data(
                    x=torch.tensor([[2.0, 0.0]] * 3),
                    edge_index=torch.empty(2, 0, dtype=torch.long),
                    y=torch.tensor(labels),
                    node_idx=torch.arange(3),
                    batch_size=3,
                )
                self.assertEqual(
                    evaluate_full_batch(model, graph, graph.node_idx, device),
                    expected,
                )
                self.assertEqual(evaluate(model, [graph], device), expected)
                del graph.node_idx
                self.assertEqual(evaluate(model, [graph], device), expected)

    def test_ddp_uses_local_device(self):
        with patch("torch.nn.parallel.DistributedDataParallel") as ddp:
            wrap_ddp(object(), torch.device("cuda:1"))
        self.assertEqual(ddp.call_args.kwargs["device_ids"], [1])

    def test_partition_labels_survive_actual_pyg_collation(self):
        with patch.object(prep, "get_dataset", return_value=TinyDataset()):
            prep.partition_dataset(
                "ogbn-arxiv", str(self.root), 2, dataset_dir=str(self.root)
            )
        partition = prep.get_partition_path("ogbn-arxiv", self.root, 2)
        stores, splits = loaders.load_dist_partition(str(partition), 0)
        ids = splits["train"]
        sampler = DistNeighborSampler(
            current_ctx=DistContext(
                rank=0,
                global_rank=0,
                world_size=2,
                global_world_size=2,
                group_name="test",
            ),
            data=stores,
            num_neighbors=[1],
        )
        loader = object.__new__(NodeLoader)
        loader.data, loader.node_sampler = stores, sampler
        loader.transform_sampler_output = loader.transform = None
        sampled = SamplerOutput(
            node=ids,
            row=torch.empty(0, dtype=torch.long),
            col=torch.empty(0, dtype=torch.long),
            edge=None,
            metadata=(ids, None),
        )
        batch = loader.filter_fn(asyncio.run(sampler._collate_fn(sampled)))
        torch.testing.assert_close(batch.y, ids % 2)
        (partition.parent / "ogbn-arxiv-label" / "label.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "labels"):
            loaders.load_dist_partition(str(partition), 0)

    def test_heterogeneous_map_and_explicit_model_boundary(self):
        partition = self.root / "ogbn-mag-partitions"
        (partition / "node_map").mkdir(parents=True)
        torch.save(torch.tensor([1, 0, 1, 0]), partition / "node_map" / "paper.pt")
        prep.save_partitions({"train": torch.arange(4)}, "ogbn-mag", 2, str(self.root))
        for index, expected in enumerate(([1, 3], [0, 2])):
            actual = torch.load(
                self.root / "ogbn-mag-train-partitions" / f"partition{index}.pt",
                weights_only=True,
            )
            self.assertEqual(actual.tolist(), expected)
        (partition / "META.json").write_text(json.dumps({"is_hetero": True}))
        with self.assertRaisesRegex(ValueError, "Heterogeneous partitions"):
            loaders.load_dist_partition(str(partition), 0)
        (partition / "node_map" / "paper.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "ownership map"):
            prep.save_partitions(
                {"train": torch.arange(4)}, "ogbn-mag", 2, str(self.root)
            )

    def test_arxiv_partition_transform_matches_regular_graph(self):
        with (
            patch.object(prep, "ensure_pickle_friendly_load"),
            patch.object(prep, "PygNodePropPredDataset") as factory,
        ):
            prep.get_dataset("ogbn-arxiv", str(self.root))
        graph = Data(
            x=torch.ones(3, 2), edge_index=torch.tensor([[0, 0, 1], [1, 1, 2]])
        )
        expected = to_undirected(graph.edge_index, num_nodes=3)
        actual = factory.call_args.kwargs["transform"](graph).edge_index
        torch.testing.assert_close(actual, expected)

    def test_measurement_rejects_only_overlapping_activity(self):
        path = self.root / "train.log"
        start = "2026-01-01 00:00:40,000 INFO | Train Device=CSX, Step=40, Loss=1.0,"
        end = "2026-01-01 00:04:00,000 INFO | Train Device=CSX, Step=240, Loss=1.0,"
        complete = "Training completed successfully!"
        for event in (
            "| Eval Device=CSX, Step=60",
            "Saving checkpoint checkpoint_100.mdl",
            "Checkpoint saved: checkpoint_100.mdl",
        ):
            path.write_text("\n".join((start, event, end, complete)))
            with self.assertRaisesRegex(ValueError, "overlaps"):
                summarize(path)
            path.write_text("\n".join((event, start, end, event, complete)))
            self.assertEqual(summarize(path)["training_window_seconds"], 200)

    def test_mag_raw_layout_and_reuse_without_download(self):
        dataset = self.root / "ogbn-mag" / "ogbn_mag"
        raw = dataset / "raw"
        metadata = {
            "split": "time",
            "binary": "False",
            "is hetero": "True",
            "has_node_attr": "True",
            "download_name": "mag",
            "url": "https://example.invalid/mag.zip",
        }
        for name, value in {
            "triplet-type-list.csv.gz": "paper,cites,paper\n",
            "num-node-dict.csv.gz": "paper\n2\n",
            "nodetype-has-label.csv.gz": "paper\nTrue\n",
            "node-label/paper/node-label.csv.gz": "0\n1\n",
            "node-feat/paper/node-feat.csv.gz": "1,2\n3,4\n",
            "relations/paper___cites___paper/edge.csv.gz": "0,1\n",
            "relations/paper___cites___paper/num-edge-list.csv.gz": "1\n",
        }.items():
            gz(raw / name, value)
        split = dataset / "split" / "time"
        gz(split / "nodetype-has-split.csv.gz", "paper\nTrue\n")
        for name in ("train", "valid", "test"):
            gz(split / "paper" / f"{name}.csv.gz", "0\n")
        self.assertTrue(downloader._ogb_raw_is_ready(dataset, metadata))
        with (
            patch.object(
                downloader, "_get_ogb_dataset_metadata", return_value=metadata
            ),
            patch.object(
                downloader,
                "_request_download",
                side_effect=AssertionError("unexpected prompt"),
            ),
            patch.object(
                downloader,
                "_download_with_resume",
                side_effect=AssertionError("unexpected network"),
            ),
            patch.object(downloader, "PygNodePropPredDataset") as processor,
        ):
            processor.return_value.get_idx_split.return_value = {}
            downloader.download_ogb_dataset("ogbn-mag", self.root)
            processor.assert_called_once()
        (split / "paper" / "test.csv.gz").unlink()
        self.assertFalse(downloader._ogb_raw_is_ready(dataset, metadata))

    def test_empty_processed_directory_is_not_prepared(self):
        processed = self.root / "ogbn-arxiv" / "ogbn_arxiv" / "processed"
        processed.mkdir(parents=True)
        with patch.object(
            downloader, "_request_download", return_value=False
        ) as request:
            downloader.download_ogb_dataset("ogbn-arxiv", self.root)
        request.assert_called_once_with("ogbn-arxiv")


if __name__ == "__main__":
    unittest.main()
