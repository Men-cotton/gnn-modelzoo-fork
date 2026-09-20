"""Exercise shared fixed batches and native training without downloading a graph."""

import copy
from datetime import datetime, timedelta
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from cerebras.modelzoo.models.gnn import fixed_shape_gpu as runner
from cerebras.modelzoo.models.gnn.data_processing.batches import GraphSAGEBatch
from cerebras.modelzoo.models.gnn.data_processing.padding import NeighborPaddingStats
from cerebras.modelzoo.models.gnn.data_processing.samplers import neighbor_tree
from cerebras.modelzoo.models.gnn.data_processing.sources.base import (
    BaseGraphDataSource,
)
from cerebras.modelzoo.models.gnn.model import GNNModel
from cerebras.modelzoo.models.gnn.tools import measure_fixed_shape, measure_window


def tiny_graph():
    # Node 2 is isolated. Three targets leave a padded tail with batch_size=2.
    return Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([0, 1, 0]),
        train_mask=torch.ones(3, dtype=torch.bool),
        val_mask=torch.ones(3, dtype=torch.bool),
        test_mask=torch.ones(3, dtype=torch.bool),
    )


def tiny_config():
    loader = dict(
        data_processor="GNNDataProcessor",
        dataset_name="ogbn-arxiv",
        sampling_mode="neighbor",
        fanouts=[2, 2],
        batch_size=2,
        num_workers=0,
        shuffle=False,
        sampler_seed=42,
        cache_fraction=1.0,
    )
    return {
        "trainer": {
            "init": {
                "seed": 42,
                "model": {
                    "name": "gnn",
                    "architecture": {
                        "type": "graphsage",
                        "n_feat": 2,
                        "n_class": 2,
                        "hidden_dim": 4,
                        "num_layers": 2,
                        "dropout": 0.0,
                        "aggregator": "mean",
                    },
                    "task": {"compute_eval_metrics": True},
                },
                "optimizer": {"AdamW": {"learning_rate": 0.01, "weight_decay": 0.0005}},
                "loop": {"max_steps": 5, "eval_frequency": 2},
                "logging": {"log_steps": 3},
            },
            "fit": {
                "train_dataloader": {**loader, "split": "train"},
                "val_dataloader": {**loader, "split": "valid"},
            },
        }
    }


class FixedShapeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.graph_patch = patch.object(
            BaseGraphDataSource, "load_graph", side_effect=tiny_graph
        )
        self.graph_patch.start()
        self.addCleanup(self.graph_patch.stop)

    def test_native_payload_matches_modelzoo_with_workers_and_host_cache(self):
        for workers in (0, 1):
            with self.subTest(workers=workers):
                config = tiny_config()["trainer"]["fit"]["train_dataloader"]
                config["num_workers"] = workers
                # The native path must never consult or wrap a Cerebras backend.
                with patch.object(
                    neighbor_tree.cstorch, "backend", side_effect=AssertionError
                ):
                    native = runner.make_loader(
                        config, num_layers=2, float_dtype=torch.float32
                    )
                processor = neighbor_tree.NeighborSamplingDataProcessor(
                    dataset_name="ogbn-arxiv",
                    data_dir=".",
                    current_split="train",
                    float_dtype=torch.float32,
                    label_dtype=torch.long,
                    adj_normalization_fn=None,
                    fanouts=[2, 2],
                    batch_size=2,
                    shuffle=False,
                    sampler_seed=42,
                    num_workers=workers,
                    pad_id=0,
                    cache_fraction=1.0,
                )
                with (
                    patch.object(neighbor_tree.cstorch, "use_cs", return_value=True),
                ):
                    modelzoo = processor.create_dataloader()
                batches = list(native)
                for actual, expected in zip(batches, modelzoo):
                    for key in actual:
                        a, b = actual[key], expected[key]
                        if torch.is_tensor(a):
                            a, b = [a], [b]
                        for x, y in zip(a, b):
                            self.assertEqual(x.device.type, "cpu")
                            torch.testing.assert_close(x, y, rtol=0, atol=0)
                self.assertEqual(
                    [b["node_features"][2].shape for b in batches],
                    [torch.Size([2, 4, 2])] * 2,
                )
                self.assertEqual(int(batches[-1]["target_mask"].sum()), 1)
                self.assertFalse(batches[-1]["neighbor_masks"][0].any())
                del native, modelzoo

    def test_padding_does_not_change_loss_or_gradients(self):
        loader = runner.make_loader(
            tiny_config()["trainer"]["fit"]["train_dataloader"],
            num_layers=2,
            float_dtype=torch.float32,
        )
        payload = list(loader)[-1]
        single = {
            key: value[:1] if torch.is_tensor(value) else [v[:1] for v in value]
            for key, value in payload.items()
        }
        cfg = tiny_config()["trainer"]["init"]["model"]
        cfg["task"]["compute_eval_metrics"] = False
        for aggregator in ("mean", "sum", "max"):
            cfg["architecture"]["aggregator"] = aggregator
            padded_model = GNNModel(cfg)
            single_model = copy.deepcopy(padded_model)
            padded_loss, single_loss = padded_model(payload), single_model(single)
            torch.testing.assert_close(padded_loss, single_loss)
            padded_loss.backward()
            single_loss.backward()
            for a, b in zip(padded_model.parameters(), single_model.parameters()):
                torch.testing.assert_close(a.grad, b.grad)

    def test_evaluation_excludes_ignored_labels_and_padding(self):
        cfg = tiny_config()
        cfg["trainer"]["init"]["model"]["task"]["compute_eval_metrics"] = False
        model = GNNModel(cfg["trainer"]["init"]["model"])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        for labels, expected in (
            ([0, -100, 0], {"accuracy": 1.0, "targets": 2}),
            ([-100, -100, -100], {"accuracy": 0.0, "targets": 0}),
        ):
            graph = tiny_graph()
            graph.y = torch.tensor(labels)
            with (
                self.subTest(labels=labels),
                patch.object(BaseGraphDataSource, "load_graph", return_value=graph),
            ):
                loader = runner.make_loader(
                    cfg["trainer"]["fit"]["val_dataloader"],
                    num_layers=2,
                    float_dtype=torch.float32,
                )
                actual = runner.evaluate(
                    model, loader, torch.device("cpu"), torch.float32
                )
                self.assertEqual(actual, expected)

    def test_neighbor_padding_counts_slots_and_valid_parent_shortages(self):
        loader = runner.make_loader(
            tiny_config()["trainer"]["fit"]["train_dataloader"],
            num_layers=2,
            float_dtype=torch.float32,
        )
        full, tail = list(loader)
        # Natural feature zeros must not be counted as padding.
        for features in full["node_features"]:
            features.zero_()
        stats = NeighborPaddingStats()
        stats.update(full)
        result = stats.summary()
        self.assertEqual([r["padding_percent"] for r in result["by_hop"]], [50, 75])
        # Weight by slots: (2 + 6) / (4 + 8), not the mean of hop percentages.
        self.assertAlmostEqual(result["overall"]["padding_percent"], 100 * 8 / 12)
        self.assertEqual(result["overall"]["slots_from_padded_parents"], 4)
        self.assertEqual(result["overall"]["padded_slots_from_valid_parents"], 4)

        tail_stats = NeighborPaddingStats()
        tail_stats.update(tail)
        result = tail_stats.summary()
        self.assertEqual(result["overall"]["padding_percent"], 100)
        self.assertEqual(result["by_hop"][0]["padding_percent_from_valid_parents"], 100)
        self.assertIsNone(result["by_hop"][1]["padding_percent_from_valid_parents"])
        stats.merge(tail_stats)
        result = stats.summary()["overall"]
        self.assertEqual(result["slots"], 24)
        self.assertEqual(result["valid_slots"], 4)
        self.assertEqual(result["padded_slots"], 20)
        self.assertEqual(result["slots_from_padded_parents"], 14)
        self.assertEqual(result["padded_slots_from_valid_parents"], 6)
        # Cached/repeated batches count every consumed occurrence.
        stats.update(full)
        self.assertEqual(stats.summary()["overall"]["valid_slots"], 8)

    def run_training(self, device, dtype=torch.float32, compile_model=False):
        with tempfile.TemporaryDirectory() as output:
            result = runner.train(
                tiny_config(),
                output,
                device=torch.device(device),
                dtype=dtype,
                warmup_steps=1,
                compile_model=compile_model,
                measure_neighbor_padding=True,
            )
            self.assertEqual(result["steps"], 4)
            self.assertEqual(result["seed_nodes"], 6)
            self.assertEqual(result["nominal_slots"], 8)
            padding = result["neighbor_padding"]
            self.assertEqual(padding["overall"]["slots"], 48)
            self.assertEqual(padding["overall"]["valid_slots"], 8)
            self.assertEqual(padding["overall"]["padded_slots"], 40)
            self.assertEqual(padding["overall"]["slots_from_padded_parents"], 28)
            self.assertEqual(padding["overall"]["padded_slots_from_valid_parents"], 12)
            self.assertAlmostEqual(padding["overall"]["padding_percent"], 100 * 40 / 48)
            records = [
                json.loads(line)
                for line in (Path(output) / "metrics.jsonl").read_text().splitlines()
            ]
            evals = [r for r in records if r["event"] == "eval"]
            windows = [r for r in records if r["event"] == "train" and r["measured"]]
            self.assertEqual(
                sum(r["neighbor_padding"]["overall"]["valid_slots"] for r in windows),
                padding["overall"]["valid_slots"],
            )
            self.assertEqual([r["step"] for r in evals], [2, 4, 5])
            self.assertTrue(all(r["targets"] == 3 for r in evals))
            checkpoint = torch.load(
                Path(output) / "checkpoint.pt", weights_only=True, map_location="cpu"
            )
            self.assertEqual(checkpoint["step"], 5)
            self.assertTrue(
                all(torch.isfinite(t).all() for t in checkpoint["model"].values())
            )
            torch.manual_seed(42)
            initial = GNNModel(tiny_config()["trainer"]["init"]["model"])
            self.assertTrue(
                any(
                    not torch.equal(v, checkpoint["model"][k])
                    for k, v in initial.state_dict().items()
                )
            )

    def test_cpu_training(self):
        self.run_training("cpu")

    def test_input_measurement_and_exact_gpu_numerator(self):
        config = tiny_config()
        config["trainer"]["init"]["loop"]["eval_frequency"] = None
        graph = tiny_graph()
        graph.y[1] = -100
        with (
            tempfile.TemporaryDirectory() as output,
            patch.object(BaseGraphDataSource, "load_graph", return_value=graph),
        ):
            result = runner.train(
                config,
                output,
                device=torch.device("cpu"),
                dtype=torch.float32,
                warmup_steps=1,
                measure_input=True,
            )
            self.assertEqual(result["seed_nodes"], 6)
            self.assertEqual(result["supervised_targets"], 4)
            self.assertEqual(result["optimizer_steps"], 4)
            self.assertEqual(result["logical_payload_bytes"], 624)
            self.assertGreater(result["loader_wait_seconds"], 0)
            self.assertNotIn("cuda_copy_stream_seconds", result)
            measured = measure_fixed_shape.summarize(
                Path(output) / "metrics.jsonl", 1, 5
            )
            self.assertEqual(measured["seed_nodes"], 6)
            self.assertEqual(measured["supervised_targets"], 4)
            # Feed the real sampler's target schedule to the CSX parser, then
            # compare its numerator with the batches the native model consumed.
            loader = runner.make_loader(
                config["trainer"]["fit"]["train_dataloader"],
                num_layers=2,
                float_dtype=torch.float32,
            )
            contract = neighbor_tree.batch_accounting_contract(
                loader.dataset,
                dataset_name="ogbn-arxiv",
                split="train",
                drop_last=False,
                static_batch_cache_size=0,
            )
            csx_rows = [
                "GNN_INPUT_CONTRACT " + json.dumps(contract),
                "Starting train loop 1 of 1, from global step 1 to 5 (5 steps)",
            ]
            for step in range(1, 6):
                timestamp = (datetime(2026, 1, 1) + timedelta(seconds=step)).strftime(
                    "%Y-%m-%d %H:%M:%S,%f"
                )[:-3]
                csx_rows.append(
                    f"{timestamp} INFO | Train Device=CSX, Step={step}, Loss=1.0,"
                )
            csx_rows.append("Training completed successfully!")
            csx_log = Path(output) / "synthetic_csx.log"
            csx_log.write_text("\n".join(csx_rows))
            csx_counts = measure_window.summarize(csx_log, 1, 5)
            for count in ("seed_nodes", "supervised_targets", "nominal_slots"):
                self.assertEqual(csx_counts[count], measured[count])
            metadata = json.loads((Path(output) / "run_metadata.json").read_text())
            self.assertEqual(metadata["optimizer_defaults"]["eps"], 1e-6)
            self.assertTrue(metadata["gnn_source_sha256"])

    def test_neighbor_padding_disabled_by_default(self):
        with (
            tempfile.TemporaryDirectory() as output,
            patch.object(runner, "NeighborPaddingStats", side_effect=AssertionError),
        ):
            result = runner.train(
                tiny_config(),
                output,
                device=torch.device("cpu"),
                dtype=torch.float32,
                warmup_steps=1,
            )
            self.assertEqual(result["seed_nodes"], 6)
            records = [
                json.loads(line)
                for line in (Path(output) / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertFalse(records[0]["measure_neighbor_padding"])
            self.assertTrue(all("neighbor_padding" not in r for r in records))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training(self):
        self.run_training("cuda")
        self.run_training("cuda", torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_cuda_forward_backward_parity(self):
        config = tiny_config()["trainer"]["init"]["model"]
        config["task"]["compute_eval_metrics"] = False
        cpu = GNNModel(config)
        gpu = copy.deepcopy(cpu).cuda()
        loader = runner.make_loader(
            tiny_config()["trainer"]["fit"]["train_dataloader"],
            num_layers=2,
            float_dtype=torch.float32,
        )
        for payload in loader:
            batch = GraphSAGEBatch.from_payload(payload)
            torch.testing.assert_close(
                cpu.model(batch), gpu.model(batch.to("cuda")).cpu()
            )
            cpu_loss, gpu_loss = cpu(batch), gpu(batch.to("cuda"))
            torch.testing.assert_close(cpu_loss, gpu_loss.cpu())
            cpu_loss.backward()
            gpu_loss.backward()
        for a, b in zip(cpu.parameters(), gpu.parameters()):
            torch.testing.assert_close(a.grad, b.grad.cpu(), atol=1e-6, rtol=1e-5)

    def test_invalid_fanouts_and_architecture(self):
        config = tiny_config()["trainer"]["fit"]["train_dataloader"]
        for fanouts in ([2], [2, 0]):
            with self.assertRaisesRegex(ValueError, "fanouts"):
                runner.make_loader(
                    {**config, "fanouts": fanouts},
                    num_layers=2,
                    float_dtype=torch.float32,
                )
        with self.assertRaisesRegex(ValueError, "neighbor-sampled"):
            runner.make_loader(
                {**config, "sampling_mode": "full_graph"},
                num_layers=2,
                float_dtype=torch.float32,
            )


if __name__ == "__main__":
    unittest.main()
