"""Observe real loaders/processes without downloads, CSX, or privileged tools."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from cerebras.modelzoo.models.gnn.data_processing import processor as facade
from cerebras.modelzoo.models.gnn.data_processing import worker_diagnostics as diag
from cerebras.modelzoo.models.gnn.data_processing.samplers import neighbor_tree
from cerebras.modelzoo.models.gnn.data_processing.worker_diagnostics_config import (
    WorkerDiagnosticsConfig,
)


def graph():
    return (
        torch.arange(24, dtype=torch.float32).view(12, 2),
        torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]),
        torch.arange(12) % 2,
        {"train": torch.ones(12, dtype=torch.bool)},
    )


def make_processor(directory, workers, enabled, **settings):
    with (
        patch.object(facade.cstorch, "use_cs", return_value=False),
        patch.object(
            facade.cstorch.amp, "get_floating_point_dtype", return_value=torch.float32
        ),
    ):
        return facade.GNNDataProcessor(
            dict(
                data_processor="GNNDataProcessor",
                dataset_name="ogbn-arxiv",
                data_dir=directory,
                sampling_mode="neighbor",
                fanouts=[2, 2],
                batch_size=2,
                split="train",
                num_workers=workers,
                persistent_workers=True,
                prefetch_factor=2,
                worker_diagnostics={
                    "enabled": enabled,
                    "output_dir": directory,
                    "max_batches": 3,
                    "max_snapshots": 2,
                    **settings,
                },
            )
        )


def make_loader(directory, workers, enabled, **settings):
    processor = make_processor(directory, workers, enabled, **settings)
    with patch.object(
        processor._processor, "prepare_graph_components", return_value=graph()
    ):
        return processor._processor.create_torch_dataloader()


def records(directory):
    return [
        json.loads(line)
        for path in Path(directory).glob("loader-*/*.jsonl")
        for line in path.read_text().splitlines()
    ]


def shutdown(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()


class WorkerDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = self.tmp.name
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)

    def test_disabled_has_original_classes_and_no_diagnostic_calls(self):
        calls = []

        def profile(frame, event, arg):
            if event == "call" and frame.f_code.co_filename == diag.__file__:
                calls.append(frame.f_code.co_name)

        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            for workers in (0, 2):
                loader = make_loader(self.directory, workers, False)
                self.assertIs(type(loader), DataLoader)
                self.assertIs(
                    type(loader.dataset), neighbor_tree.GraphSAGENeighborSamplerDataset
                )
                self.assertIsNone(loader.worker_init_fn)
                list(loader)
                list(loader)
                shutdown(loader)
        finally:
            sys.setprofile(previous)
        self.assertEqual(calls, [])
        self.assertEqual(records(self.directory), [])

    def test_real_workers_payloads_persistence_and_bounded_records(self):
        baseline = make_loader(self.directory, 0, False)
        expected = list(baseline)
        loader = make_loader(self.directory, 2, True)
        self.addCleanup(shutdown, loader)
        for _ in range(2):
            actual = list(loader)
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assertEqual(a.keys(), b.keys())
                for key in a:
                    x, y = a[key], b[key]
                    if not isinstance(x, list):
                        x, y = [x], [y]
                    for left, right in zip(x, y):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
        events = records(self.directory)
        initialized = [e for e in events if e["event"] == "worker_initialized"]
        self.assertEqual({e["worker_id"] for e in initialized}, {0, 1})
        self.assertEqual(len(initialized), 2)
        self.assertTrue(
            all(
                e["pid"] != os.getpid() and e["ppid"] == os.getpid()
                for e in initialized
            )
        )
        started = [e for e in events if e["event"] == "iterator_started"]
        self.assertEqual(
            set(started[0]["worker_pids"]), {e["pid"] for e in initialized}
        )
        self.assertTrue(all(e["settings"]["num_workers"] == 2 for e in started))
        received = [e for e in events if e["event"] == "batch_received"]
        self.assertEqual(len(received), 3)
        self.assertEqual([e["iteration_batch_index"] for e in received], [0, 1, 2])
        generated = [e for e in events if e["event"] == "batch_generated"]
        self.assertEqual(len(generated), 6)
        self.assertEqual({e["worker_id"] for e in generated}, {0, 1})
        for event in generated:
            self.assertEqual(set(event["phases"]), {"sampling", "gather"})
            self.assertTrue(all(p["wall_ns"] > 0 for p in event["phases"].values()))
        for event in generated + received:
            self.assertGreater(event["wall_ns"], 0)
            self.assertGreaterEqual(event["process_cpu_ns"], 0)
            self.assertEqual(event["wall_ns"], event["end_ns"] - event["start_ns"])
        snapshots = [e for e in events if e["event"] == "snapshot"]
        self.assertGreaterEqual(len(snapshots), 1)
        self.assertLessEqual(len(snapshots), 2)
        parent = next(p for p in snapshots[0]["processes"] if p["pid"] == os.getpid())
        self.assertIn("user_ticks", parent["stat"])
        self.assertTrue(parent["threads"])
        created = next(e for e in events if e["event"] == "loader_created")
        self.assertEqual(len(created["sources"]["dataset"]["sha256"]), 64)
        self.assertEqual(created["settings"]["num_workers"], 2)

    def test_spawn_workers_and_unique_factory_directories(self):
        loader = make_loader(self.directory, 1, True, max_snapshots=0)
        loader.multiprocessing_context = "spawn"
        self.addCleanup(shutdown, loader)
        self.assertEqual(len(list(loader)), 6)
        initialized = [
            e for e in records(self.directory) if e["event"] == "worker_initialized"
        ]
        self.assertEqual(len(initialized), 1)
        self.assertEqual(initialized[0]["worker_id"], 0)
        other = make_loader(self.directory, 0, True)
        self.assertNotEqual(loader.recorder.directory, other.recorder.directory)

    def test_cached_dataset_and_zero_workers_are_observed(self):
        processor = make_processor(self.directory, 0, True, max_snapshots=0)
        processor._processor.static_batch_cache_size = 1
        with patch.object(
            processor._processor, "prepare_graph_components", return_value=graph()
        ):
            loader = processor._processor.create_torch_dataloader()
        batches = list(loader)
        for batch in batches[1:]:
            self.assertTrue(
                torch.equal(batch["node_features"][0], batches[0]["node_features"][0])
            )
        events = records(self.directory)
        self.assertFalse(any(e["event"] == "worker_initialized" for e in events))
        generated = [e for e in events if e["event"] == "batch_generated"]
        self.assertEqual(len(generated), 3)
        self.assertTrue(all(e["worker_id"] is None for e in generated))
        self.assertTrue(all(e["phases"] == {} for e in generated))

    def test_io_failure_does_not_change_batches(self):
        blocked = Path(self.directory) / "file"
        blocked.write_text("not a directory")
        with self.assertLogs(diag.logger, level="WARNING") as logs:
            loader = make_loader(str(blocked), 0, True)
            self.assertEqual(len(list(loader)), 6)
        self.assertEqual(len(logs.output), 1)

    def test_short_epochs_do_not_create_unbounded_metadata(self):
        loader = make_loader(
            self.directory,
            0,
            True,
            max_batches=1,
            max_snapshots=2,
            snapshot_interval_seconds=3600,
        )
        for _ in range(20):
            list(loader)
        events = records(self.directory)
        self.assertEqual(sum(e["event"] == "iterator_started" for e in events), 2)
        self.assertEqual(sum(e["event"] == "batch_generated" for e in events), 1)
        self.assertEqual(sum(e["event"] == "batch_received" for e in events), 1)

    def test_native_training_enabled_matches_disabled(self):
        from test_fixed_shape_gpu import tiny_config, tiny_graph
        from cerebras.modelzoo.models.gnn import fixed_shape_gpu as runner
        from cerebras.modelzoo.models.gnn.data_processing.sources.base import (
            BaseGraphDataSource,
        )

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            losses = []
            for enabled in (False, True):
                config = tiny_config()
                config["trainer"]["fit"]["train_dataloader"]["worker_diagnostics"] = {
                    "enabled": enabled,
                    "output_dir": self.directory,
                    "max_batches": 3,
                    "max_snapshots": 1,
                }
                output = Path(self.directory) / f"{device}-{enabled}"
                with (
                    patch.object(
                        BaseGraphDataSource, "load_graph", side_effect=tiny_graph
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    runner.train(
                        config,
                        output,
                        device=torch.device(device),
                        dtype=torch.float32,
                        warmup_steps=1,
                    )
                metrics = [
                    json.loads(line)
                    for line in (output / "metrics.jsonl").read_text().splitlines()
                ]
                losses.append([r["loss"] for r in metrics if r["event"] == "train"])
            torch.testing.assert_close(
                torch.tensor(losses[0]), torch.tensor(losses[1]), rtol=0, atol=0
            )
        self.assertEqual(
            sum(e["event"] == "loader_created" for e in records(self.directory)),
            len(devices),
        )

    def test_invalid_settings(self):
        for settings in (
            {"enabled": True},
            {"max_batches": -1},
            {"max_snapshots": -1},
            {"snapshot_interval_seconds": 0},
            {"snapshot_interval_seconds": float("nan")},
            {"typo": True},
        ):
            with self.assertRaises(ValueError):
                WorkerDiagnosticsConfig(**settings)
        with self.assertRaisesRegex(ValueError, "requires neighbor"):
            facade.GNNDataProcessorConfig(
                data_processor="GNNDataProcessor",
                dataset_name="ogbn-arxiv",
                worker_diagnostics={"enabled": True, "output_dir": self.directory},
            )


class CgroupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proc = self.root / "proc"
        (self.proc / "self").mkdir(parents=True)

    def write_mount(self, member, mount_root, point, kind="cgroup2", controllers=""):
        (self.proc / "self/cgroup").write_text(f"0:{controllers}:{member}\n")
        escaped = str(point).replace(" ", "\\040")
        (self.proc / "self/mountinfo").write_text(
            f"1 0 0:1 {mount_root} {escaped} rw - {kind} cgroup rw,{controllers}\n"
        )

    def test_v2_nested_quota_cpuset_and_missing_files(self):
        mount = self.root / "cgroup mount"
        worker = mount / "job/worker"
        worker.mkdir(parents=True)
        (worker / "cpu.max").write_text("max 100000")
        (worker.parent / "cpu.max").write_text("200000 100000")
        (worker / "cpu.stat").write_text(
            "nr_periods 10\nnr_throttled 2\nthrottled_usec 123"
        )
        (worker / "cpuset.cpus.effective").write_text("0-7")
        self.write_mount("/job/worker", "/", mount)
        result = diag.cgroup_snapshot(self.proc)
        self.assertFalse(result["hidden_ancestors_checked"])
        entries = result["hierarchies"][0]["ancestors"]
        self.assertEqual(len(entries), 3)
        self.assertIsNone(entries[0]["quota_cores"])
        self.assertEqual(entries[1]["quota_cores"], 2)
        self.assertNotIn("quota_cores", entries[2])
        self.assertEqual(entries[0]["files"]["cpuset.cpus.effective"]["value"], "0-7")
        self.assertEqual(entries[2]["files"]["cpu.max"]["error"], "FileNotFoundError")

    def test_v1_cpu_controller(self):
        mount = self.root / "cpu"
        worker = mount / "worker"
        worker.mkdir(parents=True)
        (worker / "cpu.cfs_quota_us").write_text("150000")
        (worker / "cpu.cfs_period_us").write_text("100000")
        self.write_mount(
            "/worker", "/", mount, kind="cgroup", controllers="cpu,cpuacct"
        )
        entries = diag.cgroup_snapshot(self.proc)["hierarchies"][0]
        self.assertEqual(entries["version"], 1)
        self.assertEqual(entries["ancestors"][0]["quota_cores"], 1.5)

    def test_namespace_root_and_escape_are_explicit(self):
        mount = self.root / "cgroup"
        mount.mkdir()
        self.write_mount("/", "/host/job", mount)
        result = diag.cgroup_snapshot(self.proc)
        self.assertEqual(len(result["hierarchies"][0]["ancestors"]), 1)
        self.write_mount("/../../outside", "/", mount)
        result = diag.cgroup_snapshot(self.proc)
        self.assertEqual(result["hierarchies"], [])
        self.assertTrue(result["unresolved_memberships"])


if __name__ == "__main__":
    unittest.main()
