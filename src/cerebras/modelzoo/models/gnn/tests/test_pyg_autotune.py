"""Shared tuner/PyG tests; CUDA timing is mocked only in the host loop test."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data
import yaml

from cerebras.modelzoo.models.gnn.reference.pyg.data import make_loaders
from cerebras.modelzoo.models.gnn.reference.pyg.train import train_model
from cerebras.modelzoo.models.gnn.tools import autotune as tune
from cerebras.modelzoo.models.gnn.tools import measure_pyg


def write_log(path, end=440, seconds=1.0, unstable=False):
    lines = []
    for step in range(10, end + 1, 10):
        # Ten batches per block, with one short tail: 9*4 + 1*2 actual seeds.
        row = {
            "version": 1,
            "step": step,
            "wall_seconds": step * seconds,
            "seed_nodes": (step // 10) * 38,
            "loss": 0.5,
            "all_losses_finite": True,
        }
        if unstable:
            row["wall_seconds"] += max(0, step - 240)
        lines.append(measure_pyg.PREFIX + json.dumps(row))
    lines.append(f"Training Completed. Total Steps: {end} (Active: {end - 40})")
    path.write_text("\n".join(lines))


class PyGTunerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.args = tune.parse_args(
            [
                "--backend",
                "pyg",
                "--dataset",
                "arxiv",
                "--workers",
                "0",
                "1",
                "--output",
                str(self.output),
                "--budget-sec",
                "100000",
            ]
        )
        self.backend = tune.get_backend("pyg")
        self.base = tune.load_params_file(tune.GNN / "configs/autotune/arxiv_w40.yaml")

    def study(self, args=None):
        return tune.Study(
            args or self.args, self.base, self.output, {"test": "PyG host fixture"}
        )

    def fake_execute(self, cmd, log, timeout):
        self.assertEqual(cmd[4:6], ["python", "-u"])
        self.assertTrue(cmd[6].endswith("pyg_graphsage.py"))
        params = yaml.safe_load(Path(cmd[8]).read_text())
        self.assertNotIn("backend", params["trainer"]["init"])
        workers = params["trainer"]["fit"]["train_dataloader"]["num_workers"]
        write_log(
            log,
            end=params["trainer"]["init"]["loop"]["max_steps"],
            seconds=1 if workers else 2,
        )
        return {"status": "completed", "returncode": 0}

    def test_actual_seed_measurement_and_bad_logs(self):
        path = self.output / "train.log"
        write_log(path)
        result = measure_pyg.summarize(path, 40, 440)
        self.assertEqual(result["count"], 1520)
        self.assertEqual(result["throughput"], 3.8)
        self.assertEqual(result["metric"], "seed_nodes_per_second")
        self.assertTrue(result["half_window_check"]["within_tolerance"])
        original = path.read_text()
        invalid = [
            original + "\n" + original,
            original.replace('"step": 240', '"step": 241'),
            original.replace('"all_losses_finite": true', '"all_losses_finite": false'),
            original.replace('"wall_seconds": 40.0', '"wall_seconds": -1'),
            original.replace('"seed_nodes": 152', '"seed_nodes": 0'),
            original.replace("Training Completed.", "Aborted."),
            original + "\n[Eval] Step=440",
            "[Step=0040] Wall=1.0s | Loss=0.5\n[Step=0440] Wall=2.0s | Loss=0.5\nTraining Completed. Total Steps: 440",
        ]
        for content in invalid:
            path.write_text(content)
            with self.assertRaises(ValueError):
                measure_pyg.summarize(path, 40, 440)
        write_log(path, unstable=True)
        self.assertFalse(
            measure_pyg.summarize(path, 40, 440)["half_window_check"][
                "within_tolerance"
            ]
        )

    def test_shared_selection_resume_backend_identity(self):
        self.assertEqual((self.args.measure_steps, self.args.confirm_steps), (400, 800))
        with patch.object(tune, "execute", side_effect=self.fake_execute) as execute:
            study = self.study()
            self.assertEqual(study.run(), 0)
            self.assertEqual(execute.call_count, 8)
            self.assertEqual(study.state["ranking"][0]["knobs"]["num_workers"], 1)
            self.assertEqual(
                study.state["ranking"][0]["metric"], "seed_nodes_per_second"
            )
            self.assertEqual(self.study().run(), 0)
            self.assertEqual(execute.call_count, 8)
        best = yaml.safe_load((self.output / "best.yaml").read_text())
        self.assertEqual(best["trainer"]["init"]["loop"]["max_steps"], 840)
        self.assertEqual(
            best["trainer"]["fit"]["train_dataloader"]["cache_fraction"], 0.0
        )
        other = deepcopy(self.args)
        other.backend = "csx"
        with self.assertRaisesRegex(ValueError, "changed"):
            self.study(other)

    def test_budget_and_local_failure(self):
        study = self.study()
        study.state["used_sec"] = 95000
        study.save()
        with patch.object(tune, "execute") as execute:
            self.assertEqual(study.run(), 2)
            execute.assert_not_called()
        args = deepcopy(self.args)
        args.budget_sec = 110000
        calls = 0

        def run(cmd, log, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "status": "failed",
                    "returncode": 1,
                    "failure_reason": "CUDA out of memory",
                }
            return self.fake_execute(cmd, log, timeout)

        with patch.object(tune, "execute", side_effect=run):
            study = self.study(args)
            self.assertEqual(study.run(), 0)
            self.assertEqual(calls, 5)
            self.assertEqual(
                study.state["trials"][0]["failure_reason"], "CUDA out of memory"
            )

    def test_lost_client_requires_ack_and_charges_budget(self):
        study = self.study()
        study.state["trials"] = [{"status": "running", "trial_id": "lost"}]
        study.save()
        with self.assertRaisesRegex(ValueError, "Confirm its job/process"):
            self.study().run()
        args = deepcopy(self.args)
        args.acknowledge_stopped_jobs = True
        resumed = self.study(args)
        self.assertEqual(resumed.state["used_sec"], self.args.trial_timeout_sec)
        self.assertEqual(resumed.state["trials"][0]["status"], "interrupted")

    def test_train_only_real_neighbor_loader(self):
        cfg = self.backend.prepare_config(
            self.base, tune.candidate(0), self.output / "model", 60, 7200, 20
        )
        cfg["trainer"]["fit"]["train_dataloader"].update(
            batch_size=2, fanouts=[1], pin_memory=False
        )
        data = Data(
            x=torch.ones(3, 2),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            y=torch.tensor([0, 1, 0]),
        )
        for workers in (0, 1):
            with self.subTest(workers=workers):
                cfg["trainer"]["fit"]["train_dataloader"].update(
                    num_workers=workers, prefetch_factor=3, persistent_workers=True
                )
                train, val = make_loaders(data, {"train": torch.arange(3)}, cfg)
                self.assertIsNone(val)
                self.assertEqual(train.num_workers, workers)
                self.assertEqual(train.prefetch_factor, 3 if workers else None)
                self.assertEqual(train.persistent_workers, bool(workers))
                for _ in range(2):
                    batches = list(train)
                    self.assertEqual(sorted(b.batch_size for b in batches), [1, 2])
                    self.assertEqual(sum(b.batch_size for b in batches), 3)
                del train

    def test_import_without_sdk(self):
        code = """import sys, importlib.abc
class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("cerebras.pytorch", "cerebras.appliance")):
            raise RuntimeError("SDK import forbidden: " + fullname)
sys.meta_path.insert(0, NoSDK())
from cerebras.modelzoo.models.gnn.reference.pyg import runner
from cerebras.modelzoo.models.gnn.tools import autotune
print("imports succeeded without SDK")
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=45
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("imports succeeded without SDK", result.stdout)

    def run_tiny_training(self, device, mock_cuda):
        cfg = self.backend.prepare_config(
            self.base, tune.candidate(0), self.output, 60, 7200, 20
        )
        cfg["trainer"]["init"]["model"]["task"]["to_float16"] = False
        cfg["trainer"]["init"]["model_dir"] = str(self.output)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x, edge_index, batch_size=None):
                return self.linear(x)

        model = Model().to("cpu" if mock_cuda else device)
        batches = [
            Data(
                x=torch.ones(n, 2),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.zeros(n, dtype=torch.long),
                batch_size=n,
            )
            for n in (2, 1)
        ]
        stream = io.StringIO()
        with redirect_stdout(stream):
            train_model(cfg, model, (batches, None), None, None, device)
        path = self.output / "train.log"
        path.write_text(stream.getvalue())
        result = measure_pyg.summarize(path, 20, 60)
        self.assertEqual(result["count"], 60)
        self.assertFalse((self.output / "last.pt").exists())
        self.assertNotIn("[Eval]", stream.getvalue())

    def test_real_host_training_with_mock_cuda_timing(self):
        # Execute actual forward/backward/optimizer steps on CPU. This is not GPU evidence.
        zeros, ones = torch.zeros, torch.ones

        def cpu_factory(factory):
            def make(*args, **kwargs):
                kwargs["device"] = "cpu"
                return factory(*args, **kwargs)

            return make

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with (
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "Event") as event,
            patch.object(torch, "zeros", side_effect=cpu_factory(zeros)),
            patch.object(torch, "ones", side_effect=cpu_factory(ones)),
            patch.object(Data, "to", lambda self, *args, **kwargs: self),
        ):
            event.return_value.elapsed_time.return_value = 0.0
            self.run_tiny_training(torch.device("cuda"), mock_cuda=True)

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA GPU available")
    def test_cuda_pyg_runner_with_tiny_graph(self):
        config = self.backend.prepare_config(
            self.base, tune.candidate(0), self.output / "model", 60, 7200, 20
        )
        config["trainer"]["init"]["model"]["architecture"].update(
            n_feat=2, n_class=2, hidden_dim=8, num_layers=2
        )
        config["trainer"]["init"]["model"]["task"]["to_float16"] = False
        config["trainer"]["fit"]["train_dataloader"].update(
            batch_size=4, fanouts=[2, 2]
        )
        params = self.output / "params.yaml"
        params.write_text(yaml.safe_dump(config))
        code = """import sys, importlib.abc
class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("cerebras.pytorch", "cerebras.appliance")):
            raise RuntimeError("SDK import forbidden: " + fullname)
sys.meta_path.insert(0, NoSDK())
import torch
from torch_geometric.data import Data
from cerebras.modelzoo.models.gnn.reference.pyg import runner
torch.set_num_threads(1)
nodes = torch.arange(7)
edges = torch.stack([nodes, (nodes + 1) % 7])
edges = torch.cat([edges, edges.flip(0)], dim=1)
data = Data(x=torch.ones(7, 2), edge_index=edges, y=nodes % 2)
runner.load_dataset = lambda profile: (data, {"train": nodes})
sys.argv = ["pyg_graphsage.py", "--config", sys.argv[1]]
runner.main(expected_architecture="graphsage")
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(params)],
            cwd=tune.GNN,
            env={**os.environ, "NO_COMPILE": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = self.output / "runner.log"
        log.write_text(result.stdout)
        measured = measure_pyg.summarize(log, 20, 60)
        self.assertEqual(measured["count"], 140)
        self.assertIn("Caching disabled (0 nodes)", result.stdout)
        self.assertFalse((self.output / "model/last.pt").exists())

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA GPU available")
    def test_cuda_training_window(self):
        self.run_tiny_training(torch.device("cuda"), mock_cuda=False)


if __name__ == "__main__":
    unittest.main()
