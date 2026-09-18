"""Prevent inherited training controls from changing throughput accounting."""

from copy import deepcopy
from pathlib import Path
import unittest

from cerebras.modelzoo.common.utils.run.config_loader import load_params_file
from cerebras.modelzoo.models.gnn.tools.autotune_backends import CSXBackend, GNN


class BenchmarkConditionTests(unittest.TestCase):
    def setUp(self):
        self.base = load_params_file(GNN / "configs/autotune/arxiv_w00.yaml")

    def prepare(self):
        return CSXBackend.prepare_config(self.base, {}, Path("/tmp/fixture"), 240, 7200)

    def test_fresh_run_overrides_restore_and_accumulation_without_mutating_base(self):
        init = self.base["trainer"]["init"]
        init["autorestart"] = {"max_num_restarts": 3}
        init["loop"]["grad_accum_steps"] = 4
        init["checkpoint"]["autoload_last_checkpoint"] = True
        self.base["trainer"]["fit"]["ckpt_path"] = "/tmp/old-checkpoint.mdl"
        original = deepcopy(self.base)
        prepared = self.prepare()["trainer"]
        self.assertEqual(prepared["init"]["autorestart"]["max_num_restarts"], 0)
        self.assertEqual(prepared["init"]["loop"]["grad_accum_steps"], 1)
        self.assertFalse(prepared["init"]["checkpoint"]["autoload_last_checkpoint"])
        self.assertIsNone(prepared["fit"]["ckpt_path"])
        self.assertIsNone(prepared["fit"]["val_dataloader"])
        self.assertEqual(self.base, original)

    def test_normal_observers_and_microbatch_control_remain_available(self):
        callbacks = [
            {"CheckLoss": {}},
            {"RateProfiler": {}},
            {"ComputeNorm": {}},
            {"GlobalFlags": {"csx.performance.micro_batch_size": None}},
        ]
        self.base["trainer"]["init"]["callbacks"] = callbacks
        self.assertEqual(self.prepare()["trainer"]["init"]["callbacks"], callbacks)

    def test_stream_changes_and_debug_substitutions_require_separate_audit(self):
        for callback in (
            {"SkipSamples": {"n": 4096}},
            {"DebugArgsPath": {"debug_args_path": "/tmp/debug.ini"}},
            {"GlobalFlags": {"csx.debug.debug_args": {}}},
            {"UserCallback": {}},
        ):
            with self.subTest(callback=callback):
                self.base["trainer"]["init"]["callbacks"] = [callback]
                with self.assertRaises(ValueError):
                    self.prepare()
        self.base["trainer"]["init"]["callbacks"] = []
        loader = self.base["trainer"]["fit"]["train_dataloader"]
        for key, value in (("split", "valid"), ("sampling_mode", "full_graph")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                saved = loader[key]
                loader[key] = value
                try:
                    self.prepare()
                finally:
                    loader[key] = saved


if __name__ == "__main__":
    unittest.main()
