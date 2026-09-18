"""Native-loop measurement checks with CPU tensors and a simulated CUDA clock."""

from contextlib import ExitStack, nullcontext, redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import yaml


PATH = Path(__file__).resolve().parents[1] / "gpu/train.py"
SPEC = importlib.util.spec_from_file_location("non_gnn_gpu_train", PATH)
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


class GPUInputMeasurementTests(unittest.TestCase):
    def test_bert_input_positions_and_loss_targets_are_distinct(self):
        window = [
            {
                "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
                "masked_lm_mask": torch.tensor([[1, 0], [1, 0]]),
            },
            {
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                "masked_lm_mask": torch.tensor([[1, 1]]),
            },
        ]
        self.assertEqual(train.token_counts(window, "bert"), (9, 4))

    def test_llama_reuses_loss_mask_with_unequal_microbatches(self):
        window = [
            {"attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])},
            {"attention_mask": torch.tensor([[1, 0, 0, 0]])},
        ]
        self.assertEqual(train.token_counts(window, "llama"), (7, 7))

    def run_loop(self, root, model_name="bert", warmup=1, nonfinite=None):
        config = {
            "trainer": {
                "init": {
                    "seed": 42,
                    "model": {"name": model_name},
                    "optimizer": {"AdamW": {"weight_decay": 0.01, "lr": 0.01}},
                    "loop": {"grad_accum_steps": 2, "max_steps": 3},
                },
                "fit": {"train_dataloader": {}},
            }
        }
        config_path = root / "params.yaml"
        config_path.write_text(yaml.safe_dump(config))
        launch = {
            "params_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "gpu_implementation": "native",
            "gradient_checkpointing": False,
            "compile": False,
            "effective_batch_size": 2,
            "sequence_length": 4,
            "warmup_steps": warmup,
        }
        (root / "launch.json").write_text(json.dumps(launch))
        batches = []
        for inputs, targets in zip([4, 2, 3, 1, 2, 1], [1, 1, 2, 0, 1, 0]):
            batches.append(
                {
                    "input_ids": torch.ones((1, 4), dtype=torch.int64),
                    "attention_mask": torch.tensor([[1] * inputs + [0] * (4 - inputs)]),
                    "masked_lm_mask": torch.tensor(
                        [[1] * targets + [0] * (2 - targets)]
                    ),
                }
            )

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))
                self.calls = 0

            def forward(self, batch):
                self.calls += 1
                loss = self.scale * batch["attention_mask"].sum()
                if self.calls == 3 and nonfinite:
                    # A nonfinite scalar with finite derivatives can reach the
                    # log; nonfinite gradients retain the existing hard error.
                    loss = (
                        loss + float("nan")
                        if nonfinite == "loss"
                        else loss * float("nan")
                    )
                return loss

        model = TinyModel()
        optimizer = Mock(zero_grad=model.zero_grad)
        cpu = torch.device("cpu")
        clocks = [10.0, 13.0, 14.0, 17.0, 22.0]
        if warmup == 0:
            clocks = [10.0, 10.5, 13.0, 17.0, 22.0]
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(sys, "argv", [str(PATH), "--config", str(config_path)])
            )
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "models": SimpleNamespace(make_model=lambda cfg: model),
                        "transformers": SimpleNamespace(__version__="test"),
                    },
                )
            )
            stack.enter_context(
                patch.object(train, "create_loader", return_value=batches)
            )
            stack.enter_context(
                patch.object(train.time, "perf_counter", side_effect=clocks)
            )
            stack.enter_context(patch.object(torch, "device", return_value=cpu))
            stack.enter_context(
                patch.object(torch, "autocast", return_value=nullcontext())
            )
            stack.enter_context(
                patch.object(torch.optim, "AdamW", return_value=optimizer)
            )
            for name, result in {
                "is_available": True,
                "is_bf16_supported": True,
                "manual_seed_all": None,
                "synchronize": None,
                "get_device_name": "Test GPU",
                "get_device_properties": SimpleNamespace(
                    total_memory=8192, major=9, minor=0
                ),
                "max_memory_allocated": 2048,
                "max_memory_reserved": 4096,
            }.items():
                stack.enter_context(patch.object(torch.cuda, name, return_value=result))
            reset = stack.enter_context(
                patch.object(torch.cuda, "reset_peak_memory_stats")
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            train.main()
        reset.assert_called_once_with()
        self.assertEqual(optimizer.step.call_count, 3)
        return (
            [
                json.loads(line)
                for line in (root / "metrics.jsonl").read_text().splitlines()
            ],
            json.loads((root / "gpu_environment.json").read_text()),
        )

    def test_window_clock_counts_and_allocator_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            events, environment = self.run_loop(Path(directory))
        steps, summary = events[:-1], events[-1]
        self.assertEqual([step["valid_tokens"] for step in steps], [6, 10, 13])
        self.assertEqual([step["loss_target_tokens"] for step in steps], [2, 4, 5])
        self.assertEqual([step["update_valid_tokens"] for step in steps], [6, 4, 3])
        self.assertEqual(
            [step["update_loss_target_tokens"] for step in steps], [2, 2, 1]
        )
        self.assertEqual([step["nsp_examples"] for step in steps], [2, 4, 6])
        self.assertEqual([step["update_nsp_examples"] for step in steps], [2, 2, 2])
        self.assertEqual([step["loss"] for step in steps], [3.0, 2.0, 1.5])
        self.assertEqual(
            (summary["window_start_step"], summary["window_end_step"]), (1, 3)
        )
        self.assertEqual(summary["measurement_start_elapsed_seconds"], 4.0)
        self.assertGreater(
            summary["measurement_start_elapsed_seconds"], steps[0]["elapsed_seconds"]
        )
        self.assertEqual(
            summary["measurement_end_elapsed_seconds"], steps[-1]["elapsed_seconds"]
        )
        self.assertEqual(summary["measurement_window_seconds"], 8.0)
        self.assertEqual(summary["elapsed_seconds"], 8.0)
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["nominal_tokens"], 16)
        self.assertEqual(summary["measured_valid_tokens"], 7)
        self.assertEqual(summary["measured_loss_target_tokens"], 3)
        self.assertEqual(summary["measured_nsp_examples"], 4)
        self.assertEqual(summary["samples_per_second"], 0.5)
        self.assertEqual(summary["nominal_tokens_per_second"], 2.0)
        self.assertEqual(summary["valid_tokens_per_second"], 7 / 8)
        self.assertEqual(summary["loss_target_tokens_per_second"], 3 / 8)
        self.assertEqual(summary["peak_memory_scope"], "measurement_window")
        self.assertEqual(summary["peak_allocated_bytes"], 2048)
        self.assertEqual(summary["peak_reserved_bytes"], 4096)
        self.assertEqual(environment["device_total_memory_bytes"], 8192)
        self.assertEqual(environment["device_compute_capability"], [9, 0])
        self.assertEqual(environment["parameter_dtypes"], ["torch.float32"])
        self.assertEqual(environment["compute_dtype"], "bfloat16")
        for entry in [environment, summary]:
            self.assertEqual(
                entry["valid_tokens_definition"], "nonpadding_input_positions"
            )
            self.assertEqual(
                entry["loss_target_tokens_definition"],
                "masked_language_model_positions",
            )
            self.assertEqual(
                entry["nsp_examples_definition"], "next_sentence_prediction_examples"
            )

    def test_zero_warmup_uses_actual_synchronized_start_and_all_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            events, environment = self.run_loop(
                Path(directory), model_name="llama", warmup=0
            )
        summary = events[-1]
        self.assertEqual(summary["window_start_step"], 0)
        self.assertEqual(summary["measurement_start_elapsed_seconds"], 0.5)
        self.assertEqual(summary["measurement_window_seconds"], 11.5)
        self.assertEqual(summary["measured_optimizer_steps"], 3)
        self.assertEqual(summary["measured_loss_target_tokens"], 13)
        self.assertEqual(summary["measured_valid_tokens"], 13)
        self.assertNotIn("measured_nsp_examples", summary)
        for event in events[:-1]:
            self.assertAlmostEqual(event["loss"], 1.0)
            self.assertFalse(event["warmup"])
        for entry in [environment, summary]:
            self.assertEqual(entry["valid_tokens_definition"], "loss_target_positions")
            self.assertEqual(
                entry["loss_target_tokens_definition"],
                "causal_language_model_positions",
            )

    def test_nonfinite_loss_is_reported_without_changing_training_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            events, _ = self.run_loop(Path(directory), nonfinite="loss")
        self.assertTrue(math.isnan(events[1]["loss"]))
        self.assertFalse(events[1]["loss_is_finite"])
        summary = events[-1]
        self.assertEqual(summary["loss_nonfinite_steps"], 1)
        self.assertEqual(summary["measurement_loss_nonfinite_steps"], 1)
        self.assertFalse(summary["loss_is_finite"])
        self.assertFalse(summary["measurement_loss_is_finite"])

    def test_nonfinite_gradient_still_fails_and_leaves_no_success_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                self.run_loop(root, nonfinite="gradient")
            events = [
                json.loads(line)
                for line in (root / "metrics.jsonl").read_text().splitlines()
            ]
        self.assertEqual([event["event"] for event in events], ["train"])


if __name__ == "__main__":
    unittest.main()
