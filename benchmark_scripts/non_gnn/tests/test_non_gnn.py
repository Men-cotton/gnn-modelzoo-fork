"""CPU semantic checks and scheduler-boundary tests; never submit real jobs."""

import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmark_scripts/non_gnn"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load_module("non_gnn_run", BENCH / "run.py")
models = load_module("non_gnn_models", BENCH / "gpu/models.py")
train = load_module("non_gnn_train", BENCH / "gpu/train.py")


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test-non-gnn-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def args(self, profile="bert_large_msl128", backend="GPU", *extra):
        return run.parser().parse_args(
            [
                "--backend",
                backend,
                "--profile",
                profile,
                "--data-dir",
                str(self.path),
                "--output-dir",
                str(self.path / "output"),
                "--num-workers",
                "0",
                *extra,
            ]
        )

    def test_all_profile_pairings_and_schema(self):
        from cerebras.modelzoo.trainer.validate import validate_trainer_params

        for profile in yaml.safe_load(run.PROFILES.read_text()):
            pair = []
            for backend in ("CSX", "GPU"):
                with self.subTest(profile=profile, backend=backend):
                    params, meta = run.build_config(self.args(profile, backend))
                    validate_trainer_params(copy.deepcopy(params))
                    init = params["trainer"]["init"]
                    self.assertEqual(meta["nominal_tokens_per_update"], 32768)
                    self.assertEqual(
                        meta["loader_batch_size"] * meta["grad_accum_steps"],
                        meta["effective_batch_size"],
                    )
                    self.assertEqual(init["precision"]["fp16_type"], "bfloat16")
                    pair.append(init)
            self.assertEqual(pair[0]["model"], pair[1]["model"])
            self.assertEqual(pair[0]["optimizer"], pair[1]["optimizer"])

    def test_invalid_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, "divide"):
            run.build_config(
                self.args("llama3p2_1b_msl2048", "GPU", "--gpu-micro-batch-size", "3")
            )

    def test_llama_shapes_and_native_loader(self):
        args = self.args("llama3p2_1b_msl1024")
        params, meta = run.build_config(args)
        with h5py.File(self.path / "sample.h5", "w") as handle:
            data = handle.create_dataset("data", shape=(64, 3, 1024), dtype="i4")
            data[:, 0, :] = 7
            data[:, 1, :] = 1
            data[:, 2, :] = 11
        run.check_data(args, meta)
        batch = next(
            iter(train.create_loader(params["trainer"]["fit"]["train_dataloader"]))
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 1024))
        self.assertTrue(torch.all(batch["labels"] == 11))
        other = self.args("llama3p2_1b_msl2048")
        _, wrong = run.build_config(other)
        with self.assertRaisesRegex(ValueError, "expected"):
            run.check_data(other, wrong)

    def test_llama_corpus_shift(self):
        args = self.args("llama3p2_1b_msl1024", "GPU", "--llama-data-format", "corpus")
        params, meta = run.build_config(args)
        with h5py.File(self.path / "sample.h5", "w") as handle:
            handle.create_dataset("data", data=list(range(4097)), dtype="i4")
        run.check_data(args, meta)
        batch = next(
            iter(train.create_loader(params["trainer"]["fit"]["train_dataloader"]))
        )
        torch.testing.assert_close(batch["input_ids"] + 1, batch["labels"])

    def test_bert_native_loader_and_model_backward(self):
        with (self.path / "sample.csv").open("w") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["tokens", "segment_ids", "is_random_next"]
            )
            writer.writeheader()
            for _ in range(32):
                writer.writerow(
                    {
                        "tokens": repr(["[CLS]", "hello", "world", "[SEP]"]),
                        "segment_ids": repr([0, 0, 0, 0]),
                        "is_random_next": "0",
                    }
                )
        (self.path / "meta.dat").write_text("sample.csv 32\n")
        args = self.args()
        params, meta = run.build_config(args)
        run.check_data(args, meta)
        loader = train.create_loader(params["trainer"]["fit"]["train_dataloader"])
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["input_ids"].shape), (16, 128))
        cfg = params["trainer"]["init"]["model"]
        cfg.update(hidden_size=16, filter_size=32, num_hidden_layers=1, num_heads=2)
        model = models.make_model(cfg)
        loss = model(batch) / 16
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(
            model.model.bert.encoder.layer[0].attention.self.query.weight.grad
        )

    def test_bert_loss_matches_modelzoo_value_and_gradient(self):
        from cerebras.modelzoo.losses.BertPretrainModelLoss import BertPretrainModelLoss

        logits = torch.randn(2, 3, 17, requires_grad=True)
        nsp = torch.randn(2, 2, requires_grad=True)
        batch = {
            "labels": torch.randint(17, (2, 3)),
            "masked_lm_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
            "next_sentence_label": torch.tensor([[0], [1]]),
        }
        native = models.bert_loss_sum(logits, nsp, batch, 0.058) / 2
        reference = BertPretrainModelLoss(mlm_loss_weight=0.058)(
            logits,
            17,
            batch["labels"],
            nsp,
            batch["next_sentence_label"],
            batch["masked_lm_mask"],
        )
        torch.testing.assert_close(native, reference)
        for actual, expected in zip(
            torch.autograd.grad(native, (logits, nsp), retain_graph=True),
            torch.autograd.grad(reference, (logits, nsp)),
        ):
            torch.testing.assert_close(actual, expected)

    def test_llama_loss_and_accumulation_with_unequal_masks(self):
        from cerebras.modelzoo.losses.GPTLMHeadModelLoss import GPTLMHeadModelLoss

        logits = torch.randn(2, 4, 17, requires_grad=True)
        batch = {
            "labels": torch.randint(17, (2, 4)),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]]),
        }
        denominator = batch["attention_mask"].sum()
        full = GPTLMHeadModelLoss(17, "num_tokens", 1.0)(
            logits, batch["labels"], batch["attention_mask"]
        )
        accumulated = (
            sum(
                models.llama_loss_sum(
                    logits[i : i + 1], {k: v[i : i + 1] for k, v in batch.items()}
                )
                for i in range(2)
            )
            / denominator
        )
        torch.testing.assert_close(full, accumulated)
        torch.testing.assert_close(
            torch.autograd.grad(full, logits, retain_graph=True)[0],
            torch.autograd.grad(accumulated, logits)[0],
        )

    def test_llama_model_backward_and_rope(self):
        params, _ = run.build_config(self.args("llama3p2_1b_msl1024"))
        cfg = params["trainer"]["init"]["model"]
        cfg.update(
            hidden_size=16,
            filter_size=32,
            num_hidden_layers=1,
            num_heads=2,
            vocab_size=32,
            extra_attention_params={"num_kv_groups": 1},
        )
        model = models.make_model(cfg)
        batch = {
            "input_ids": torch.randint(32, (2, 8)),
            "labels": torch.randint(32, (2, 8)),
            "attention_mask": torch.ones((2, 8), dtype=torch.int32),
        }
        loss = model(batch) / 16
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIs(model.model.lm_head.weight, model.model.model.embed_tokens.weight)
        self.assertEqual(model.model.config.rope_scaling["factor"], 32.0)
        self.assertEqual(model.model.config._attn_implementation, "sdpa")

    def test_dry_run_never_calls_qsub_or_writes(self):
        fake = self.path / "qsub"
        marker = self.path / "called"
        fake.write_text(f"#!/bin/bash\ntouch '{marker}'\nexit 99\n")
        fake.chmod(0o755)
        out = self.path / "out with spaces"
        env = {**os.environ, "PATH": str(self.path) + os.pathsep + os.environ["PATH"]}
        result = subprocess.run(
            [
                str(ROOT / "benchmark_scripts/pegasus/submit_non_gnn_nqsv.sh"),
                "--profile",
                "bert_large_msl128",
                "--data-dir",
                "/missing/data",
                "--output-dir",
                str(out),
                "--dry-run",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("qsub", result.stdout)
        self.assertFalse(marker.exists())
        self.assertFalse(out.exists())

    def test_qsub_preserves_config_path_with_spaces(self):
        # Exercise real preparation, but replace the external scheduler only.
        with h5py.File(self.path / "sample.h5", "w") as handle:
            handle.create_dataset("data", shape=(32, 3, 1024), dtype="i4")
        fake = self.path / "qsub"
        captured = self.path / "arguments.json"
        fake.write_text(
            f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\nPath({str(captured)!r}).write_text(json.dumps(sys.argv[1:]))\nprint('test.job')\n"
        )
        fake.chmod(0o755)
        out = self.path / "out with spaces"
        env = {**os.environ, "PATH": str(self.path) + os.pathsep + os.environ["PATH"]}
        result = subprocess.run(
            [
                str(ROOT / "benchmark_scripts/pegasus/submit_non_gnn_nqsv.sh"),
                "--profile",
                "llama3p2_1b_msl1024",
                "--data-dir",
                str(self.path),
                "--output-dir",
                str(out),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads(captured.read_text())
        self.assertEqual(argv[:2], ["-v", f"NON_GNN_CONFIG={out}/params.yaml"])
        self.assertEqual(
            json.loads((out / "launch.json").read_text())["effective_batch_size"], 32
        )


if __name__ == "__main__":
    unittest.main()
