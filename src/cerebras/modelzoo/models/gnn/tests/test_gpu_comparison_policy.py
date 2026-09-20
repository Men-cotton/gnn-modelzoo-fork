"""Native GPU policy: matched knobs without emulating CSX-specific algorithms."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cerebras.modelzoo.models.gnn.gpu_policy import (
    adamw_kwargs,
    adamw_param_groups,
    grad_scaler_kwargs,
    optimizer_policy,
    OptimizerStepCounter,
)
from cerebras.modelzoo.models.gnn.reference.pyg.runner import compile_model
from cerebras.modelzoo.models.gnn.reference.pyg.model import get_model
from cerebras.modelzoo.models.gnn.tools import learning_campaign, hpcasia_campaign
from cerebras.modelzoo.models.gnn.tools import measure_pyg
from cerebras.modelzoo.models.gnn.tools.autotune_backends import PyGBackend


class GPUComparisonPolicyTests(unittest.TestCase):
    def test_native_adamw_groups_exclude_bias_and_norm(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
        init = {"optimizer": {"AdamW": {"lr": 0.003, "weight_decay": 0.0005}}}
        optimizer = torch.optim.AdamW(adamw_param_groups(model), **adamw_kwargs(init))
        self.assertIs(type(optimizer), torch.optim.AdamW)
        groups = {
            id(p): g["weight_decay"]
            for g in optimizer.param_groups
            for p in g["params"]
        }
        self.assertEqual(len(groups), len(list(model.parameters())))
        for name, p in model.named_parameters():
            self.assertEqual(groups[id(p)], 0.0005 if name == "0.weight" else 0)
            p.data.fill_(1)
            p.grad = torch.zeros_like(p)
        optimizer.step()
        self.assertTrue(bool((model[0].weight < 1).all()))
        self.assertTrue(torch.equal(model[0].bias, torch.ones_like(model[0].bias)))
        policy = optimizer_policy(
            model, optimizer, grad_scaler_kwargs({}, torch.float16)
        )
        json.dumps(policy, allow_nan=False)
        self.assertEqual(policy["optimizer_defaults"]["eps"], 1e-6)
        self.assertEqual(policy["amp"]["init_scale"], 32768)
        self.assertEqual(policy["optimizer_groups"][1]["weight_decay"], 0)
        self.assertNotIn("eps", init["optimizer"]["AdamW"])

    def test_amp_explicit_settings_and_unsupported_clamps(self):
        self.assertEqual(
            grad_scaler_kwargs({}, torch.float16),
            dict(
                enabled=True,
                init_scale=32768.0,
                growth_interval=2000,
                growth_factor=2.0,
                backoff_factor=0.5,
            ),
        )
        init = {"precision": {"initial_loss_scale": 8192.0, "steps_per_increase": 100}}
        self.assertEqual(
            grad_scaler_kwargs(init, torch.float16),
            dict(
                enabled=True,
                init_scale=8192.0,
                growth_interval=100,
                growth_factor=2.0,
                backoff_factor=0.5,
            ),
        )
        for dtype in (torch.float32, torch.bfloat16):
            self.assertEqual(grad_scaler_kwargs(init, dtype), {"enabled": False})
        for entry in (
            {"initial_loss_scale": float("nan")},
            {"initial_loss_scale": 0},
            {"steps_per_increase": 0},
            {"loss_scaling_factor": 1},
            {"max_loss_scale": 32768},
            {"min_loss_scale": 1e-6},
            {"max_gradient_norm": 1},
        ):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                grad_scaler_kwargs({"precision": entry}, torch.float16)

    def test_native_cpu_scaler_skip_is_counted(self):
        # Real AMP/AdamW behavior on CPU; no GPU timing claim.
        p = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([p], eps=1e-6)
        counter = OptimizerStepCounter(optimizer)
        scaler = torch.amp.GradScaler("cpu", **grad_scaler_kwargs({}, torch.float16))
        for value, expected in ((float("inf"), 0), (1.0, 1)):
            optimizer.zero_grad(set_to_none=True)
            scaler.scale((p * value).sum()).backward()
            scaler.step(optimizer)
            scaler.update()
            self.assertEqual(counter.steps, expected)
        self.assertEqual(scaler.get_scale(), 16384)

    def test_fused_counter_accounts_for_device_overflow_flag(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
        counter = OptimizerStepCounter(optimizer)
        for flag in (0.0, 1.0, 0.0):
            # Mimic the post-hook contract of fused AdamW, which is invoked
            # even when the kernel suppresses the overflowing update.
            optimizer.found_inf = torch.tensor(flag)
            counter._completed(optimizer)
        self.assertEqual(counter.steps, 2)

    def test_compile_is_explicit_and_dynamic(self):
        model = torch.nn.Linear(2, 2)
        for disabled in ("", "1"):
            with (
                patch.dict("os.environ", {"NO_COMPILE": disabled}),
                patch.object(torch, "compile", return_value=model) as compile_call,
                redirect_stdout(io.StringIO()) as stream,
            ):
                self.assertIs(compile_model(model), model)
                if disabled:
                    compile_call.assert_not_called()
                else:
                    compile_call.assert_called_once_with(model, dynamic=True)
                self.assertEqual(
                    json.loads(stream.getvalue().removeprefix("[Compile] ")),
                    dict(enabled=not bool(disabled), dynamic=not bool(disabled)),
                )

    def test_standard_pyg_sage_bias_and_shared_config(self):
        for dataset in ("arxiv", "products"):
            base = learning_campaign.shared_base(dataset)
            model = get_model(base)
            for conv in model.gnn.convs:
                self.assertIsNone(conv.lin_r.bias)
                self.assertIsNotNone(conv.lin_l.bias)
            self.assertEqual(model.gnn.num_layers, 3)
            for kind in ("learning_r1", "throughput_r1", "cache_r1"):
                configs = []
                for backend in ("csx", "pyg"):
                    args = hpcasia_campaign.parse_args(
                        [
                            "--backend",
                            backend,
                            "--dataset",
                            dataset,
                            "--output",
                            "/tmp/policy-preview-unused",
                            "--run-id",
                            f"seed_43/{kind}",
                        ]
                    )
                    configs.append(hpcasia_campaign.plan(args, base)[0]["config"])
                a, b = [c["trainer"]["init"] for c in configs]
                for field in ("model", "optimizer", "precision", "loop", "seed"):
                    self.assertEqual(a[field], b[field], field)
                self.assertEqual(b["precision"]["initial_loss_scale"], 32768)

    def test_dynamic_pyg_graph_capture_on_cpu(self):
        # Exercise Dynamo with native PyG and changing shapes; eager backend is
        # intentional, so this is not an Inductor/CUDA performance measurement.
        cfg = learning_campaign.shared_base("arxiv")
        cfg["trainer"]["init"]["model"]["architecture"].update(
            n_feat=3, hidden_dim=4, num_layers=2, n_class=2
        )
        model = get_model(cfg).eval()
        compiled = torch.compile(model, dynamic=True, fullgraph=True, backend="eager")
        for nodes in (4, 7):
            x = torch.randn(nodes, 3, requires_grad=True)
            ids = torch.arange(nodes)
            edges = torch.stack([ids, (ids + 1) % nodes])
            expected = model(x, edges)
            actual = compiled(x, edges)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                torch.autograd.grad(actual.sum(), x)[0],
                torch.autograd.grad(expected.sum(), x)[0],
            )

    def test_new_measurements_keep_warmup_skips_and_reject_measured_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "train.log"

            def write(measured_skip=False):
                rows = []
                for step in (10, 20, 30, 40, 50, 60):
                    skips = 1 + int(measured_skip and step >= 40)
                    rows.append(
                        "[Autotune] "
                        + json.dumps(
                            dict(
                                version=2,
                                step=step,
                                wall_seconds=float(step),
                                seed_nodes=step * 4,
                                loss=0.5,
                                all_losses_finite=True,
                                optimizer_steps=step - skips,
                                skipped_optimizer_steps=skips,
                                loss_scale=16384,
                            )
                        )
                    )
                log.write_text(
                    "\n".join(rows + ["Training Completed. Total Steps: 60"])
                )

            write()
            measurement = PyGBackend.measure(log, 20, 60, 2)
            self.assertEqual(measurement["optimizer_steps"], 40)
            self.assertEqual(measurement["skipped_optimizer_steps"], 0)
            write(True)
            self.assertEqual(
                measure_pyg.summarize(log, 20, 60, 2)["skipped_optimizer_steps"], 1
            )
            with self.assertRaisesRegex(ValueError, "AMP skipped"):
                PyGBackend.measure(log, 20, 60, 2)
            text = log.read_text()
            for bad in (
                text.replace('"optimizer_steps": 9', '"optimizer_steps": 10'),
                text.replace('"loss_scale": 16384', '"loss_scale": -1'),
                text.replace('"version": 2', '"version": 1', 1),
            ):
                log.write_text(bad)
                with self.assertRaises(ValueError):
                    measure_pyg.read_points(log)


if __name__ == "__main__":
    unittest.main()
