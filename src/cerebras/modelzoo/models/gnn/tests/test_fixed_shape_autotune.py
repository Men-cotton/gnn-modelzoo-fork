"""Exercise GPU comparison orchestration without an accelerator or remote jobs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from cerebras.modelzoo.models.gnn.tools import autotune as tune


class FixedShapeTunerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.args = tune.parse_args(
            [
                "--backend",
                "fixed_shape",
                "--dataset",
                "arxiv",
                "--mode",
                "sensitivity",
                "--workers",
                "0",
                "1",
                "--prefetch-factor",
                "3",
                "--compile",
                "--output",
                str(self.output),
                "--budget-sec",
                "100000",
            ]
        )
        self.base = tune.load_params_file(tune.GNN / "configs/autotune/arxiv_w40.yaml")
        self.base["trainer"]["init"].pop("backend")
        self.base["trainer"]["init"]["optimizer"]["AdamW"].update(
            eps=1e-6, betas=[0.9, 0.999]
        )

    def execute(self, command, log, timeout):
        self.assertTrue(command[2].endswith("fixed_shape_gpu.py"))
        self.assertIn("--compile", command)
        config = yaml.safe_load(
            Path(command[command.index("--config") + 1]).read_text()
        )
        init = config["trainer"]["init"]
        loader = config["trainer"]["fit"]["train_dataloader"]
        self.assertNotIn("backend", init)
        self.assertEqual(init["optimizer"], self.base["trainer"]["init"]["optimizer"])
        self.assertIsNone(config["trainer"]["fit"]["val_dataloader"])
        self.assertEqual(
            loader["prefetch_factor"], 3 if loader["num_workers"] else None
        )
        self.assertTrue(loader["measure_batch_accounting"])
        seconds = 1 if loader["num_workers"] else 2
        contract = dict(
            event="gnn_input_contract",
            version=1,
            split="train",
            dataset_name="ogbn-arxiv",
            num_streamers=1,
            batch_index_origin=0,
            traversal="continuous_sequential_batches",
            traversal_scope="single_data_executor",
            restartable=False,
            batch_size=4096,
            static_batch_cache_size=0,
            ordered_targets_and_labels_sha256="a" * 64,
            seed_nodes_by_batch=[4096] * 9 + [1234],
            supervised_targets_by_batch=[4096] * 9 + [1234],
        )
        rows = [dict(event="run", backend="fixed_shape_gpu", input_contract=contract)]
        for step in range(10, init["loop"]["max_steps"] + 1, 10):
            rows.append(
                dict(
                    event="train",
                    step=step,
                    start_step=step - 10,
                    steps=10,
                    boundary_monotonic_seconds=100 + step * seconds,
                    seconds=10 * seconds,
                    loss=0.5,
                    seed_nodes=9 * 4096 + 1234,
                    supervised_targets=9 * 4096 + 1234,
                    nominal_slots=10 * 4096,
                    optimizer_steps=10,
                    skipped_optimizer_steps=0,
                )
            )
        blocks = (init["loop"]["max_steps"] - 40) // 10
        rows.append(
            dict(
                event="summary",
                completed=True,
                start_step=40,
                end_step=init["loop"]["max_steps"],
                steps=blocks * 10,
                seed_nodes=blocks * (9 * 4096 + 1234),
                supervised_targets=blocks * (9 * 4096 + 1234),
                nominal_slots=blocks * 10 * 4096,
                optimizer_steps=blocks * 10,
                skipped_optimizer_steps=0,
                seconds=blocks * 10 * seconds,
            )
        )
        log.write_text("\n".join(json.dumps(row) for row in rows))
        return dict(status="completed", returncode=0)

    def test_repeated_actual_seed_measurement_and_resume(self):
        with patch.object(tune, "execute", side_effect=self.execute) as execute:
            study = tune.Study(self.args, self.base, self.output, {"test": "host"})
            self.assertEqual(study.run(), 0)
            self.assertEqual(execute.call_count, 6)
            rows = study.state["trials"]
            self.assertEqual(
                [row["knobs"]["num_workers"] for row in rows], [0, 1, 1, 0, 0, 1]
            )
            measured = rows[1]["measurement"]
            self.assertEqual(measured["metric"], "seed_nodes_per_second")
            self.assertEqual(measured["seed_nodes"], 40 * (9 * 4096 + 1234))
            self.assertEqual(measured["nominal_slots"], 400 * 4096)
            self.assertLess(
                measured["throughput"], measured["nominal_slots_per_second"]
            )
            self.assertEqual(
                tune.Study(self.args, self.base, self.output, {"test": "host"}).run(), 0
            )
            self.assertEqual(execute.call_count, 6)

    def test_explicit_base_config_keeps_optimizer_in_preview(self):
        source = self.output / "selected.yaml"
        source.write_text(yaml.safe_dump(self.base))
        preview = self.output / "preview"
        self.assertEqual(
            tune.main(
                [
                    "--backend",
                    "fixed_shape",
                    "--dataset",
                    "arxiv",
                    "--workers",
                    "0",
                    "--base-config",
                    str(source),
                    "--output",
                    str(preview),
                    "--budget-sec",
                    "100000",
                    "--dry-run",
                ]
            ),
            0,
        )
        config = yaml.safe_load((preview / "preview_w00.yaml").read_text())
        self.assertEqual(
            config["trainer"]["init"]["optimizer"],
            self.base["trainer"]["init"]["optimizer"],
        )
        with self.assertRaisesRegex(ValueError, "dataset"):
            tune.main(
                [
                    "--backend",
                    "fixed_shape",
                    "--dataset",
                    "products",
                    "--workers",
                    "0",
                    "--base-config",
                    str(source),
                    "--output",
                    str(self.output / "wrong"),
                    "--budget-sec",
                    "100000",
                    "--dry-run",
                ]
            )


if __name__ == "__main__":
    unittest.main()
