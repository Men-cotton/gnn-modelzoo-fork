"""Numerators and interval boundaries must agree before comparing hardware."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

from cerebras.modelzoo.models.gnn.tools import (
    measure_fixed_shape,
    measure_pyg,
    measure_window,
)


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "train.log"

    def csx_log(self, contract=None, extra=""):
        rows = []
        if contract is not None:
            rows.append("GNN_INPUT_CONTRACT " + json.dumps(contract))
        rows.append(extra)
        rows.append("Starting train loop 1 of 1, from global step 1 to 10 (10 steps)")
        for step in range(1, 11):
            stamp = (datetime(2026, 1, 1) + timedelta(seconds=step)).strftime(
                "%Y-%m-%d %H:%M:%S,%f"
            )[:-3]
            rows.append(f"{stamp} INFO | Train Device=CSX, Step={step}, Loss=1.0,")
        rows.append("Training completed successfully!")
        self.log.write_text("\n".join(rows))

    def test_quarantined_inputs_are_rejected(self):
        directory = Path(self.temp.name) / "invalid"
        directory.mkdir()
        path = directory / "train.log"
        path.write_text("")
        for parser in (measure_window, measure_fixed_shape, measure_pyg):
            with (
                self.subTest(parser=parser.__name__),
                self.assertRaisesRegex(ValueError, "outside the active"),
            ):
                parser.summarize(path, 2, 10)

    @staticmethod
    def contract(**overrides):
        return dict(
            event="gnn_input_contract",
            version=1,
            split="train",
            num_streamers=1,
            batch_index_origin=0,
            traversal="continuous_sequential_batches",
            traversal_scope="single_data_executor",
            restartable=False,
            ordered_targets_and_labels_sha256="0" * 64,
            batch_size=4,
            seed_nodes_by_batch=[4, 4, 1],
            supervised_targets_by_batch=[3, 4, 0],
            static_batch_cache_size=0,
            **overrides,
        )

    def test_exact_schedule_matches_enumerated_consumption(self):
        self.csx_log(self.contract())
        for start in range(1, 8):
            for end in range(start + 1, 11):
                result = measure_window.summarize(self.log, start, end)
                seeds = sum(
                    [4, 4, 1][(step - 1) % 3] for step in range(start + 1, end + 1)
                )
                supervised = sum(
                    [3, 4, 0][(step - 1) % 3] for step in range(start + 1, end + 1)
                )
                self.assertEqual(result["seed_nodes"], seeds)
                self.assertEqual(result["supervised_targets"], supervised)
                self.assertEqual(result["nominal_slots"], 4 * (end - start))
                self.assertEqual(result["seed_nodes_per_second"], seeds / (end - start))

    def test_missing_conflicting_static_and_restarted_contracts(self):
        self.csx_log()
        with self.assertRaisesRegex(ValueError, "Missing"):
            measure_window.summarize(self.log, 2, 10)
        contract = self.contract()
        contract["static_batch_cache_size"] = 1
        self.csx_log(contract)
        probe = measure_window.summarize_probe(self.log, 2, 10)
        self.assertEqual(probe["metric"], "probe_nominal_slots_per_second")
        self.assertNotIn("seed_nodes_per_second", probe)
        with self.assertRaisesRegex(ValueError, "Static replay"):
            measure_window.summarize(self.log, 2, 10)
        for extra in (
            "INFO Evaluation completed",
            "INFO Loading checkpoint from model_dir",
        ):
            self.csx_log(self.contract(), extra)
            with self.assertRaisesRegex(ValueError, "restart or checkpoint restore"):
                measure_window.summarize(self.log, 4, 10)
        self.csx_log(self.contract())
        with self.assertRaisesRegex(ValueError, "batch size differs"):
            measure_window.summarize(self.log, 2, 10, 8)

    def native_records(self):
        contract = self.contract()
        contract.update(
            batch_size=2, seed_nodes_by_batch=[2, 1], supervised_targets_by_batch=[1, 1]
        )
        records = [
            dict(event="run", backend="fixed_shape_gpu", input_contract=contract)
        ]
        for step in range(1, 6):
            seeds = 2 if step % 2 else 1
            records.append(
                dict(
                    event="train",
                    step=step,
                    end_step=step,
                    start_step=step - 1,
                    boundary_monotonic_seconds=float(step),
                    steps=1,
                    seconds=0.8,
                    loss=1.0,
                    seed_nodes=seeds,
                    supervised_targets=1,
                    nominal_slots=2,
                    optimizer_steps=1,
                    skipped_optimizer_steps=0,
                )
            )
        records.append(
            dict(
                event="summary",
                completed=True,
                start_step=1,
                end_step=5,
                steps=4,
                seconds=3.2,
                seed_nodes=6,
                supervised_targets=4,
                nominal_slots=8,
                optimizer_steps=4,
                skipped_optimizer_steps=0,
            )
        )
        return records

    def test_gpu_exact_endpoint_time_includes_logging(self):
        self.log.write_text(
            "startup diagnostic\n" + "\n".join(map(json.dumps, self.native_records()))
        )
        result = measure_fixed_shape.summarize(self.log, 1, 5)
        self.assertEqual(result["seed_nodes"], 6)
        self.assertEqual(result["supervised_targets"], 4)
        self.assertEqual(result["nominal_slots"], 8)
        self.assertEqual(result["optimizer_steps"], 4)
        self.assertEqual(result["throughput"], 1.5)
        self.assertEqual(result["summed_training_windows_seconds"], 3.2)
        self.assertTrue(result["half_window_check"]["within_tolerance"])
        for mutation in (
            "eval",
            "incomplete",
            "gap",
            "nonfinite",
            "mixed",
            "counter",
            "summary",
            "seconds",
            "updates",
        ):
            rows = self.native_records()
            if mutation == "eval":
                rows.insert(2, dict(event="eval", step=1))
            elif mutation == "incomplete":
                rows[-1]["completed"] = False
            elif mutation == "gap":
                rows[3]["start_step"] = 1
            elif mutation == "nonfinite":
                rows[3]["loss"] = float("nan")
            elif mutation == "counter":
                rows[3]["seed_nodes"] = 3
            elif mutation == "summary":
                rows[-1]["seed_nodes"] = 7
            elif mutation == "seconds":
                rows[3]["seconds"] = float("nan")
            elif mutation == "updates":
                rows[3]["optimizer_steps"] = 0
            else:
                rows += rows
            self.log.write_text("\n".join(map(json.dumps, rows)))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                measure_fixed_shape.summarize(self.log, 1, 5)


if __name__ == "__main__":
    unittest.main()
