"""Campaign control-flow/configuration and real stdlib observer tests; no CSX."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from cerebras.modelzoo.common.utils.run.config_loader import load_params_file
from cerebras.modelzoo.models.gnn.data_processing import worker_resources as resources
from cerebras.modelzoo.models.gnn.tools import worker_campaign as campaign
from test_autotune import write_log


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = campaign.parse_args(
            ["--output", str(self.root / "campaign"), "--no-archive"]
        )
        self.args.output.mkdir()
        self.base = load_params_file(campaign.GNN / "configs/autotune/arxiv_w40.yaml")

    def test_plan_preserves_reference_and_separates_interventions(self):
        stages = campaign.plan(self.args, self.base)
        self.assertEqual(sum(s["trials"] for s in stages), 37)
        self.assertEqual(stages[0]["workers"], 4)
        for stage in stages:
            if stage["kind"] == "sensitivity":
                self.assertIn("--continue-on-failure", stage["command"])
        first_large = next(i for i, s in enumerate(stages) if s["workers"] > 4)
        self.assertTrue(
            all(
                i < first_large
                for i, s in enumerate(stages)
                if s["kind"] == "intervention"
            )
        )
        for s in stages:
            if "config" not in s:
                continue
            loader = s["config"]["trainer"]["fit"]["train_dataloader"]
            self.assertEqual(loader["batch_size"], 4096)
            self.assertEqual(loader["fanouts"], [15, 10, 5])
            self.assertEqual(
                loader["worker_diagnostics"]["enabled"], s["kind"] == "diagnostic"
            )
            if s["control"]:
                baseline = campaign.configuration(
                    self.base,
                    4,
                    Path(s["directory"]) / "model",
                    440,
                    self.args.job_time_sec,
                )
                expected = baseline["trainer"]["fit"]["train_dataloader"]
                changed = {key for key in loader if loader[key] != expected.get(key)}
                self.assertEqual(changed, set(campaign.CONTROLS[s["control"]]))

    def test_failed_stage_continues_and_resume_does_not_retry(self):
        stages = [
            s for s in campaign.plan(self.args, self.base) if s["kind"] == "sensitivity"
        ][:2]
        with patch.object(campaign, "run_sensitivity", side_effect=[1, 0]) as run:
            self.assertEqual(campaign.execute(self.args, stages, "identity"), 2)
            self.assertEqual(campaign.execute(self.args, stages, "identity"), 2)
            self.assertEqual(run.call_count, 2)
        state = json.loads((self.args.output / "campaign.json").read_text())
        self.assertEqual(state["status"], "completed_with_failures")
        self.assertEqual(state["stages"][0]["status"], "failed")
        self.assertEqual(state["stages"][1]["status"], "completed")

    def test_direct_timeout_continues_to_successful_measurement(self):
        stages = campaign.plan(self.args, self.base)[1:3]
        calls = []

        def run(command, log, timeout):
            calls.append(command)
            if len(calls) == 1:
                return {"status": "timeout"}
            write_log(log, end=440)
            return {"status": "completed", "returncode": 0}

        with patch.object(campaign.autotune, "execute", side_effect=run):
            self.assertEqual(campaign.execute(self.args, stages, "identity"), 2)
        state = json.loads((self.args.output / "campaign.json").read_text())
        self.assertEqual(len(calls), 2)
        self.assertEqual(state["status"], "completed_with_failures")
        self.assertEqual(state["stages"][0]["client_status"], "timeout")
        self.assertGreater(
            state["stages"][1]["measurement"]["nominal_slots_per_second"], 0
        )

    def test_nested_missing_measurements_do_not_stop_next_stage(self):
        stages = [
            s for s in campaign.plan(self.args, self.base) if s["kind"] == "sensitivity"
        ][:2]
        calls = []

        def run(command, log):
            calls.append(command)
            if len(calls) == 1:
                campaign.autotune.write_json(
                    log.parent / "study.json",
                    dict(status="completed_with_missing_measurements", trials=[]),
                )
                return 2
            return 0

        with patch.object(campaign, "run_sensitivity", side_effect=run):
            self.assertEqual(campaign.execute(self.args, stages, "identity"), 2)
            self.assertEqual(campaign.execute(self.args, stages, "identity"), 2)
        self.assertEqual(len(calls), 2)
        state = json.loads((self.args.output / "campaign.json").read_text())
        self.assertEqual(state["stages"][0]["status"], "completed_with_failures")
        self.assertEqual(state["stages"][1]["status"], "completed")

    def test_explicit_interrupt_still_stops_later_submissions(self):
        stages = campaign.plan(self.args, self.base)[1:3]
        with patch.object(
            campaign.autotune, "execute", return_value={"status": "interrupted"}
        ) as run:
            self.assertEqual(campaign.execute(self.args, stages, "identity"), 2)
            self.assertEqual(run.call_count, 1)
        state = json.loads((self.args.output / "campaign.json").read_text())
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["stages"][1]["status"], "pending")
        with self.assertRaisesRegex(ValueError, "interrupted"):
            campaign.execute(self.args, stages, "identity")

    def test_completed_resume_skips_training_and_changed_identity_is_rejected(self):
        stages = campaign.plan(self.args, self.base)[:1]
        with patch.object(campaign, "run_sensitivity", return_value=0) as run:
            self.assertEqual(campaign.execute(self.args, stages, "identity"), 0)
            self.assertEqual(campaign.execute(self.args, stages, "identity"), 0)
            self.assertEqual(run.call_count, 1)
        with self.assertRaisesRegex(ValueError, "changed"):
            campaign.execute(self.args, stages, "different")

    def test_budget_pause_preserves_inner_cumulative_budget_on_resume(self):
        stages = campaign.plan(self.args, self.base)[:1]
        directory = Path(stages[0]["directory"])
        seen_budgets = []

        def run(command, log):
            seen_budgets.append(int(command[command.index("--budget-sec") + 1]))
            if len(seen_budgets) == 1:
                campaign.autotune.write_json(
                    directory / "study.json",
                    dict(status="budget_exhausted", used_sec=1000, trials=[]),
                )
                return 2
            return 0

        with patch.object(campaign, "run_sensitivity", side_effect=run):
            self.assertEqual(campaign.execute(self.args, stages, "identity"), 2)
            state = json.loads((self.args.output / "campaign.json").read_text())
            self.assertEqual(state["status"], "budget_exhausted")
            self.assertEqual(state["stages"][0]["status"], "pending")
            self.assertEqual(campaign.execute(self.args, stages, "identity"), 0)
        self.assertGreater(seen_budgets[1], seen_budgets[0] + 990)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_v1_memory_and_v2_limits_errors_and_ancestors(self):
        proc = self.root / "proc/123"
        proc.mkdir(parents=True)
        mount = self.root / "cgroup"
        leaf = mount / "worker"
        leaf.mkdir(parents=True)
        for version in (1, 2):
            (proc / "cgroup").write_text(
                "5:memory:/worker\n" if version == 1 else "0::/worker\n"
            )
            (proc / "mountinfo").write_text(
                f"1 0 0:1 / {mount} rw - "
                + (
                    "cgroup cgroup rw,memory\n"
                    if version == 1
                    else "cgroup2 cgroup rw\n"
                )
            )
            name = "memory.limit_in_bytes" if version == 1 else "memory.max"
            (leaf / name).write_text("1048576")
            result = resources.cgroups(123, self.root / "proc")
            hierarchy = result["hierarchies"][0]
            self.assertEqual(hierarchy["version"], version)
            self.assertEqual(len(hierarchy["ancestors"]), 2)
            self.assertEqual(
                hierarchy["ancestors"][0]["files"][name]["value"], "1048576"
            )
            self.assertIn("error", hierarchy["ancestors"][1]["files"][name])

    def test_independent_observer_records_stalled_parent_and_exits(self):
        parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        observer = None
        try:
            identity = resources.task_stat(Path("/proc") / str(parent.pid) / "stat")
            output = self.root / "resources.jsonl"
            observer = subprocess.Popen(
                [
                    sys.executable,
                    resources.__file__,
                    "--parent",
                    str(parent.pid),
                    "--start-ticks",
                    str(identity["start_ticks"]),
                    "--output",
                    str(output),
                    "--interval",
                    "0.1",
                    "--samples",
                    "100",
                    "--pss",
                ]
            )
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if output.exists() and len(output.read_text().splitlines()) >= 2:
                    break
                time.sleep(0.05)
            parent.terminate()
            parent.wait(timeout=3)
            self.assertEqual(observer.wait(timeout=3), 0)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertGreaterEqual(len(rows), 3)
            self.assertEqual(rows[-1]["reason"], "parent_exited_or_reused")
            self.assertIn("shared_memory", rows[0])
            self.assertIn("smaps_rollup", rows[0]["processes"][0])
        finally:
            for process in (observer, parent):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()

    def test_reused_pid_emits_no_samples(self):
        output = self.root / "resources.jsonl"
        resources.observe(os.getpid(), -1, output, 0.1, 1)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reason"], "parent_exited_or_reused")

    def test_loader_starts_observer_only_when_requested(self):
        from test_worker_diagnostics import make_loader, records, shutdown

        loader = make_loader(
            str(self.root),
            2,
            True,
            resource_monitor=True,
            resource_monitor_pss=True,
            snapshot_interval_seconds=0.1,
        )
        try:
            list(loader)
            events = records(self.root)
            started = [e for e in events if e["event"] == "resource_monitor_started"]
            self.assertEqual(len(started), 1)
            path = Path(started[0]["output"])
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if path.exists() and "resource_monitor_stopped" in path.read_text():
                    break
                time.sleep(0.05)
            samples = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(samples[-1]["reason"], "sample_budget")
            self.assertEqual(len(samples), 3)
            self.assertTrue(
                any(e["event"] == "batch_layout" for e in records(self.root))
            )
        finally:
            shutdown(loader)


if __name__ == "__main__":
    unittest.main()
