"""Offline campaign tests: real data processors, fake tokenizer and scheduler."""

import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
import shutil
import signal
import time
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmark_scripts/non_gnn"
sys.path.insert(0, str(BENCH))
campaign = importlib.import_module("campaign")
prepare_data = importlib.import_module("prepare_data")
run = importlib.import_module("run")


class FakeTokenizer:
    bos_token_id = 128000
    eos_token_id = 128001

    def __call__(self, documents, **kwargs):
        return {
            "input_ids": [
                [100 + i % 100 for i, _ in enumerate(text.split())]
                for text in documents
            ]
        }

    def save_pretrained(self, path):
        path.mkdir()
        (path / "tokenizer.json").write_text('{"test": true}')


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test-non-gnn-campaign-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.profiles = yaml.safe_load(run.PROFILES.read_text())
        self.documents = [
            "The cat sat on the mat. A dog runs in the park. " * 15 for _ in range(48)
        ]

    def argv(self, backend="GPU", *extra):
        return [
            "--backend",
            backend,
            "--data-root",
            str(self.path / "data"),
            "--output-dir",
            str(self.path / f"output_{backend}"),
            "--num-workers",
            "0",
            "--repeats",
            "1",
            "--effective-batch-size",
            "2",
            "--gpu-micro-batch-size",
            "1",
            *extra,
        ]

    def args(self, backend="GPU", *extra):
        args = campaign.parser().parse_args(self.argv(backend, *extra))
        args.only = args.only or list(self.profiles)
        return args

    def prepare(self):
        with (
            mock.patch.object(
                prepare_data,
                "load_llama_tokenizer",
                return_value=(FakeTokenizer(), {"source": "test"}),
            ),
            mock.patch.object(
                prepare_data,
                "acquire_documents",
                return_value=(self.documents, {"kind": "test"}),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return prepare_data.prepare(self.args(), self.profiles, run.VOCAB)

    def test_uv_exposes_compiler_to_shell_children_without_activation(self):
        uv = shutil.which("uv")
        self.assertIsNotNone(uv)
        project = self.path / "uv-project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname="compiler-path-test"\nversion="0.0.0"\n'
        )
        created = subprocess.run(
            [
                uv,
                "venv",
                "--offline",
                "--python",
                sys.executable,
                str(project / ".venv"),
            ],
            env={**os.environ, "UV_CACHE_DIR": str(self.path / "uv-cache")},
            text=True,
            capture_output=True,
        )
        self.assertEqual(created.returncode, 0, created.stderr)
        bin_dir = project / ".venv/bin"
        compiler = bin_dir / "torch-cirh-opt"
        compiler.write_text("#!/bin/sh\necho compiler-found\n")
        compiler.chmod(0o755)
        env = {
            **os.environ,
            "PATH": "/usr/bin:/bin",
            "UV_CACHE_DIR": str(self.path / "uv-cache"),
        }
        for name in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
            env.pop(name, None)
        missing = subprocess.run(
            ["/bin/sh", "-c", "torch-cirh-opt"], env=env, capture_output=True
        )
        self.assertEqual(missing.returncode, 127)
        found = subprocess.run(
            [
                uv,
                "run",
                "--offline",
                "--no-sync",
                "--project",
                str(project),
                "/bin/sh",
                "-c",
                "torch-cirh-opt",
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(found.returncode, 0, found.stderr)
        self.assertEqual(found.stdout.strip(), "compiler-found")
        self.assertFalse((project / "uv.lock").exists())

    def test_missing_compiler_stops_campaign_before_preprocessing(self):
        with (
            mock.patch.object(run.shutil, "which", return_value=None),
            mock.patch.object(prepare_data, "prepare") as prepare,
            mock.patch.object(run, "execute_prepared") as start,
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            with self.assertRaises(SystemExit):
                campaign.main(self.argv("CSX"))
        self.assertIn("torch-cirh-opt is not on PATH", errors.getvalue())
        self.assertIn("uv run --no-sync", errors.getvalue())
        prepare.assert_not_called()
        start.assert_not_called()
        self.assertFalse((self.path / "data").exists())

    def test_missing_dependencies_fail_before_data_or_submission(self):
        for backend in ("CSX", "GPU"):
            errors = io.StringIO()
            with (
                mock.patch.object(campaign.shutil, "which", return_value="qsub"),
                mock.patch.object(
                    run.importlib,
                    "import_module",
                    side_effect=ModuleNotFoundError("No module named 'datasets'"),
                ),
                mock.patch.object(prepare_data, "prepare") as prepare,
                mock.patch.object(campaign, "submit_gpu") as submit,
                mock.patch.object(run, "execute_prepared") as start,
                contextlib.redirect_stderr(errors),
            ):
                with self.assertRaises(SystemExit):
                    campaign.main(self.argv(backend))
            prepare.assert_not_called()
            submit.assert_not_called()
            start.assert_not_called()
            self.assertIn("datasets: ModuleNotFoundError", errors.getvalue())
            self.assertIn("benchmark_scripts/non_gnn/setup.sh", errors.getvalue())
            self.assertIn(sys.executable, errors.getvalue())
            self.assertFalse((self.path / "data").exists())

    def test_setup_preserves_installed_cpu_and_cuda_builds(self):
        # Exercise the shell script with fake Python/uv; never install into .venv.
        for flavor in ("cpu", "cu121"):
            root = self.path / flavor
            folder = root / "benchmark_scripts/non_gnn"
            folder.mkdir(parents=True)
            script = folder / "setup.sh"
            script.write_text((BENCH / "setup.sh").read_text())
            python = root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text(
                f'#!/bin/bash\nif [[ "$1" == "-" ]]; then cat >/dev/null; echo "2.4.0+{flavor} {flavor}"; else exit 0; fi\n'
            )
            python.chmod(0o755)
            captured = root / "install.json"
            uv = root / "uv"
            uv.write_text(
                f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
                "args=sys.argv[1:]\n"
                "if args[0] == 'run':\n"
                " import os\n"
                " root=Path(args[args.index('--project')+1])\n"
                " pos=args.index('python')\n"
                " os.execv(str(root/'.venv/bin/python'), [str(root/'.venv/bin/python'), *args[pos+1:]])\n"
                "constraints=Path(args[args.index('--constraint')+1]).read_text()\n"
                f"Path({str(captured)!r}).write_text(json.dumps([args,constraints]))\n"
            )
            uv.chmod(0o755)
            result = subprocess.run(
                ["bash", str(script)],
                text=True,
                capture_output=True,
                env={**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"]},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            args, constraints = json.loads(captured.read_text())
            self.assertIn(f"torch===2.4.0+{flavor}", constraints)
            self.assertIn(f"torchvision==0.19.0+{flavor}", constraints)
            self.assertIn(f"https://download.pytorch.org/whl/{flavor}", args)
            self.assertEqual(args[args.index("--python") + 1], str(python))
            self.assertFalse(Path(args[args.index("--constraint") + 1]).exists())

    def test_cache_hit_skips_source_and_tokenizer_on_both_backends(self):
        data, manifest = self.prepare()
        with (
            mock.patch.object(
                prepare_data,
                "acquire_documents",
                side_effect=AssertionError("download called"),
            ),
            mock.patch.object(
                prepare_data,
                "load_llama_tokenizer",
                side_effect=AssertionError("tokenizer called"),
            ),
            mock.patch.object(
                prepare_data,
                "write_bert",
                side_effect=AssertionError("BERT preprocessing called"),
            ),
            mock.patch.object(
                prepare_data,
                "write_llama",
                side_effect=AssertionError("Llama preprocessing called"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for backend in ("CSX", "GPU"):
                actual, reused = prepare_data.prepare(
                    self.args(backend), self.profiles, run.VOCAB
                )
                self.assertEqual(actual, data)
                self.assertEqual(reused, manifest)

    def test_corruption_and_missing_files_rebuild_before_reuse(self):
        data, manifest = self.prepare()
        path = data / "bert_msl128/part-000.csv"
        original = path.read_bytes()
        # Same size corruption must fail the checksum, not just a size check.
        path.write_bytes(original.replace(b"tokens", b"broken", 1))
        self.assertIsNone(prepare_data.inspect_cache(data, manifest["request"]))
        rebuilt, _ = self.prepare()
        self.assertEqual(rebuilt, data)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(list(data.parent.glob(data.name + ".invalid-*"))), 1)
        (data / "llama_corpus/tokens.h5").unlink()
        self.assertIsNone(prepare_data.inspect_cache(data, manifest["request"]))
        self.prepare()
        self.assertIsNotNone(prepare_data.inspect_cache(data, manifest["request"]))

    def test_changed_source_or_preprocessing_invalidates_cache(self):
        raw = self.path / "source.txt"
        raw.write_text("First document.\n\nSecond document.")
        args = self.args("CSX", "--raw-text", str(raw))
        before = prepare_data.request_spec(args, self.profiles, run.VOCAB)
        raw.write_text("Changed document.\n\nSecond document.")
        after = prepare_data.request_spec(args, self.profiles, run.VOCAB)
        self.assertNotEqual(
            prepare_data.cache_path(args.data_root, before),
            prepare_data.cache_path(args.data_root, after),
        )
        args.seed += 1
        self.assertNotEqual(
            after, prepare_data.request_spec(args, self.profiles, run.VOCAB)
        )

    def test_generated_data_is_consumed_by_existing_processors(self):
        from gpu.train import create_loader

        data, manifest = self.prepare()
        for name in self.profiles:
            job = next(
                job
                for job in campaign.jobs(self.args(), self.profiles, data)
                if job["profile"] == name and job["gpu_implementation"] == "native"
            )
            params, _ = run.build_config(
                run.parser().parse_args(job["prepare_arguments"])
            )
            config = params["trainer"]["fit"]["train_dataloader"]
            config["shuffle"] = False
            batch = next(iter(create_loader(config)))
            length = self.profiles[name]["sequence_length"]
            self.assertEqual(tuple(batch["input_ids"].shape), (1, length))
            if name.startswith("bert"):
                self.assertGreater(int(batch["masked_lm_mask"].sum()), 0)
                self.assertTrue(
                    set(batch["next_sentence_label"].flatten().tolist()) <= {0, 1}
                )
                folder = data / f"bert_msl{length}"
                count = 0
                for line in (folder / "meta.dat").read_text().splitlines():
                    filename, expected = line.split()
                    with (folder / filename).open() as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(len(rows), int(expected))
                    count += len(rows)
                self.assertEqual(
                    count, manifest["stats"]["bert"][str(length)]["samples"]
                )
            else:
                with h5py.File(data / "llama_corpus/tokens.h5") as handle:
                    expected = handle["data"][: length + 1]
                np.testing.assert_array_equal(batch["input_ids"][0], expected[:-1])
                np.testing.assert_array_equal(batch["labels"][0], expected[1:])
                self.assertEqual(int(expected[0]), FakeTokenizer.bos_token_id)
                self.assertIn(FakeTokenizer.eos_token_id, expected)

    def test_default_wrappers_dry_run_without_download_or_writes(self):
        for wrapper in ("cerebras/run_non_gnn.sh", "pegasus/submit_non_gnn_nqsv.sh"):
            output = self.path / wrapper.split("/")[0]
            result = subprocess.run(
                [
                    str(ROOT / "benchmark_scripts" / wrapper),
                    "--dry-run",
                    "--data-root",
                    str(self.path / "missing"),
                    "--output-dir",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Salesforce/wikitext", result.stdout)
            self.assertEqual(
                result.stdout.count("--execute-config")
                + result.stdout.count("qsub -v"),
                12,
            )
            self.assertFalse(output.exists())
            self.assertFalse((self.path / "missing").exists())

    def test_cached_gpu_campaign_submits_all_native_and_modelzoo_jobs(self):
        self.prepare()
        fake = self.path / "qsub"
        captured = self.path / "qsub.jsonl"
        fake.write_text(
            f"#!{sys.executable}\nimport json,sys\nwith open({str(captured)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\nprint('test.job')\n"
        )
        fake.chmod(0o755)
        with (
            mock.patch.dict(
                os.environ, {"PATH": str(self.path) + os.pathsep + os.environ["PATH"]}
            ),
            mock.patch.object(
                prepare_data,
                "acquire_documents",
                side_effect=AssertionError("download called"),
            ),
            mock.patch.object(
                prepare_data,
                "load_llama_tokenizer",
                side_effect=AssertionError("tokenizer called"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                campaign.main(self.argv("GPU", "--gpu-implementation", "both")), 0
            )
        submissions = [json.loads(line) for line in captured.read_text().splitlines()]
        self.assertEqual(len(submissions), 8)
        state = json.loads((self.path / "output_GPU/campaign.json").read_text())
        self.assertTrue(all(job["state"] == "qsub_returned" for job in state["jobs"]))
        for job in state["jobs"]:
            config = Path(job["config"])
            launch = json.loads((config.parent / "launch.json").read_text())
            self.assertTrue(config.is_file())
            self.assertEqual(launch["gpu_implementation"], job["gpu_implementation"])
        # CSX uses exactly the same cache, without any source/tokenizer lookup.
        with (
            mock.patch.object(
                prepare_data,
                "acquire_documents",
                side_effect=AssertionError("download called"),
            ),
            mock.patch.object(
                prepare_data,
                "load_llama_tokenizer",
                side_effect=AssertionError("tokenizer called"),
            ),
            mock.patch.object(
                run,
                "execute_prepared",
                return_value={
                    "state": "completed",
                    "exit_code": 0,
                    "measurement_status": "valid",
                },
            ) as launch_csx,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(campaign.main(self.argv("CSX")), 0)
        self.assertEqual(launch_csx.call_count, 4)

    def test_all_configs_validated_before_any_submission(self):
        self.prepare()
        calls = 0
        real_main = run.main

        def fail_second(argv):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("invalid second config")
            return real_main(argv)

        with (
            mock.patch.object(campaign.shutil, "which", return_value="qsub"),
            mock.patch.object(run, "main", side_effect=fail_second),
            mock.patch.object(campaign, "submit_gpu") as submit,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit):
                campaign.main(self.argv())
        submit.assert_not_called()
        self.assertEqual(calls, 2)
        summary = json.loads((self.path / "output_GPU/summary.json").read_text())
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertEqual(summary["counts"]["pending"], 3)
        self.assertEqual(summary["counts"]["completed"], 0)
        self.assertEqual(summary["counts"]["valid"], 0)
        self.assertEqual(summary["runs"][1]["state"], "prepare_failed")
        self.assertIn("invalid second config", summary["runs"][1]["reason"])

    def test_repeats_reverse_profile_order_and_keep_seed(self):
        args = self.args("CSX", "--repeats", "3")
        jobs = campaign.jobs(args, self.profiles, self.path / "data")
        expected = list(self.profiles)
        self.assertEqual(
            [job["profile"] for job in jobs], expected + expected[::-1] + expected
        )
        self.assertEqual([job["repeat"] for job in jobs], [1] * 4 + [2] * 4 + [3] * 4)
        self.assertEqual(len({job["config"] for job in jobs}), 12)
        for job in jobs:
            parsed = run.parser().parse_args(job["prepare_arguments"])
            self.assertEqual(parsed.seed, args.seed)
            self.assertEqual(parsed.study_dir, args.output_dir)
            self.assertFalse(parsed.detach)

    def test_detach_happens_before_preparation_and_keeps_output_new(self):
        with (
            mock.patch.object(run, "check_runtime"),
            mock.patch.object(run, "check_dependencies"),
            mock.patch.object(prepare_data, "prepare") as prepare,
            mock.patch.object(
                run.benchmark_launcher, "launch", return_value=0
            ) as launch,
        ):
            self.assertEqual(campaign.main(self.argv("CSX", "--detach")), 0)
            prepare.assert_not_called()
            command, metadata_dir = launch.call_args.args
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(
                command[-3:],
                ["--foreground", "--output-dir", str(self.path / "output_CSX")],
            )
            self.assertEqual(
                metadata_dir, Path(str(self.path / "output_CSX") + ".launcher")
            )
            self.assertFalse((self.path / "output_CSX").exists())
            (self.path / "output_CSX").mkdir()
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                campaign.main(self.argv("CSX", "--detach"))
            self.assertEqual(launch.call_count, 1)

    def test_sequential_clients_continue_failure_and_stop_interrupt(self):
        self.prepare()
        states = iter(
            [
                {
                    "state": "failed",
                    "exit_code": 7,
                    "measurement_status": "unavailable",
                },
                {"state": "completed", "exit_code": 0, "measurement_status": "invalid"},
                {
                    "state": "interrupted",
                    "exit_code": 143,
                    "measurement_status": "unavailable",
                },
            ]
        )
        calls = []

        def execute(config, **kwargs):
            prepared = list((self.path / "output_CSX").glob("r*/*/launch.json"))
            self.assertEqual(len(prepared), 4)
            calls.append(config)
            return next(states)

        with (
            mock.patch.object(run, "execute_prepared", side_effect=execute),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(campaign.main(self.argv("CSX")), 143)
        self.assertEqual(len(calls), 3)
        state = json.loads((self.path / "output_CSX/campaign.json").read_text())
        self.assertEqual(state["state"], "interrupted")
        self.assertEqual(
            [job["state"] for job in state["jobs"]],
            ["failed", "completed", "interrupted", "prepared"],
        )

    def test_completed_client_without_measurement_fails_campaign(self):
        self.prepare()
        with (
            mock.patch.object(
                run,
                "execute_prepared",
                return_value={
                    "state": "completed",
                    "exit_code": 0,
                    "measurement_status": "unavailable",
                },
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(campaign.main(self.argv("CSX")), 2)
        self.assertEqual(
            json.loads((self.path / "output_CSX/campaign.json").read_text())["state"],
            "failed",
        )

    def test_interruption_during_config_preparation_keeps_journal(self):
        self.prepare()

        def interrupt(_argv):
            os.kill(os.getpid(), signal.SIGTERM)

        with (
            mock.patch.object(run, "main", side_effect=interrupt),
            mock.patch.object(run, "execute_prepared") as execute,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(campaign.main(self.argv("CSX")), 143)
        execute.assert_not_called()
        state = json.loads((self.path / "output_CSX/campaign.json").read_text())
        self.assertEqual(state["state"], "interrupted")
        self.assertEqual(state["jobs"][0]["state"], "interrupted")
        self.assertTrue(all(job["state"] == "planned" for job in state["jobs"][1:]))

    def prepared_client(self, name, code):
        output = self.path / name
        output.mkdir()
        config = output / "params.yaml"
        config.write_text("test: true\n")
        launch = {
            "backend": "GPU",
            "gpu_implementation": "native",
            "profile": "bert_large_msl128",
            "sequence_length": 128,
            "effective_batch_size": 2,
            "warmup_steps": 1,
            "max_optimizer_steps": 2,
            "repeat": 1,
            "params_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "command": [sys.executable, "-u", "-c", code],
        }
        (output / "launch.json").write_text(json.dumps(launch))
        return config

    def test_client_records_success_failure_and_refuses_replay(self):
        for code in (0, 7):
            config = self.prepared_client(
                f"client_{code}", f"print('client output'); raise SystemExit({code})"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                status = run.execute_prepared(config, backend="GPU")
            self.assertEqual(status["state"], "completed" if code == 0 else "failed")
            self.assertEqual(status["exit_code"], code)
            self.assertEqual(status["measurement_status"], "unavailable")
            self.assertGreater(status["wall_seconds"], 0)
            self.assertIn("client output", (config.parent / "console.log").read_text())
            original = (config.parent / "client_status.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "already has execution artifacts"):
                run.execute_prepared(config)
            self.assertEqual(
                (config.parent / "client_status.json").read_bytes(), original
            )

    def test_prepared_config_rejects_changed_config_source_and_backend(self):
        config = self.prepared_client("changed", "raise SystemExit(99)")
        with self.assertRaisesRegex(ValueError, "backend"):
            run.prepared_config(config, backend="CSX")
        launch_path = config.parent / "launch.json"
        launch = json.loads(launch_path.read_text())
        launch["source_sha256"] = "old source"
        launch_path.write_text(json.dumps(launch))
        with (
            mock.patch.object(
                run.records,
                "source_identity",
                return_value={"source_sha256": "new source"},
            ),
            self.assertRaisesRegex(ValueError, "source changed"),
        ):
            run.prepared_config(config)
        config.write_text("changed: true\n")
        with self.assertRaisesRegex(ValueError, "params.yaml changed"):
            run.prepared_config(config)
        self.assertFalse((config.parent / "console.log").exists())

    def test_preflight_failure_is_recorded_without_starting_training(self):
        config = self.prepared_client("preflight", "raise SystemExit(99)")
        config.write_text("changed: true\n")
        with (
            mock.patch.object(run.subprocess, "Popen") as spawn,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = run.execute_prepared(config)
        spawn.assert_not_called()
        self.assertEqual(
            (status["state"], status["phase"], status["exit_code"]),
            ("failed", "preflight", 2),
        )
        self.assertIn("params.yaml changed", status["error"])
        self.assertEqual(status["measurement_status"], "unavailable")
        self.assertTrue((config.parent / "result.json").exists())

    def test_changed_data_and_vocabulary_keep_mismatch_evidence(self):
        for changed in ("data", "vocabulary"):
            config = self.prepared_client("changed_" + changed, "raise SystemExit(99)")
            data = config.parent / "data"
            data.mkdir()
            csv = data / "sample.csv"
            vocabulary = config.parent / "vocab.txt"
            csv.write_text("old data")
            vocabulary.write_text("old vocab")
            launch_path = config.parent / "launch.json"
            launch = json.loads(launch_path.read_text())
            launch["dataset_identity"] = run.records.dataset_identity(data, vocabulary)
            launch_path.write_text(json.dumps(launch))
            (csv if changed == "data" else vocabulary).write_text("new content")
            with (
                mock.patch.object(run.subprocess, "Popen") as spawn,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = run.execute_prepared(config)
            spawn.assert_not_called()
            self.assertEqual(status["exit_code"], 2)
            self.assertIn("changed after preparation", status["error"])
            self.assertNotEqual(
                status["observed_dataset_identity"], launch["dataset_identity"]
            )

    def test_external_preflight_failure_preserves_exit_code(self):
        config = self.prepared_client("shell_preflight", "raise SystemExit(99)")
        with (
            mock.patch.object(run.subprocess, "Popen") as spawn,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = run.main(
                [
                    "--backend",
                    "GPU",
                    "--execute-config",
                    str(config),
                    "--preflight-error",
                    "CUDA setup failed",
                    "--preflight-exit-code",
                    "4",
                ]
            )
        spawn.assert_not_called()
        self.assertEqual(code, 4)
        status = json.loads((config.parent / "client_status.json").read_text())
        self.assertEqual(status["error"], "CUDA setup failed")
        self.assertEqual(status["phase"], "preflight")

    def test_client_timeout_preserves_partial_log_and_reaps_process(self):
        config = self.prepared_client(
            "timeout", "import time; print('partial'); time.sleep(60)"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            status = run.execute_prepared(config, timeout_sec=0.2)
        self.assertEqual((status["state"], status["exit_code"]), ("timeout", 124))
        self.assertIn("partial", (config.parent / "console.log").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(status["client_pid"], 0)

    def test_successful_client_cannot_leave_detached_input_workers(self):
        marker = self.path / "orphan-wrote"
        child = f"import time; from pathlib import Path; time.sleep(0.5); Path({str(marker)!r}).write_text('orphan')"
        config = self.prepared_client(
            "orphan",
            "import subprocess,sys; subprocess.Popen([sys.executable, '-c', "
            + repr(child)
            + "], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            status = run.execute_prepared(config)
        self.assertEqual(status["exit_code"], 0)
        time.sleep(0.6)
        self.assertFalse(marker.exists())

    def test_sigterm_stops_client_and_records_interruption(self):
        config = self.prepared_client(
            "interrupt", "import time; print('ready'); time.sleep(60)"
        )
        proc = subprocess.Popen(
            [
                sys.executable,
                str(BENCH / "run.py"),
                "--backend",
                "GPU",
                "--execute-config",
                str(config),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.addCleanup(lambda: proc.kill() if proc.poll() is None else None)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            console = config.parent / "console.log"
            if console.exists() and "ready" in console.read_text():
                break
            if proc.poll() is not None:
                self.fail(proc.communicate()[0])
            time.sleep(0.05)
        else:
            self.fail("client did not start")
        proc.send_signal(signal.SIGTERM)
        output, _ = proc.communicate(timeout=15)
        self.assertEqual(proc.returncode, 143, output)
        status = json.loads((config.parent / "client_status.json").read_text())
        self.assertEqual(status["state"], "interrupted")
        with self.assertRaises(ProcessLookupError):
            os.kill(status["client_pid"], 0)


if __name__ == "__main__":
    unittest.main()
