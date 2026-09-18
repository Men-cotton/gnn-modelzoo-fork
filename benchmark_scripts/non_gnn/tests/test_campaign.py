"""Offline campaign tests: real data processors, fake tokenizer and scheduler."""

import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
import shutil
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
        with mock.patch.object(
            prepare_data,
            "load_llama_tokenizer",
            return_value=(FakeTokenizer(), {"source": "test"}),
        ), mock.patch.object(
            prepare_data,
            "acquire_documents",
            return_value=(self.documents, {"kind": "test"}),
        ), contextlib.redirect_stdout(
            io.StringIO()
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
        with mock.patch.object(
            run.shutil, "which", return_value=None
        ), mock.patch.object(prepare_data, "prepare") as prepare, mock.patch.object(
            campaign, "start_csx"
        ) as start, contextlib.redirect_stderr(
            io.StringIO()
        ) as errors:
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
            with mock.patch.object(
                campaign.shutil, "which", return_value="qsub"
            ), mock.patch.object(
                run.importlib,
                "import_module",
                side_effect=ModuleNotFoundError("No module named 'datasets'"),
            ), mock.patch.object(
                prepare_data, "prepare"
            ) as prepare, mock.patch.object(
                campaign, "submit_gpu"
            ) as submit, mock.patch.object(
                campaign, "start_csx"
            ) as start, contextlib.redirect_stderr(
                errors
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
        with mock.patch.object(
            prepare_data,
            "acquire_documents",
            side_effect=AssertionError("download called"),
        ), mock.patch.object(
            prepare_data,
            "load_llama_tokenizer",
            side_effect=AssertionError("tokenizer called"),
        ), mock.patch.object(
            prepare_data,
            "write_bert",
            side_effect=AssertionError("BERT preprocessing called"),
        ), mock.patch.object(
            prepare_data,
            "write_llama",
            side_effect=AssertionError("Llama preprocessing called"),
        ), contextlib.redirect_stdout(
            io.StringIO()
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
                if job["name"] == name + "_native"
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
                result.stdout.count("--config") + result.stdout.count("qsub -v"), 4
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
        with mock.patch.dict(
            os.environ, {"PATH": str(self.path) + os.pathsep + os.environ["PATH"]}
        ), mock.patch.object(
            prepare_data,
            "acquire_documents",
            side_effect=AssertionError("download called"),
        ), mock.patch.object(
            prepare_data,
            "load_llama_tokenizer",
            side_effect=AssertionError("tokenizer called"),
        ), contextlib.redirect_stdout(
            io.StringIO()
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
            self.assertEqual(launch["gpu_implementation"], job["name"].split("_")[-1])
        # CSX uses exactly the same cache, without any source/tokenizer lookup.
        with mock.patch.object(
            prepare_data,
            "acquire_documents",
            side_effect=AssertionError("download called"),
        ), mock.patch.object(
            prepare_data,
            "load_llama_tokenizer",
            side_effect=AssertionError("tokenizer called"),
        ), mock.patch.object(
            campaign, "start_csx"
        ) as launch_csx, contextlib.redirect_stdout(
            io.StringIO()
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

        with mock.patch.object(
            campaign.shutil, "which", return_value="qsub"
        ), mock.patch.object(run, "main", side_effect=fail_second), mock.patch.object(
            campaign, "submit_gpu"
        ) as submit, contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            io.StringIO()
        ):
            with self.assertRaises(SystemExit):
                campaign.main(self.argv())
        submit.assert_not_called()
        self.assertEqual(calls, 2)

    def test_csx_client_records_success_and_failure(self):
        for code in (0, 7):
            output = self.path / f"client_{code}"
            output.mkdir()
            config = output / "params.yaml"
            config.write_text("test: true\n")
            (output / "launch.json").write_text(
                json.dumps(
                    {
                        "backend": "CSX",
                        "params_sha256": hashlib.sha256(
                            config.read_bytes()
                        ).hexdigest(),
                        "command": [
                            sys.executable,
                            "-c",
                            "import shutil; assert shutil.which('torch-cirh-opt'); "
                            f"print('client output'); raise SystemExit({code})",
                        ],
                    }
                )
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(BENCH / "execute_csx.py"),
                    "--config",
                    str(config),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, code, result.stderr)
            status = json.loads((output / "client_status.json").read_text())
            self.assertEqual(status["state"], "completed" if code == 0 else "failed")
            self.assertEqual(status["exit_code"], code)
            self.assertIn("client output", (output / "console.log").read_text())


if __name__ == "__main__":
    unittest.main()
