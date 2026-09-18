"""Content identity and SDK job metadata without data downloads or jobs."""

import json
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
records = importlib.import_module("records")


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_dirty_source_content_changes_identity_without_revision_change(self):
        source = self.root / "src/model.py"
        source.parent.mkdir()
        source.write_text("value = 1\n")
        with (
            patch.object(records, "ROOT", self.root),
            patch.object(records.subprocess, "check_output", return_value="same"),
        ):
            first = records.source_identity()
            source.write_text("value = 2\n")
            second = records.source_identity()
            self.assertEqual(first["git_commit"], second["git_commit"])
            self.assertNotEqual(first["source_sha256"], second["source_sha256"])
            (self.root / "console.log").write_text("new measurement")
            self.assertEqual(second, records.source_identity())

    def test_dataset_content_and_vocabulary_are_independently_identified(self):
        data = self.root / "cache/bert_msl128"
        data.mkdir(parents=True)
        source = data / "part.csv"
        source.write_text("first example")
        (data / "meta.dat").write_text("part.csv 1\n")
        manifest = data.parent / "manifest.json"
        manifest.write_text(json.dumps({"request": {"source": "local-text"}}))
        vocab = self.root / "vocab.txt"
        vocab.write_text("word\n")
        first = records.dataset_identity(data, vocab)
        self.assertEqual(first["manifest"]["sha256"], records.sha256(manifest))
        source.write_text("other example")
        second = records.dataset_identity(data, vocab)
        self.assertNotEqual(first["content_sha256"], second["content_sha256"])
        self.assertEqual(first["vocabulary"], second["vocabulary"])
        vocab.write_text("next\n")
        third = records.dataset_identity(data, vocab)
        self.assertEqual(second["content_sha256"], third["content_sha256"])
        self.assertNotEqual(second["vocabulary"], third["vocabulary"])

    def test_output_cannot_change_its_own_source_identity(self):
        with patch.object(records, "ROOT", self.root):
            for folder in ("src", "src/custom/run", "benchmark_scripts/results"):
                with (
                    self.subTest(folder=folder),
                    self.assertRaisesRegex(ValueError, "outside"),
                ):
                    records.validate_output_dir(self.root / folder)
            records.validate_output_dir(self.root / "model_dirs/new")
            records.validate_output_dir(self.root / "external-shared/new")

    def test_job_labels_validate_with_sdk_and_share_campaign_identity(self):
        from cerebras.appliance.cluster_config import ClusterConfig

        args = SimpleNamespace(
            output_dir=self.root / "実験 with spaces/bert_r01",
            study_dir=self.root / "実験 with spaces",
            profile="bert_large_msl128",
            num_workers=2,
            repeat=1,
        )
        metadata = {"sequence_length": 128, "effective_batch_size": 256}
        first = records.job_labels(args, metadata)
        ClusterConfig(job_labels=first)
        args.repeat = 2
        args.output_dir = args.output_dir.with_name("bert_r02")
        second = records.job_labels(args, metadata)
        ClusterConfig(job_labels=second)
        self.assertIn("run=bert-large-s128-b256-w2-r1", first)
        self.assertIn("run=bert-large-s128-b256-w2-r2", second)
        self.assertEqual(
            [value for value in first if value.startswith("study=")],
            [value for value in second if value.startswith("study=")],
        )


if __name__ == "__main__":
    unittest.main()
