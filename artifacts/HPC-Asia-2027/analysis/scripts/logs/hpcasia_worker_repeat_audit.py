"""Audit all saved, compatible arxiv worker runs without changing the R04 subset.

The inclusion rule depends on configuration, completion, and input identity,
never on the measured rate. Also report every leave-one-out subset of the
included w8 runs, solely as a sensitivity check.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import yaml

import hpcasia_input_analysis as input_analysis


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = "hpcasia_final/arxiv_20260919T033125.784128Z/seed_42/throughput_r1"
WORKERS = (4, 8, 12, 16)
ALLOWED = {
    "trainer.init.model_dir",
    "trainer.init.backend.cluster_config.job_labels",
    "trainer.init.loop.max_steps",
    "trainer.fit.train_dataloader.num_workers",
}
SOURCE_FILES = [
    "src/cerebras/modelzoo/models/gnn/architectures/graphsage.py",
    "src/cerebras/modelzoo/models/gnn/task/loss.py",
    "src/cerebras/modelzoo/models/gnn/task/wrapper.py",
    "src/cerebras/modelzoo/models/gnn/data_processing/samplers/neighbor_tree.py",
    "src/cerebras/modelzoo/models/gnn/data_processing/batches.py",
]


def read_tsv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {key: leaf for name, child in value.items()
                for key, leaf in flatten(child, f"{prefix}.{name}".strip(".")).items()}
    return {prefix: value}


def write_tsv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                minimum=min(values), maximum=max(values))


def provenance(run_dir):
    for directory in (run_dir.parent, run_dir.parent.parent):
        for name in ("study.json", "campaign.json", "learning_campaign.json"):
            path = directory / name
            if path.exists():
                value = json.loads(path.read_text())
                if "environment" in value:
                    return path, value["environment"]
    raise ValueError(f"No source environment found for {run_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelzoo-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/worker_repeat_review")
    args = parser.parse_args()
    modelzoo = args.modelzoo_root.resolve()
    sys.path.insert(0, str(modelzoo / "src"))
    measure = importlib.import_module("cerebras.modelzoo.models.gnn.tools.measure_window")
    index = ROOT / "records/raw_logs/hpcasia/RUNS.tsv"
    old_index = ROOT / "records/raw_logs/hpcasia/input_pipeline/R04.tsv"
    rows = read_tsv(index)
    previous = read_tsv(old_index)
    old_ids = {row["run"] for row in previous}
    reference = next(row for row in rows if row["run"] == REFERENCE)
    reference_cfg = yaml.safe_load((ROOT / reference["params"]).read_text())
    reference_identity = input_analysis.config_identity(reference_cfg)
    reference_flat = flatten(reference_cfg)
    ref_measurement = measure.summarize(ROOT / reference["log"], 40, 440)
    expected_hash = ref_measurement["input_contract"]["ordered_targets_and_labels_sha256"]
    included, excluded, differences = [], [], []
    source_hashes = {str(p.relative_to(ROOT)): sha(p)
                     for p in (index, old_index, Path(__file__), Path(input_analysis.__file__))}
    versions = {}
    for row in rows:
        if (row["platform"] != "cs3" or row["dataset"] != "arxiv"
                or row["purpose"] not in ("input_pipeline", "throughput")
                or int(row["workers"]) not in WORKERS):
            continue
        paths = {key: ROOT / row[key] for key in ("params", "log", "result")}
        cfg = yaml.safe_load(paths["params"].read_text())
        flat = flatten(cfg)
        changed = sorted(key for key in flat.keys() | reference_flat.keys()
                         if flat.get(key) != reference_flat.get(key))
        forbidden = sorted(set(changed) - ALLOWED)
        if input_analysis.config_identity(cfg) != reference_identity:
            excluded.append(dict(run=row["run"], reason="configuration differs",
                                 fields=json.dumps(forbidden)))
            continue
        if forbidden:
            raise ValueError(f"Unexpected normalized configuration mismatch: {row['run']}")
        result = json.loads(paths["result"].read_text())
        if result["status"] not in ("completed", "unstable") or result["returncode"] != 0:
            excluded.append(dict(run=row["run"], reason="not completed", fields=""))
            continue
        measured = measure.summarize(paths["log"], 40, 440)
        contract = measured["input_contract"]
        if contract["ordered_targets_and_labels_sha256"] != expected_hash:
            excluded.append(dict(run=row["run"], reason="different first-pass input order", fields=""))
            continue
        # Compare the entire archived input contract, not only its order hash.
        if contract != ref_measurement["input_contract"]:
            raise ValueError(f"Input contract differs: {row['run']}")
        environment_path, environment = provenance(paths["params"].parent)
        revision = environment["git_commit"]
        if revision not in versions:
            versions[revision] = {
                name: sha(ROOT / "sources" / revision / name)
                for name in SOURCE_FILES
            }
        for path in (*paths.values(), environment_path):
            source_hashes[str(path.relative_to(ROOT))] = sha(path)
        snapshot = environment_path.parent / "source"
        if snapshot.is_dir():
            for name in SOURCE_FILES:
                path = snapshot / name
                source_hashes[str(path.relative_to(ROOT))] = sha(path)
                if sha(path) != versions[revision][name]:
                    raise ValueError(f"Archived source differs from recorded revision: {path}")
        for field in changed:
            differences.append(dict(run=row["run"], field=field,
                                    reference=json.dumps(reference_flat.get(field)),
                                    value=json.dumps(flat.get(field))))
        item = dict(row, workers=int(row["workers"]), phase=result.get("phase", "final_throughput"),
                    prior_r04_subset=row["run"] in old_ids, started_at=result.get("started_at", ""),
                    source_revision=revision, environment=str(environment_path.relative_to(ROOT)),
                    max_steps=cfg["trainer"]["init"]["loop"]["max_steps"],
                    start_step=40, end_step=440, seed_nodes=measured["seed_nodes"],
                    supervised_targets=measured["supervised_targets"], nominal_slots=measured["nominal_slots"],
                    seconds=measured["training_window_seconds"], targets_per_second=measured["throughput"],
                    half_difference_percent=measured["half_window_check"]["symmetric_difference_percent"],
                    input_order_sha256=expected_hash)
        included.append(item)
    if not old_ids <= {r["run"] for r in included}:
        raise ValueError("The original R04 subset is not fully contained in the compatible cohort")
    if any(hashes != next(iter(versions.values())) for hashes in versions.values()):
        raise ValueError("Recorded model, loss, sampler or batch source differs within the cohort")
    summaries = []
    for name in ("all_compatible", "previous_r04", "coarse_workers_only"):
        for workers in WORKERS:
            group = [r for r in included if r["workers"] == workers
                     and (name == "all_compatible" or name == "previous_r04" and r["prior_r04_subset"]
                          or name == "coarse_workers_only" and r["phase"] == "workers")]
            summaries.append(dict(cohort=name, workers=workers,
                                  **describe([r["targets_per_second"] for r in group])))
    w8 = [r for r in included if r["workers"] == 8]
    sensitivity = []
    for omit in [None, *w8]:
        group = [r for r in w8 if r is not omit]
        item = dict(omitted_run=omit["run"] if omit else "", **describe([r["targets_per_second"] for r in group]))
        sensitivity.append(item)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, records in (("runs", included), ("excluded", excluded), ("configuration_differences", differences),
                          ("summary", summaries), ("w8_sensitivity", sensitivity)):
        write_tsv(args.output / f"{name}.tsv", records)
    report = dict(
        inclusion="All active completed arxiv CS-3 input/throughput runs at workers 4/8/12/16 with the reference configuration, full input contract, and common model/loss/sampler/batch source; only worker count, total step budget, output and job labels may differ",
        reference_run=REFERENCE, measured_window=[40, 440],
        counts=dict(sorted(Counter(r["workers"] for r in included).items())),
        run_count=len(included),
        source_revisions=versions, source_sha256=source_hashes,
        measurement_parser_sha256=sha(Path(measure.__file__)),
        summaries=summaries, w8_sensitivity=sensitivity,
        limitations=["Exploratory, adaptively selected loader/confirmation, and final runs are mixed; descriptive comparisons only",
                     "Leave-one-out results are sensitivity checks, not an outcome-based exclusion rule",
                     "Newly collected per-pass reshuffle runs would form a different sampler cohort"],
    )
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_count": len(included), "counts": report["counts"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
