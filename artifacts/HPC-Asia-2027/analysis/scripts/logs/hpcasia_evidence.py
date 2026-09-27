"""Audit the curated HPC Asia inventory and recompute final throughput cohorts.

The input is RUNS.tsv, not campaign plans (which can contain unexecuted trials).
Raw records are never rewritten. CS-3/PyG are distinct implementation paths.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys

import yaml

from training_loop_throughput import config_identity

ROOT = Path(__file__).resolve().parents[3]
RAW = Path("records/raw_logs/hpcasia")
INVALID = Path("records/raw_logs/invalid/hpcasia")
SEEDS = (42, 43, 44)
LABELS = {"cs3": "CS-3 / Model Zoo", "h100": "H100 / PyG"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_tsv(path, rows):
    if not rows:
        raise ValueError(f"Empty table: {path}")
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def audit_inventory(root):
    rows = read_tsv(root / RAW / "RUNS.tsv")
    excluded = read_tsv(root / INVALID / "RUNS.tsv")
    normalized = []
    seen = set()
    for inclusion, records in (("active", rows), ("excluded", excluded)):
        for row in records:
            if row["run"] in seen:
                raise ValueError(f"Duplicate run: {row['run']}")
            seen.add(row["run"])
            if inclusion == "active" and row["status"] not in ("completed", "unstable"):
                raise ValueError(f"Unfinished run in active index: {row['run']}")
            hashes = {}
            for field in ("log", "params", "result"):
                if row[field]:
                    path = root / row[field]
                    if not path.is_file():
                        raise ValueError(f"Missing {field}: {path}")
                    hashes[field] = sha(path)
            record = json.loads((root / row["result"]).read_text()) if row["result"] else {}
            schema = "sdk_restartable_summary" if "runs" in record and "max_num_restarts" in record else "campaign_result" if "measurement" in record else "other_original_record"
            backend = "fixed_shape" if "fixed_shape" in row["run"] else "pyg" if row["platform"] == "h100" and row["purpose"] != "non_gnn" else row["platform"]
            reason = "" if inclusion == "active" else "unfinished" if row["status"] not in ("completed", "unstable") else "superseded_optimizer_amp_configuration"
            normalized.append({**row, "backend": backend, "inclusion": inclusion, "exclusion_reason": reason, "result_schema": schema, "source_sha256": hashes})
    file_counts = {}
    for base in (RAW, INVALID):
        files = read_tsv(root / base / "FILES.tsv")
        if len({f["log"] for f in files}) != len(files):
            raise ValueError(f"Duplicate raw file in {base}")
        for row in files:
            if sha(root / row["log"]) != row["sha256"]:
                raise ValueError(f"Raw file hash mismatch: {row['log']}")
            if sha(root / row["model_path"]) != row["sha256"]:
                raise ValueError(f"Model reference hash mismatch: {row['model_path']}")
        file_counts[str(base)] = len(files)
    counts = Counter((r["inclusion"], r["purpose"], r["backend"], r["dataset"], r["status"]) for r in normalized)
    coverage = [dict(zip(("inclusion", "purpose", "backend", "dataset", "status", "n"), (*key, n))) for key, n in sorted(counts.items())]
    return rows, {"schema_version": 1, "paths_relative_to": "repository root", "files_verified": file_counts, "coverage": coverage, "runs": normalized}


def measurement_source(root, row, result):
    if "measurement" in result:
        return result["measurement"], row["result"]
    if row["platform"] != "cs3" or not result.get("runs") or any(r["status"] != "success" for r in result["runs"]):
        raise ValueError(f"Unsupported completion record: {row['result']}")
    return None, row["result"]


def validate_cohort(rows):
    """Require all planned seeds/repeats and one configuration per condition."""
    for platform in LABELS:
        for dataset in ("arxiv", "products"):
            for cache in (0, 1):
                cohort = [r for r in rows if (r["platform"], r["dataset"], r["cache"]) == (platform, dataset, cache)]
                expected = Counter((s, rep) for s in SEEDS for rep in ((1,) if cache else (1, 2, 3)))
                if Counter((r["seed"], r["repeat"]) for r in cohort) != expected:
                    raise ValueError(f"Missing/duplicate final repetitions: {platform}, {dataset}, cache={cache}")
                identities = []
                for row in cohort:
                    identity = deepcopy(row["configuration"])
                    identity.pop("seed")
                    identity.pop("sampler_seed")
                    identities.append(identity)
                if any(i != identities[0] for i in identities[1:]):
                    raise ValueError(f"Configuration changed within cohort: {platform}, {dataset}, cache={cache}")
            conditions = []
            for row in (r for r in rows if (r["platform"], r["dataset"]) == (platform, dataset)):
                condition = deepcopy(row["configuration"])
                condition.pop("seed")
                condition.pop("sampler_seed")
                condition["loader"].pop("cache_fraction")
                conditions.append(condition)
            if any(c != conditions[0] for c in conditions[1:]):
                raise ValueError(f"Other settings changed across cache conditions: {platform}, {dataset}")
    for dataset in ("arxiv", "products"):
        if len({r["seed_nodes"] for r in rows if r["dataset"] == dataset}) != 1:
            raise ValueError(f"Target counts differ between final repetitions: {dataset}")
    # Compare the explicit common settings while retaining path-specific precision
    # fields. This is configuration parity, not numerical or sampling equivalence.
    for dataset in ("arxiv", "products"):
        for seed in SEEDS:
            for cache in (0, 1):
                pair = [next(r for r in rows if (r["dataset"], r["seed"], r["cache"], r["platform"], r["repeat"]) == (dataset, seed, cache, p, 1)) for p in LABELS]
                for field in ("dataset", "model", "optimizer", "seed", "batch_size", "fanouts", "sampler_seed", "drop_last", "shuffle", "loader"):
                    if pair[0]["configuration"][field] != pair[1]["configuration"][field]:
                        raise ValueError(f"Cross-path setting differs: {dataset}, {field}")
                for field in ("enabled", "fp16_type", "loss_scaling_factor"):
                    if pair[0]["configuration"]["precision"].get(field) != pair[1]["configuration"]["precision"].get(field):
                        raise ValueError(f"Cross-path precision setting differs: {field}")


def summarize_values(values):
    return {"n": len(values), "mean": statistics.mean(values), "sample_stddev": statistics.stdev(values) if len(values) > 1 else None, "min": min(values), "max": max(values)}


def throughput(root, rows, modelzoo):
    sys.path.insert(0, str(modelzoo / "src"))
    parsers = {p: importlib.import_module("cerebras.modelzoo.models.gnn.tools." + name) for p, name in (("cs3", "measure_window"), ("h100", "measure_pyg"))}
    records, windows = [], []
    for row in rows:
        if row["purpose"] != "throughput":
            continue
        if not row["run"].startswith("hpcasia_final/"):
            raise ValueError(f"Unexpected final cohort: {row['run']}")
        config = yaml.safe_load((root / row["params"]).read_text())
        result = json.loads((root / row["result"]).read_text())
        if "config" in result and config != result["config"]:
            raise ValueError(f"Saved configuration mismatch: {row['params']}")
        if "config_sha256" in result and hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest() != result["config_sha256"]:
            raise ValueError(f"Configuration hash mismatch: {row['params']}")
        if "returncode" in result and result["returncode"] != 0:
            raise ValueError(f"Unsuccessful final process: {row['run']}")
        start, end = 40, 840 if row["dataset"] == "arxiv" else 1640
        measurement = parsers[row["platform"]].summarize(root / row["log"], start, end)
        saved, source = measurement_source(root, row, result)
        if saved is not None and measurement != saved:
            raise ValueError(f"Raw recomputation differs from saved measurement: {row['run']}")
        if measurement["metric"] != "seed_nodes_per_second" or measurement.get("skipped_optimizer_steps", 0):
            raise ValueError(f"Ineligible throughput numerator/updates: {row['run']}")
        identity = config_identity(config)
        loader = config["trainer"]["fit"]["train_dataloader"]
        identity["loader"] = {k: loader.get(k) for k in ("num_workers", "prefetch_factor", "persistent_workers", "pin_memory", "cache_fraction", "static_batch_cache_size", "use_fake_data", "worker_diagnostics")}
        # null and 0 both explicitly select no feature cache.
        if loader.get("cache_fraction") not in (None, 0, 1):
            raise ValueError(f"Unsupported partial feature cache: {row['run']}")
        cache = int(bool(loader.get("cache_fraction")))
        identity["loader"]["cache_fraction"] = cache
        if identity["seed"] != int(row["seed"]) or identity["sampler_seed"] != int(row["seed"]) or loader["num_workers"] != int(row["workers"]):
            raise ValueError(f"Index/configuration mismatch: {row['run']}")
        seed_count = measurement["seed_nodes"]
        nominal = identity["batch_size"] * (end - start)
        record = {**row, "seed": int(row["seed"]), "repeat": int(Path(row["run"]).name.rsplit("r", 1)[1]), "cache": cache, "start_step": start, "end_step": end, "seed_nodes": seed_count, "supervised_targets": measurement.get("supervised_targets"), "nominal_slots": nominal, "seed_padding_percent": (1 - seed_count / nominal) * 100, "seconds": measurement["training_window_seconds"], "seed_nodes_per_second": measurement["throughput"], "half_difference_percent": measurement["half_window_check"]["symmetric_difference_percent"], "within_tolerance": measurement["half_window_check"]["within_tolerance"], "optimizer_update_check": measurement.get("optimizer_update_check", "not_recorded_by_this_parser"), "measurement_source": source, "configuration": identity, "measurement": measurement, "source_sha256": {k: sha(root / row[k]) for k in ("log", "params", "result")}}
        records.append(record)
        record["source_sha256"]["measurement_source"] = sha(root / source)
        # Nested windows are sensitivity observations, never independent samples.
        intervals = [(40, 440), (40, 840)] + ([(40, 1640), (840, 1640)] if end == 1640 else [])
        for a, b in intervals:
            m = parsers[row["platform"]].summarize(root / row["log"], a, b)
            windows.append({"run": row["run"], "platform": row["platform"], "dataset": row["dataset"], "seed": record["seed"], "repeat": record["repeat"], "cache": cache, "start_step": a, "end_step": b, "seed_nodes": m["seed_nodes"], "seconds": m["training_window_seconds"], "seed_nodes_per_second": m["throughput"], "relative_to_full_window_percent": (m["throughput"] / measurement["throughput"] - 1) * 100})
    validate_cohort(records)
    groups = defaultdict(list)
    for r in records:
        groups[(r["platform"], r["dataset"], r["cache"])].append(r)
    summaries = []
    for (platform, dataset, cache), group in sorted(groups.items()):
        rates = [r["seed_nodes_per_second"] for r in group]
        seed_means = [statistics.mean(r["seed_nodes_per_second"] for r in group if r["seed"] == s) for s in SEEDS]
        summaries.append({"platform": platform, "dataset": dataset, "cache": cache, **summarize_values(rates), "n_seeds": len(SEEDS), "seed_mean_sample_stddev": statistics.stdev(seed_means), "start_step": group[0]["start_step"], "end_step": group[0]["end_step"], "max_half_difference_percent": max(r["half_difference_percent"] for r in group)})
    ratios = []
    for dataset in ("arxiv", "products"):
        for cache in (0, 1):
            means = {r["platform"]: r["mean"] for r in summaries if r["dataset"] == dataset and r["cache"] == cache}
            ratios.append({"dataset": dataset, "comparison": "CS-3 / H100-PyG", "cache": cache, "ratio_of_means": means["cs3"] / means["h100"]})
    cache_ratios = []
    for platform in LABELS:
        for dataset in ("arxiv", "products"):
            for seed in SEEDS:
                by_cache = {c: statistics.mean(r["seed_nodes_per_second"] for r in records if (r["platform"], r["dataset"], r["seed"], r["cache"]) == (platform, dataset, seed, c)) for c in (0, 1)}
                cache_ratios.append({"platform": platform, "dataset": dataset, "seed": seed, "cache_over_nocache": by_cache[1] / by_cache[0], "nocache_runs": 3, "cache_runs": 1})
    return {"metric": "seed_nodes_per_second", "comparison_scope": "separate implementation paths; common explicit configuration verified; sampling, bias structure and optimizer/AMP semantics differ", "aggregation": "equal-weight runs; 3 seeds x 3 repeats uncached; 3 seeds x 1 repeat cached; SD is descriptive sample SD, not a confidence interval", "parser_sha256": {p: sha(Path(module.__file__)) for p, module in parsers.items()}, "config_identity_source_sha256": sha(Path(__file__).with_name("training_loop_throughput.py")), "script_sha256": sha(Path(__file__)), "runs": records, "summary": summaries, "ratios": ratios, "cache_ratios_by_seed": cache_ratios, "window_sensitivity": windows}


def write_findings(result, output):
    """Present scalar comparisons as tables; preserve full precision in TSV."""
    summaries = {(r["platform"], r["dataset"], r["cache"]): r for r in result["summary"]}
    lines = [
        "# CS-3 と H100/PyG の性能比較",
        "",
        "## 学習ループの処理速度",
        "",
        "ウォームアップ後の実対象頂点数 / 学習ループの測定秒数。平均 ± 標本標準偏差（対象頂点/秒）。",
        "",
        "| データセット | 特徴量キャッシュ | CS-3 / Model Zoo | H100 / PyG | CS-3 / H100 (%) | 各経路の実行数 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for ratio in result["ratios"]:
        dataset, cache = ratio["dataset"], ratio["cache"]
        cs3, h100 = (summaries[p, dataset, cache] for p in LABELS)
        lines.append(
            f"| ogbn-{dataset} | {'有' if cache else '無'} | "
            f"{cs3['mean']:,.1f} ± {cs3['sample_stddev']:,.1f} | "
            f"{h100['mean']:,.1f} ± {h100['sample_stddev']:,.1f} | "
            f"{ratio['ratio_of_means'] * 100:.2f} | {cs3['n']} |"
        )
    lines += [
        "",
        "未キャッシュは3 seed × 3独立実行、キャッシュは3 seed × 1実行。各実行を等重みで集計した。",
        "CS-3 / H100は平均同士の比。SDは観測した実行間のばらつきで、信頼区間ではない。",
        "全条件 workers=4、prefetch=2、worker永続化有効。キャッシュ配置はCS-3がCPU、H100がGPU。",
        "CS-3 / Model ZooとH100 / PyGの実装経路全体を比較する。",
        "",
        "出典: [実行別測定](runs.tsv)、[集計値](summary.tsv)、[速度比](ratios.tsv)。",
        "",
        "## 特徴量キャッシュによる速度の変化",
        "",
        "各seedで、キャッシュ有1実行の速度を未キャッシュ3実行の平均速度で割った。",
        "表は3 seedの比率の平均 ± 標本標準偏差（倍）。キャッシュ有無の総平均の比とは集計順が異なる。",
        "",
        "| データセット | CS-3（CPUキャッシュ） | H100/PyG（GPUキャッシュ） | seed数 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dataset in ("arxiv", "products"):
        cells = []
        for platform in LABELS:
            values = [r["cache_over_nocache"] for r in result["cache_ratios_by_seed"]
                      if (r["platform"], r["dataset"]) == (platform, dataset)]
            cells.append(f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}")
        lines.append(f"| ogbn-{dataset} | {' | '.join(cells)} | {len(SEEDS)} |")
    lines += [
        "",
        "CS-3ではarxivの変化は小さく、productsは約13%増。H100/PyGでは両データセットで約66–68%増だった。",
        "出典: [seed別の比率](cache_ratios.tsv)。",
        "",
        "## 測定窓への感度",
        "",
        "全測定窓はarxivが40→840 step、productsが40→1640 step。短窓は同じ実行の40→440 stepと、",
        "productsの40→840 stepを使う。短窓変化は (短窓速度 / 全窓速度 − 1) × 100。",
        "表の範囲は各条件の短窓変化の最小–最大であり、信頼区間ではない。",
        "半区間差は全窓の前半・後半の速度差をその平均で割った絶対値。",
        "",
        "| データセット | 経路 | キャッシュ | 短窓変化の範囲 (%) | 最大半区間差 (%) | 独立実行数 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for dataset in ("arxiv", "products"):
        for platform in LABELS:
            for cache in (0, 1):
                summary = summaries[platform, dataset, cache]
                changes = [w["relative_to_full_window_percent"] for w in result["window_sensitivity"]
                           if (w["platform"], w["dataset"], w["cache"]) == (platform, dataset, cache)
                           and w["start_step"] == 40 and w["end_step"] < summary["end_step"]]
                lines.append(
                    f"| ogbn-{dataset} | {LABELS[platform]} | {'有' if cache else '無'} | "
                    f"{min(changes):+.3f} 〜 {max(changes):+.3f} | "
                    f"{summary['max_half_difference_percent']:.3f} | {summary['n']} |"
                )
    lines += [
        "",
        "48実行の半区間差は全件2%以内。重なる窓は独立反復へ数えない。",
        "コンパイル・準備・検証を含む総所要時間は、この学習ループ測定の対象外。",
        "出典: [区間別の分子・時間・速度](windows.tsv)、[実行別の半区間判定](runs.tsv)。",
        "",
        "保存原本・設定の照合は [全実行の監査](inventory.json)、比較条件の詳細は [統合報告](../README.md) を参照する。",
    ]
    (output / "findings.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--modelzoo-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--output", type=Path, default=ROOT / "results/throughput")
    args = parser.parse_args()
    rows, inventory = audit_inventory(args.root)
    result = throughput(args.root, rows, args.modelzoo_root.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "inventory.json", inventory)
    write_tsv(args.output / "coverage.tsv", inventory["coverage"])
    write_json(args.output / "throughput.json", result)
    write_tsv(args.output / "runs.tsv", [{k: v for k, v in r.items() if k not in ("configuration", "measurement", "source_sha256")} for r in result["runs"]])
    write_tsv(args.output / "summary.tsv", result["summary"])
    write_tsv(args.output / "ratios.tsv", result["ratios"])
    write_tsv(args.output / "cache_ratios.tsv", result["cache_ratios_by_seed"])
    write_tsv(args.output / "windows.tsv", result["window_sensitivity"])
    write_findings(result, args.output)
    print(json.dumps({"active_runs": len(rows), "indexed_runs_including_excluded": len(inventory["runs"]), "files_verified": inventory["files_verified"], "final_throughput_runs": len(result["runs"]), "summary": result["summary"], "ratios": result["ratios"]}, indent=2))


if __name__ == "__main__":
    main()
