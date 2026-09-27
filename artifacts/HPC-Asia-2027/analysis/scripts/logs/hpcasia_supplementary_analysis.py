"""Recompute bounded non-GNN context and audit CS-3 SDK executor records.

Run with the repository environment::

    .venv/bin/python analyze/scripts/logs/hpcasia_supplementary_analysis.py

The output never interprets SDK FLOPs utilization as a compute-time fraction.
Preparation intervals are kept separate; their nesting precludes addition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from compare_non_gnn_20260918 import compare


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "records/raw_logs/hpcasia"
SDK_INDEX = ROOT / "records/model_dirs/hpcasia_final/SDK_FILES.tsv"
DEFAULT_OUTPUT = ROOT / "results/supplementary"


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def duration(track: dict, name: str):
    item = track.get(name)
    if item is None:
        return None
    value = float(item[0])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {name} duration: {value}")
    return value


def runtime_records() -> tuple[list[dict], list[dict], dict]:
    """Associate each executor with its source run and SDK stage metadata."""
    with SDK_INDEX.open() as stream:
        files = list(csv.DictReader(stream, delimiter="\t"))
    with (RAW / "RUNS.tsv").open() as stream:
        runs = {row["run"]: row for row in csv.DictReader(stream, delimiter="\t")}
    by_path = {row["path"]: row for row in files}
    if len(by_path) != len(files):
        raise ValueError("Duplicate paths in SDK_FILES.tsv")
    records = []
    for indexed in files:
        performance_path = ROOT / indexed["path"]
        if performance_path.name != "performance.json":
            continue
        track_path = performance_path.with_name("track.json")
        track_index = by_path[relative(track_path)]
        for path, entry in ((performance_path, indexed), (track_path, track_index)):
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Source hash mismatch: {relative(path)}")
        run_id = indexed["run"]
        source_run = runs[run_id]
        model_run = (ROOT / source_run["params"]).parent
        if track_index["run"] != run_id or not performance_path.is_relative_to(model_run):
            raise ValueError(f"SDK/run association mismatch: {run_id}")
        metadata_path = performance_path.with_name("metadata.json")
        metadata = read_json(metadata_path)
        performance, track = read_json(performance_path), read_json(track_path)
        stamp = performance_path.parents[2].name
        executor = performance_path.parent.name
        dataset, seed, trial = source_run["dataset"], source_run["seed"], model_run.name
        if source_run["platform"] != "cs3":
            raise ValueError(f"Runtime/run association mismatch: {run_id}")
        log_path = ROOT / source_run["log"]
        log_text = log_path.read_text()
        endpoint_seconds = None
        if source_run["purpose"] in ("throughput", "reshuffle_throughput"):
            matches = re.findall(
                r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Train Device=CSX, Step=(\d+),",
                log_text, re.MULTILINE,
            )
            times = {int(step): datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f")
                     for stamp, step in matches}
            end_step = 840 if dataset == "arxiv" else 1640
            endpoint_seconds = (times[end_step] - times[40]).total_seconds()
        stage = metadata["stage"]
        if stage not in ("train", "validate"):
            raise ValueError(f"Unrecognized SDK stage: {stage}")
        steps = int(metadata["num_steps"])
        samples = float(performance["total_samples"])
        seconds = float(performance["total_time"])
        if steps <= 0 or seconds <= 0 or not math.isfinite(seconds):
            raise ValueError(f"Invalid runtime counters: {run_id}/{executor}")
        # These SDK samples are reserved batch slots, including padded positions.
        if samples != steps * 4096:
            raise ValueError(f"Unexpected SDK sample accounting: {run_id}/{executor}")
        reported_rate = float(performance["samples_per_sec"])
        if not math.isfinite(reported_rate) or reported_rate <= 0:
            raise ValueError(f"Invalid SDK reported rate: {run_id}/{executor}")
        cumulative_rate = samples / seconds
        records.append({
            "run": run_id,
            "purpose": source_run["purpose"],
            "dataset": dataset,
            "seed": int(seed),
            "condition": "cache" if trial.startswith("cache_") else (
                "no_cache" if trial.startswith("throughput_") else "learning"
            ),
            "sdk_timestamp": stamp,
            "executor": executor,
            "stage": stage,
            "num_steps": steps,
            "sdk_total_nominal_slots": samples,
            "sdk_total_time_seconds": seconds,
            "sdk_cumulative_nominal_slots_per_second": cumulative_rate,
            "sdk_samples_per_sec_reported": reported_rate,
            "sdk_rate_relative_difference_percent": 100 * (reported_rate / cumulative_rate - 1),
            "post_warmup_log_endpoint_seconds": endpoint_seconds,
            "image_build_fallback_logged": "Falling back to venv mounting." in log_text,
            "sdk_flops_utilization_raw": performance.get("flops_utilization"),
            "initialization_seconds": duration(track, "Initialization"),
            "compile_record_seconds": duration(track, "compile"),
            "execute_till_recv_loss_seconds": duration(track, "execute_till_recv_loss"),
            "performance_path": relative(performance_path),
            "performance_sha256": indexed["sha256"],
            "track_path": relative(track_path),
            "track_sha256": track_index["sha256"],
            "metadata_path": relative(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "train_log_path": relative(log_path),
            "train_log_sha256": sha256(log_path),
        })
    records.sort(key=lambda row: (row["run"], row["sdk_timestamp"], row["executor"]))
    track_count = sum(item["path"].endswith("/track.json") for item in files)
    if track_count != len(records):
        raise ValueError("Unpaired runtime performance/track entries")
    final = [row for row in records if row["purpose"] == "throughput"]
    counts = Counter(row["run"] for row in final)
    expected = {key for key, run in runs.items() if run["purpose"] == "throughput" and run["platform"] == "cs3"}
    if set(counts) != expected or any(value != 1 for value in counts.values()):
        raise ValueError("Standalone final throughput run/executor coverage mismatch")
    if any(row["stage"] != "train" for row in final):
        raise ValueError("Evaluation executor in throughput-only run")
    groups = defaultdict(list)
    for row in final:
        groups[(row["dataset"], row["condition"])].append(row)
    summary = []
    for (dataset, condition), group in sorted(groups.items()):
        result = {"dataset": dataset, "condition": condition, "independent_runs": len(group)}
        for name in ("sdk_total_time_seconds", "sdk_cumulative_nominal_slots_per_second",
                     "initialization_seconds", "compile_record_seconds", "execute_till_recv_loss_seconds"):
            values = [row[name] for row in group if row[name] is not None]
            result[f"{name}_n"] = len(values)
            for label, function in (("mean", mean), ("min", min), ("max", max)):
                result[f"{name}_{label}"] = function(values) if values else None
        summary.append(result)
    coverage = {
        "paired_executors": len(records),
        "source_runs": len({row["run"] for row in records}),
        "stages": dict(Counter(row["stage"] for row in records)),
        "learning_train_executors": sum(row["purpose"] == "learning" and row["stage"] == "train" for row in records),
        "reshuffle_train_executors": sum(row["purpose"] == "reshuffle" and row["stage"] == "train" for row in records),
        "reshuffle_throughput_executors": sum(row["purpose"] == "reshuffle_throughput" and row["stage"] == "train" for row in records),
        "standalone_throughput_executors": len(final),
        "max_sdk_rate_relative_difference_percent": max(
            abs(row["sdk_rate_relative_difference_percent"]) for row in records
        ),
        "missing_metadata": 0,
        "sdk_files_index": relative(SDK_INDEX),
        "sdk_files_index_sha256": sha256(SDK_INDEX),
        "runs_index_sha256": sha256(RAW / "RUNS.tsv"),
    }
    return records, summary, coverage


def summary_tables(non_gnn: dict, records: list[dict]) -> tuple[str, str]:
    """Present exact comparisons and timing definitions without redundant figures."""
    non_gnn_lines = [
        "| モデル | 系列長 | 実効バッチ | CS-3（サンプル/秒） | H100（サンプル/秒） | CS-3/H100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in non_gnn["rows"]:
        model = "BERT large" if row["profile"].startswith("bert_") else "Llama 3.2 1B"
        non_gnn_lines.append(
            f"| {model} | {row['sequence_length']} | {row['effective_batch_size']} | "
            f"{row['cs3_samples_per_second']:,.1f} | {row['h100_samples_per_second']:,.1f} | "
            f"{row['cs3_over_h100']:.2f} |"
        )
    runtime_lines = [
        "| データセット | 特徴量キャッシュ | 実行数 | SDK累積時間（秒） | 主比較の計時端点間（秒） |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in ("arxiv", "products"):
        for condition, label in (("no_cache", "なし"), ("cache", "ホスト")):
            group = [row for row in records if row["purpose"] == "throughput"
                     and row["dataset"] == dataset and row["condition"] == condition]
            intervals = []
            for metric in ("sdk_total_time_seconds", "post_warmup_log_endpoint_seconds"):
                values = [row[metric] for row in group]
                intervals.append(f"{mean(values):,.1f}（{min(values):,.1f}–{max(values):,.1f}）")
            runtime_lines.append(
                f"| ogbn-{dataset} | {label} | {len(group)} | " + " | ".join(intervals) + " |"
            )
    return "\n".join(non_gnn_lines), "\n".join(runtime_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    non_gnn = compare()
    records, summary, coverage = runtime_records()
    document = {
        "schema_version": 1,
        "source_paths_relative_to": "repository_root_except_non_gnn_paths_relative_to_artifacts",
        "scope": "Supplementary descriptive evidence; no hardware-wide performance or stage-causal claim.",
        "non_gnn": non_gnn,
        "runtime_coverage": coverage,
        "runtime_final_throughput_summary": summary,
        "runtime_metric_contract": {
            "samples": "SDK total_samples equals metadata.num_steps * 4096 nominal slots, including padding.",
            "time": "SDK total_time is elapsed since profiler initialization, including preparation/waiting; distinct from the selected post-warmup log window.",
            "rate": "Both SDK-reported samples_per_sec and total_samples/total_time are retained; they are not forced to agree.",
            "preparation": "track.json intervals retain original SDK names. Nested intervals are not additive.",
            "flops": "Raw SDK flops_utilization retained without deriving compute/communication time fractions.",
        },
        "script_sha256": sha256(Path(__file__).resolve()),
        "non_gnn_script_sha256": sha256(Path(__file__).with_name("compare_non_gnn_20260918.py")),
    }
    (output / "supplementary.json").write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    write_tsv(output / "non_gnn.tsv", non_gnn["rows"])
    write_tsv(output / "runtime_executors.tsv", records)
    write_tsv(output / "runtime_final_throughput.tsv", summary)
    non_gnn_table, runtime_table = summary_tables(non_gnn, records)
    ratios = [row["cs3_over_h100"] for row in non_gnn["rows"]]
    largest = max((row for row in records if row["purpose"] == "throughput"),
                  key=lambda row: row["sdk_total_time_seconds"] / row["post_warmup_log_endpoint_seconds"])
    text = f"""# HPC Asia補助記録の再集計

## CS-3の計時範囲

主比較には、40ステップのウォームアップ後の生ログの端点時間を用いる。
SDKの累積時間は、プロファイラ生成後の実行準備・待ち・ウォームアップを含む。
独立性能測定{coverage['standalone_throughput_executors']}実行の計時範囲を以下に示す。値は平均（最小–最大）。

{runtime_table}

主比較の区間はogbn-arxivで40→840、ogbn-productsで40→1640ステップ。
例えば`{largest['run']}`ではSDK累積時間が{largest['sdk_total_time_seconds']:.3f}秒、
ウォームアップ後のログ端点間は{largest['post_warmup_log_endpoint_seconds']:.3f}秒だった。
当該ログはイメージビルド失敗後のvenvへの切替と、その後の学習完了を記録している。
実行準備の影響を含むSDK累積時間と、主比較の計時範囲を区別する事例である。

SDKの`total_samples`は全件で`num_steps × 4096`と一致した。
末尾パディングを含む公称枠数であり、主比較の実対象頂点数/秒とは分子が異なる。
SDKの報告速度と`total_samples / total_time`には最大{coverage['max_sdk_rate_relative_difference_percent']:.3f}%の差があるため、両値を別々に保存した。
`track.json`の初期化・compile・最初のloss受信までの区間も保存した。
これらは重複を許す区間であり、WSEの計算・転送・入力待ちへの時間分解には追加の測定が必要である。

SDK記録は{coverage['source_runs']}実行に由来する{coverage['paired_executors']}組で、
学習executor {coverage['stages']['train']}件、評価executor {coverage['stages']['validate']}件だった。
学習executorのうち{coverage['learning_train_executors']}件は旧入力方針の学習run、
{coverage['reshuffle_train_executors']}件は追加のreshuffle確認runを定期評価で分割した区間である。
reshuffleの1実行は[新旧比較](../learning/reshuffle_comparison.md)で別に扱い、主比較へ混ぜない。
さらに毎周shuffle後の独立性能{coverage['reshuffle_throughput_executors']}実行を
[速度の新旧比較](../throughput/reshuffle_comparison.md)で扱う。
独立性能測定は1実行1学習executorに対応した。
全組でmodel側の同じディレクトリにある`metadata.json`のstage・num_stepsと、performance・trackを対応付けた。
全performance・trackのSHA-256を`records/model_dirs/hpcasia_final/SDK_FILES.tsv`に照合した。

## 別ワークロードへの拡張候補

BERT・Llamaの4条件では、CS-3/Model Zooの学習区間速度がH100/native GPUの{min(ratios):.2f}–{max(ratios):.2f}倍だった。
各条件・各環境1実行。同じ系列長・実効バッチ・BF16の21–200ステップを比較した。

{non_gnn_table}

この比較は実装、勾配蓄積と実行系の差を含む。
GraphSAGEの主結果と分けて、ワークロード間の差を調べる拡張候補として扱う。
入力・モデル設定のハッシュ、終了状態、全ステップの有限lossと計時端点を既存の
`compare_non_gnn_20260918.compare()`で再検査した。

## 出力と再生成

このページの2表を再生成する。全件の数値と出典は以下に保存する。

- [non_gnn.tsv](non_gnn.tsv)：別ワークロード4条件の再計算値と原ログハッシュ。
- [runtime_executors.tsv](runtime_executors.tsv)：全executorの元パス、stage、step、SDK値、主比較の端点時間、ハッシュ。
- [runtime_final_throughput.tsv](runtime_final_throughput.tsv)：dataset・cache条件別の実行数、平均、最小、最大。
- [supplementary.json](supplementary.json)：集計、分子・時間の定義、索引とコードのハッシュ。

```bash
.venv/bin/python analyze/scripts/logs/hpcasia_supplementary_analysis.py
```

原記録は`records/raw_logs/hpcasia/non_gnn/`および
`records/model_dirs/hpcasia_final/`。個別パスはTSVに残した。
"""
    (output / "findings.md").write_text(text)
    print(json.dumps({"output": str(output), "non_gnn_conditions": len(non_gnn["rows"]),
                      **coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
