"""Regenerate CS-3 input-pipeline evidence tables from the curated run indexes.

Run from any directory with the repository virtual environment. Measurements use
existing Model Zoo parsers, whose hashes are written to provenance.json. Failed
and timed-out trials are retained only in coverage tables. H100 exploration is
supporting evidence; the paper-facing tables focus on CS-3.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
VALID = {"completed", "unstable"}


def read_tsv(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-9)


def config_identity(config, *, controls=False):
    """Remove only the explicitly permitted differences for each panel."""
    config = deepcopy(config)
    init, loader = (
        config["trainer"]["init"],
        config["trainer"]["fit"]["train_dataloader"],
    )
    init.pop("model_dir", None)
    init.get("backend", {}).get("cluster_config", {}).pop("job_labels", None)
    init.get("loop", {}).pop("max_steps", None)
    loader.pop("num_workers", None)
    if controls:
        for key in (
            "prefetch_factor",
            "persistent_workers",
            "cache_fraction",
            "static_batch_cache_size",
        ):
            loader.pop(key, None)
        if loader.get("worker_diagnostics") == {"enabled": False}:
            loader.pop("worker_diagnostics")
    return config


def backend(row):
    if row["platform"] == "cs3":
        return "csx"
    return "fixed_shape" if "fixed_shape" in row["run"] else "pyg"


def load_trial(row, parsers, root=ROOT, excluded=False):
    config = yaml.safe_load((root / row["params"]).read_text())
    result = json.loads((root / row["result"]).read_text())
    loader = config["trainer"]["fit"]["train_dataloader"]
    if result.get("status") != row["status"]:
        raise ValueError(f"Index/result status mismatch: {row['run']}")
    if result.get("config_sha256") != digest(config):
        raise ValueError(f"Configuration hash mismatch: {row['run']}")
    if loader["num_workers"] != int(row["workers"]):
        raise ValueError(f"Worker configuration mismatch: {row['run']}")
    kind = backend(row)
    measurement = result.get("measurement") or {}
    checked = {}
    if row["status"] in VALID and not excluded:
        start, end = measurement["start_step"], measurement["end_step"]
        log = root / row["log"]
        if row["metric_basis"] == "nominal_slots":
            # Older logs predate exact seed accounting. Keep this numerator.
            parser = parsers["csx"]
            points = parser.read_points(log)
            parser._check_training_only_window(log, start, end)
            checked = parser.measure(points, start, end, loader["batch_size"])
            checked["metric"] = "nominal_slots_per_second"
            checked["throughput"] = checked["nominal_slots_per_second"]
        else:
            checked = parsers[kind].summarize(log, start, end)
        for field in (
            "training_window_seconds",
            "seed_nodes",
            "nominal_slots_per_second",
            "seed_nodes_per_second",
            "optimizer_steps",
            "skipped_optimizer_steps",
        ):
            if field in measurement and (
                field not in checked or not close(measurement[field], checked[field])
            ):
                raise ValueError(f"Recomputed {field} mismatch: {row['run']}")
        if checked.get("skipped_optimizer_steps", 0):
            raise ValueError(f"Skipped optimizer updates: {row['run']}")
    elif measurement and excluded:
        # Archive status is authoritative even if a terminal log summary exists.
        checked = {}
    role = "worker_sweep"
    if "/tuning/" in row["run"]:
        role = "tuning_" + result.get("phase", "unknown")
    elif "/control_" in row["run"]:
        role = (
            "static_replay_probe"
            if loader.get("static_batch_cache_size")
            else "input_intervention"
        )
    elif "/diagnostic_" in row["run"]:
        role = "instrumented_diagnostic"
    elif "hpcasia_final/" in row["run"]:
        role = "independent_final_reference"
    check = checked.get("half_window_check") or {}
    contract = checked.get("input_contract") or {}
    out = dict(row)
    study_path = (
        root
        / row["params"].replace(
            "records/model_dirs/invalid/", "records/model_dirs/"
        )
    ).parent.parent / "study.json"
    if not study_path.exists():
        study_path = (root / row["params"]).parent.parent / "study.json"
    environment = (
        json.loads(study_path.read_text()).get("environment", {})
        if study_path.exists()
        else {}
    )
    out.update(
        backend=kind,
        cohort="retry" if "fixed_shape_retry_" in row["run"] else "original",
        excluded=excluded,
        role=role,
        workers=loader["num_workers"],
        prefetch_factor=loader.get("prefetch_factor"),
        persistent_workers=loader.get("persistent_workers"),
        cache_fraction=loader.get("cache_fraction"),
        static_batch_cache_size=loader.get("static_batch_cache_size", 0),
        repeat=result.get("repeat", 1),
        returncode=result.get("returncode"),
        failure_reason=result.get("failure_reason"),
        start_step=checked.get("start_step"),
        end_step=checked.get("end_step"),
        metric=checked.get("metric"),
        throughput=checked.get("throughput"),
        seed_nodes=checked.get("seed_nodes"),
        nominal_slots_per_second=checked.get("nominal_slots_per_second"),
        seed_nodes_per_second=checked.get("seed_nodes_per_second"),
        training_window_seconds=checked.get("training_window_seconds"),
        half_difference_percent=check.get("symmetric_difference_percent"),
        half_within_tolerance=check.get("within_tolerance"),
        skipped_optimizer_steps=checked.get("skipped_optimizer_steps"),
        input_order_sha256=contract.get("ordered_targets_and_labels_sha256"),
        config_sha256=digest(config),
        comparison_config_sha256=digest(config_identity(config)),
        log_sha256=sha(root / row["log"]),
        result_sha256=sha(root / row["result"]),
        study_git_commit=environment.get("git_commit"),
        study_source_sha256=environment.get("source_sha256"),
    )
    return out, config


def matched_r04(rows, parsers, root=ROOT):
    indexed = {r["run"]: r for r in rows}
    selected, identities, hashes = [], [], []
    for source in read_tsv(root / "records/raw_logs/hpcasia/input_pipeline/R04.tsv"):
        row = dict(indexed[source["run"]])
        m = parsers["csx"].summarize(
            root / row["log"], int(source["start_step"]), int(source["end_step"])
        )
        if not close(m["throughput"], source["seed_nodes_per_second"]):
            raise ValueError("R04 curated rate differs from raw log")
        order = m["input_contract"]["ordered_targets_and_labels_sha256"]
        if order != source["input_order_sha256"]:
            raise ValueError("R04 target order differs from curated reference")
        identities.append(
            config_identity(yaml.safe_load((root / row["params"]).read_text()))
        )
        hashes.append(order)
        row.update(
            start_step=m["start_step"],
            end_step=m["end_step"],
            throughput=m["throughput"],
            seed_nodes_per_second=m["throughput"],
            nominal_slots_per_second=m["nominal_slots_per_second"],
            training_window_seconds=m["training_window_seconds"],
            seed_nodes=m["seed_nodes"],
            half_difference_percent=(m.get("half_window_check") or {}).get(
                "symmetric_difference_percent"
            ),
            half_within_tolerance=(m.get("half_window_check") or {}).get(
                "within_tolerance"
            ),
        )
        selected.append(row)
    if len(set(hashes)) != 1 or any(item != identities[0] for item in identities[1:]):
        raise ValueError("R04 configuration or target order mismatch")
    return selected


def summarize(rows, keys, field="throughput"):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, items in sorted(groups.items(), key=lambda p: str(p[0])):
        values = [r[field] for r in items if r[field] is not None]
        row = dict(zip(keys, group))
        row.update(
            n_total=len(items),
            n_measured=len(values),
            mean=statistics.mean(values) if values else None,
            sd=statistics.stdev(values) if len(values) > 1 else None,
            minimum=min(values) if values else None,
            maximum=max(values) if values else None,
            status_counts=json.dumps(
                dict(sorted(Counter(r["status"] for r in items).items()))
            ),
        )
        result.append(row)
    return result


def retry_pairs(rows, configs):
    fixed = [r for r in rows if r["backend"] == "fixed_shape"]
    original = {
        (r["dataset"], r["workers"], r["repeat"]): r
        for r in fixed
        if r["cohort"] == "original"
    }
    pairs = []
    for new in fixed:
        if new["cohort"] != "retry":
            continue
        old = original[new["dataset"], new["workers"], new["repeat"]]
        a, b = deepcopy(configs[old["run"]]), deepcopy(configs[new["run"]])
        a["trainer"]["init"].pop("model_dir")
        b["trainer"]["init"].pop("model_dir")
        if a != b:
            raise ValueError("Retry configuration differs beyond output directory")
        pairs.append(
            dict(
                dataset=new["dataset"],
                workers=new["workers"],
                repeat=new["repeat"],
                original_run=old["run"],
                original_status=old["status"],
                retry_run=new["run"],
                retry_status=new["status"],
                seed_nodes_per_second=new["seed_nodes_per_second"],
                original_params=old["params"],
                retry_params=new["params"],
                configuration_match_except_output=True,
            )
        )
    return pairs


def retry_resources(out):
    """Read job-boundary cgroup counters without attributing victims to a PID."""
    rows, sources = [], []
    pattern = "fixed_shape*/w*/resources/*.log"
    roots = [
        ROOT / "records/raw_logs/hpcasia/input_pipeline/h100/products",
        ROOT / "records/raw_logs/invalid/hpcasia/input_pipeline/h100/products",
    ]
    for path in sorted(path for directory in roots for path in directory.glob(pattern)):
        content = path.read_text()
        if "runner_exit=" not in content:
            continue
        before, after = content.split("runner_exit=", 1)
        sources.append(path)
        states = []
        for text in (before, after):
            state, scope, events = {}, None, False
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("/sys/fs/cgroup/"):
                    scope, name = line.rsplit("/", 1)
                    state.setdefault(scope, {})
                    events = name == "memory.events"
                    if name in {"memory.max", "memory.swap.max"}:
                        state[scope][name] = lines[i + 1]
                elif events and line.split(" ", 1)[0] in {"oom", "oom_kill", "max"}:
                    key, value = line.split()
                    state[scope][key] = int(value)
            states.append(state)
        for scope in sorted(set(states[0]) & set(states[1])):
            a, b = states[0][scope], states[1][scope]
            if not all(key in a and key in b for key in ("oom", "oom_kill", "max")):
                continue
            row = dict(
                workers=int(path.parent.parent.name.removeprefix("w")),
                scope=scope,
                memory_max_bytes=a.get("memory.max"),
                swap_max_bytes=a.get("memory.swap.max"),
                source=str(path.relative_to(ROOT)),
                source_sha256=sha(path),
            )
            for key in ("oom", "oom_kill", "max"):
                row[key + "_before"], row[key + "_after"] = a[key], b[key]
                row[key + "_delta"] = b[key] - a[key]
            rows.append(row)
    write_tsv(out / "fixed_shape_retry_resources.tsv", rows)
    return rows, sources


def diagnostics(rows, out):
    """Separate remote CPU events from the client's compile-time data probing."""
    batches, summaries, layouts, sources = [], [], [], []
    for row in rows:
        if row["role"] != "instrumented_diagnostic" or row["excluded"]:
            continue
        events = []
        directory = (ROOT / row["params"]).parent / "worker_diagnostics"
        for path in sorted(directory.rglob("*.jsonl")):
            sources.append(path)
            for ordinal, line in enumerate(path.read_text().splitlines(), 1):
                event = json.loads(line)
                if event.get("hostname", "").startswith("wsjob-"):
                    events.append((event, str(path.relative_to(ROOT)), ordinal))
        generated = [(e, p, n) for e, p, n in events if e["event"] == "batch_generated"]
        iterators = [e for e, _, _ in events if e["event"] == "iterator_started"]
        if not iterators or any(
            len(e["worker_pids"]) != row["workers"] for e in iterators
        ):
            raise ValueError(
                "Remote effective worker count differs from configured count"
            )
        if any(e["settings"]["num_workers"] != row["workers"] for e in iterators):
            raise ValueError("Remote loader settings differ from configured workers")
        if any(e["settings"] != iterators[0]["settings"] for e in iterators):
            raise ValueError("Effective loader settings changed across iterations")
        for key in ("prefetch_factor", "persistent_workers"):
            if iterators[0]["settings"][key] != row[key]:
                raise ValueError(f"Configured {key} differs from remote loader")
        reused_pids = len({tuple(e["worker_pids"]) for e in iterators}) == 1
        for event, source, line in generated:
            sampling = event["phases"]["sampling"]["wall_ns"] / 1e9
            gather = event["phases"]["gather"]["wall_ns"] / 1e9
            wall = event["wall_ns"] / 1e9
            batches.append(
                dict(
                    run=row["run"],
                    workers=row["workers"],
                    hostname=event["hostname"],
                    pid=event["pid"],
                    worker_id=event["worker_id"],
                    batch_index=event["batch_index"],
                    process_batch_ordinal=event["process_batch_ordinal"],
                    sampling_seconds=sampling,
                    gather_seconds=gather,
                    wall_seconds=wall,
                    other_seconds=wall - sampling - gather,
                    source=source,
                    line=line,
                )
            )
        payloads = [(e, p, n) for e, p, n in events if e["event"] == "batch_layout"]
        sizes = {e["logical_bytes"] for e, _, _ in payloads}
        if len(sizes) != 1:
            raise ValueError("Diagnostic layouts have inconsistent payload sizes")
        for event, source, line in payloads:
            if (
                sum(t["logical_bytes"] for t in event["tensors"])
                != event["logical_bytes"]
            ):
                raise ValueError("Diagnostic payload total differs from tensor sum")
            for tensor in event["tensors"]:
                layouts.append(
                    dict(
                        run=row["run"],
                        workers=row["workers"],
                        pid=event["pid"],
                        **{
                            k: (json.dumps(v) if isinstance(v, list) else v)
                            for k, v in tensor.items()
                        },
                        source=source,
                        line=line,
                    )
                )
        own = [b for b in batches if b["run"] == row["run"]]
        stats = dict(
            run=row["run"],
            configured_workers=row["workers"],
            effective_workers=row["workers"],
            generated_batches=len(generated),
            received_batches=sum(e["event"] == "batch_received" for e, _, _ in events),
            iterator_starts=len(iterators),
            worker_pids_reused_across_iterators=reused_pids,
            logical_bytes_per_batch=sizes.pop(),
            layout_records=len(payloads),
            effective_pin_memory=iterators[0]["settings"]["pin_memory"],
            effective_prefetch_factor=iterators[0]["settings"]["prefetch_factor"],
            effective_persistent_workers=iterators[0]["settings"]["persistent_workers"],
        )
        for field in (
            "sampling_seconds",
            "gather_seconds",
            "other_seconds",
            "wall_seconds",
        ):
            stats[field + "_mean"] = statistics.mean(b[field] for b in own)
            stats[field + "_median"] = statistics.median(b[field] for b in own)
            stats[field + "_p90"] = statistics.quantiles(
                (b[field] for b in own), n=10, method="inclusive"
            )[8]
        summaries.append(stats)
    write_tsv(out / "cs3_diagnostic_batches.tsv", batches)
    write_tsv(out / "cs3_diagnostic_summary.tsv", summaries)
    write_tsv(out / "cs3_diagnostic_layouts.tsv", layouts)
    return summaries, layouts, sources


def mean_sd(summary):
    if not summary["n_measured"]:
        return "—"
    sd = summary["sd"]
    return f"{summary['mean']:,.1f} ± {sd:.1f}" if sd is not None else f"{summary['mean']:,.1f}"


def supporting_tables(rows, layouts, resources, out):
    """Keep implementation exploration available without promoting it to a result figure."""
    lines = [
        "# 入力経路の補助集計",
        "",
        "[CS-3の入力調査](findings.md)を補足する。H100単独の探索は実装経路を追加比較するための資料であり，本文のCS-3対H100性能比較には独立した最終反復を使う。",
        "",
        "## H100のworker探索",
        "",
        "seed 42，prefetch=2，worker永続化有効，特徴量キャッシュなし。warmup 40 step後，arxivは800 step，productsは1600 stepを計測する。",
        "速度は同期端点間の実対象頂点/秒，平均 ± 標本標準偏差。完了した不安定な測定も含め，時間切れ・失敗の速度は集計から除く。",
        "半区間差は前半・後半の速度a,bについて100×|a−b|/平均(a,b)とした範囲であり，2%以下を完了，2%超を不安定と記す。これは探索時の変動の確認に用いる。",
        "",
    ]
    state_names = {"completed": "完了", "unstable": "不安定", "timeout": "時間切れ", "failed": "失敗"}
    gpu = [r for r in rows if r["platform"] == "h100"]
    for dataset in ("arxiv", "products"):
        for kind, name in (("pyg", "PyG部分グラフ"), ("fixed_shape", "固定形状PyTorch")):
            selected = [r for r in gpu if r["dataset"] == dataset and r["backend"] == kind and r["cohort"] == "original"]
            lines += [
                f"### ogbn-{dataset} / {name}",
                "",
                "| worker数 | 状態別件数 | 速度（実対象頂点/秒） | 測定完了数 | 半区間差の範囲（%） |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
            for summary in sorted(summarize(selected, ["workers"]), key=lambda r: r["workers"]):
                own = [r for r in selected if r["workers"] == summary["workers"]]
                counts = Counter(r["status"] for r in own)
                status = "，".join(f"{label} {counts[key]}" for key, label in state_names.items() if counts[key])
                diffs = [r["half_difference_percent"] for r in own if r["half_difference_percent"] is not None]
                spread = f"{min(diffs):.2f}–{max(diffs):.2f}" if diffs else "—"
                lines.append(f"| {summary['workers']} | {status} | {mean_sd(summary)} | {summary['n_measured']} | {spread} |")
            lines.append("")
    retries = [r for r in gpu if r["cohort"] == "retry"]
    if retries:
        lines += [
            "## 固定形状productsの再実行",
            "",
            "全12組でparams.yamlの差は出力ディレクトリだけだった。初回と再実行は別々に集計する。失敗の終了コードは-9。",
            "",
            "| worker数 | 初回の状態別件数 | 再実行の状態別件数 | 再実行の速度（実対象頂点/秒） |",
            "| ---: | --- | --- | ---: |",
        ]
        for summary in sorted(summarize(retries, ["workers"]), key=lambda r: r["workers"]):
            state_cells = []
            for cohort in ("original", "retry"):
                counts = Counter(r["status"] for r in gpu if r["backend"] == "fixed_shape" and r["dataset"] == "products" and r["cohort"] == cohort and r["workers"] == summary["workers"])
                state_cells.append("，".join(f"{label} {counts[key]}" for key, label in state_names.items() if counts[key]))
            lines.append(f"| {summary['workers']} | {' | '.join(state_cells)} | {mean_sd(summary)} |")
        lines += ["", "[初回と再実行の対応](fixed_shape_retry_pairs.tsv)・[全試行の状態](fixed_shape_status.tsv)。", ""]
        for item in resources:
            if item["workers"] == 48 and item["scope"].endswith("/nqs-jsv.service"):
                lines += [
                    f"workers 48の再実行期間中，スケジューラサービスのcgroupでoomカウンタは{item['oom_before']}→{item['oom_after']}，oom_killは{item['oom_kill_before']}→{item['oom_kill_after']}となった。メモリ上限は{int(item['memory_max_bytes']):,} byte，swap上限は{item['swap_max_bytes']} byteだった。",
                    "このサービスcgroupでOOM killが発生した記録が得られた。対象PIDと他ジョブの寄与は記録されておらず，3試行それぞれの終了原因の帰属には追加の記録が必要である。",
                    "[資源境界の集計](fixed_shape_retry_resources.tsv)から元ログとカウンタ差を追える。",
                    "",
                ]
    first = layouts[0]
    one = [r for r in layouts if r["run"] == first["run"] and r["pid"] == first["pid"]]
    groups = [
        ("深さ0–1の特徴量", ["batch.node_features[0]", "batch.node_features[1]"]),
        ("深さ2の特徴量", ["batch.node_features[2]"]),
        ("深さ3の特徴量", ["batch.node_features[3]"]),
        ("マスク・ラベル", [r["name"] for r in one if "node_features" not in r["name"]]),
    ]
    lines += [
        "## CS-3診断バッチの論理サイズ内訳",
        "",
        "同じ固定形状のバッチについて，テンソルのshapeとdtypeから算出した。転送バイト数と常駐メモリ量は別の測定対象である。",
        "",
        "| テンソル群 | 論理サイズ（MiB） |",
        "| --- | ---: |",
    ]
    for name, names in groups:
        size = sum(r["logical_bytes"] for r in one if r["name"] in names) / 2**20
        lines.append(f"| {name} | {size:.2f} |")
    lines += [
        f"| 合計 | {sum(r['logical_bytes'] for r in one) / 2**20:.2f} |",
        "",
        "[形状記録](cs3_diagnostic_layouts.tsv)に各テンソルと元JSONLの行番号を保存した。",
        "",
    ]
    (out / "supporting_tables.md").write_text("\n".join(lines))


def findings(rows, r04, controls, diagnostic_summary, resources, out):
    worker_summary = {int(r["workers"]): r for r in summarize(r04, ["workers"])}
    control_summary = {r["control"]: r for r in summarize(controls, ["control"])}
    means = {worker: summary["mean"] for worker, summary in worker_summary.items()}
    cmeans = {control: summary["mean"] for control, summary in control_summary.items()}
    lines = [
        "# CS-3の入力経路の追加検証",
        "",
        "保存済みの測定から，worker数と入力設定がCS-3のGraphSAGE性能をどの程度変えるかを確認する。各表の速度は平均 ± 標本標準偏差であり，nは反復数である。探索の反復はseed 42を共有し，複数seedの最終性能比較とは分ける。",
        "",
        "## CS-3のworker数",
        "",
        "ogbn-arxiv，40→440 step。worker数・総step数以外の設定と対象順序のハッシュを照合した。",
        "",
        "| CPU worker数 | 速度（実対象頂点/秒） | workers 4比 | n | 取得した測定 |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for worker in (4, 8, 12, 16):
        item = worker_summary[worker]
        source = "独立した最終性能反復" if worker == 4 else "入力探索"
        lines.append(f"| {worker} | {mean_sd(item)} | {item['mean'] / means[4]:.3f} | {item['n_measured']} | {source} |")
    lines += [
        "",
        f"workers 16の平均は4の{means[16] / means[4]:.3f}倍であり，この範囲のworker増加による改善は小さい。異なる取得段階を含むため，単一の無作為化されたworker比較としての統計検定は行わない。",
        "[13件の再集計](cs3_matched_workers.tsv)。",
        "",
        "## CS-3の入力設定への介入",
        "",
        "ogbn-arxiv，workers 4，40→440 step。同一キャンペーン内で操作差を設定・ログから照合した。旧ログは対象頂点の厳密な計数を持たないため，分子はパディングを含む公称枠とする。",
        "",
        "| 入力設定 | 速度（公称枠/秒） | 基準比 | n |",
        "| --- | ---: | ---: | ---: |",
    ]
    for control, label in (
        ("baseline", "基準"),
        ("prefetch1", "prefetch = 1"),
        ("persistent_off", "worker永続化なし"),
        ("feature_cache", "ホスト特徴量キャッシュ"),
        ("static_batch", "静的バッチ再利用（機構検証）"),
    ):
        item = control_summary[control]
        lines.append(f"| {label} | {mean_sd(item)} | {item['mean'] / cmeans['baseline']:.3f} | {item['n_measured']} |")
    lines += [
        "",
        f"通常の入力設定への介入では基準との差は小さく，静的再利用では{cmeans['static_batch'] / cmeans['baseline']:.3f}倍となった。静的再利用は各workerが準備済みの1バッチを繰り返す機構検証であり，通常のサンプリング学習との精度比較から独立して扱う。",
        "[全15件](cs3_input_interventions.tsv)。",
        "",
        "## CS-3のCPU入力準備",
        "",
        "ogbn-arxivの計測を有効にした診断で，遠隔Worker上のCPU入力準備を集計した。設定worker数と実効子PID数は一致し，prefetch=2，worker永続化有効，pin_memory無効を記録した。クライアント側のコンパイル時入力確認はhostnameで除いた。",
        "",
        "| CPU worker数 | 生成バッチ数 | 受信バッチ数 | サンプリング（秒） | 特徴量収集（秒） | バッチ生成全体（秒） |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(diagnostic_summary, key=lambda r: r["configured_workers"]):
        durations = " | ".join(f"{item[field + '_median']:.2f} / {item[field + '_p90']:.2f}" for field in ("sampling_seconds", "gather_seconds", "wall_seconds"))
        lines.append(f"| {item['configured_workers']} | {item['generated_batches']} | {item['received_batches']} | {durations} |")
    lines += [
        "",
        "時間は生成バッチごとの中央値 / 90パーセンタイル。90パーセンタイルは昇順値の線形補間で求める。各診断は1実行であり，生成バッチ数には先読み分を含む。各段階の分位点は異なるバッチに対応するため，段階間で足し合わせない。",
        "worker間で準備が重なるため，これらのCPUバッチ時間から学習ループ時間に占める割合は求めない。28件の形状記録はすべて968,015,872 byte（923.17 MiB）の論理テンソルサイズを持つ。",
        "[診断集計](cs3_diagnostic_summary.tsv)・[バッチ別時間](cs3_diagnostic_batches.tsv)・[論理サイズ内訳](supporting_tables.md#cs-3診断バッチの論理サイズ内訳)。",
        "",
        "## IA³からの主張範囲と追加の方向性",
        "",
        "主張はGraphSAGEを高水準の固定形状経路で実行した範囲に保つ。worker数と入力設定の確認は，CS-3の入力設定が十分に検討されているかという疑問への証拠を増やす。",
        "静的再利用の速度差は入力準備を省く操作への応答を示す。総処理時間の原因別分解には，CPU準備・転送・デバイス計算の対応した段階計測が必要である。",
        "追加の方向として，同じ固定形状モデル・samplerをH100へ持ち込んだ経路を調べられるようになった。通常PyGとはbias構成・サンプリングも異なるため，現時点では実装経路全体の比較となる。独立した最終性能反復と複数seedの学習品質を加えることで，表現と実行環境を分けた比較を広げられる。",
        "H100単独のworker探索，再実行，失敗範囲は[補助集計](supporting_tables.md)にまとめた。固定形状productsのworkers 48は再実行でも未完了であり，条件ごとの失敗原因と不安定条件の時間変動が残る観測対象である。",
        "",
        "## 再生成と資料",
        "",
        "```bash",
        ".venv/bin/python analyze/scripts/logs/hpcasia_input_analysis.py",
        "```",
        "",
        "入力索引は [RUNS.tsv](../../../../records/raw_logs/hpcasia/RUNS.tsv)，worker比較の選定は [R04.tsv](../../../../records/raw_logs/hpcasia/input_pipeline/R04.tsv)。",
        "処理分担の実装解釈は [CPU入力経路](../../../../context/fact/host-neighbor-sampling.md)に対応する。形状・時間は保存されたworker_diagnosticsのbatch_layout・batch_generatedから読む。",
        "[全実行表](input_trials.tsv)に状態，分子，計測区間，設定・ログ・結果のパスとSHA-256を保存し，[条件別集計](input_summary.tsv)では分子と区間を分けた。[provenance.json](provenance.json)に索引と再利用parserのSHA-256を記録する。",
        "",
    ]
    (out / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modelzoo-root", type=Path, default=ROOT.parents[1]
    )
    parser.add_argument(
        "--index", type=Path, default=ROOT / "records/raw_logs/hpcasia/RUNS.tsv"
    )
    parser.add_argument(
        "--excluded-index",
        type=Path,
        default=ROOT / "records/raw_logs/invalid/hpcasia/UNFINISHED_RUNS.tsv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/input_pipeline"
    )
    args = parser.parse_args()
    if not args.excluded_index.exists():
        parser.error("An excluded-run index is required to show incomplete coverage")
    sys.path.insert(0, str(args.modelzoo_root.resolve() / "src"))
    parsers = {
        key: importlib.import_module("cerebras.modelzoo.models.gnn.tools." + module)
        for key, module in (
            ("csx", "measure_window"),
            ("pyg", "measure_pyg"),
            ("fixed_shape", "measure_fixed_shape"),
        )
    }
    selected_ids = {
        r["run"]
        for r in read_tsv(ROOT / "records/raw_logs/hpcasia/input_pipeline/R04.tsv")
    }
    inputs = [
        (r, False)
        for r in read_tsv(args.index)
        if r["purpose"] == "input_pipeline" or r["run"] in selected_ids
    ]
    if args.excluded_index.exists():
        inputs += [
            (r, True)
            for r in read_tsv(args.excluded_index)
            if r["purpose"] == "input_pipeline"
        ]
    if len({r["run"] for r, _ in inputs}) != len(inputs):
        raise ValueError("Duplicate active and excluded run")
    rows, configs = [], {}
    for source, excluded in inputs:
        row, config = load_trial(source, parsers, excluded=excluded)
        rows.append(row)
        configs[row["run"]] = config
    r04 = matched_r04(rows, parsers)
    controls = []
    for row in rows:
        name = row["run"].rsplit("/", 1)[-1]
        if name.startswith("control_"):
            control = name.removeprefix("control_").rsplit("_r", 1)[0]
        elif "/workers_w04/sensitivity_w04_" in row["run"]:
            control = "baseline"
        else:
            continue
        controls.append(dict(row, control=control))
    identities = [config_identity(configs[r["run"]], controls=True) for r in controls]
    if any(item != identities[0] for item in identities[1:]):
        raise ValueError("Control panel differs beyond permitted intervention settings")
    args.output.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output / "input_trials.tsv", rows)
    write_tsv(
        args.output / "input_summary.tsv",
        summarize(
            rows,
            [
                "platform",
                "dataset",
                "backend",
                "role",
                "cohort",
                "comparison_config_sha256",
                "metric",
                "workers",
                "prefetch_factor",
                "persistent_workers",
                "cache_fraction",
                "static_batch_cache_size",
                "start_step",
                "end_step",
            ],
        ),
    )
    write_tsv(args.output / "cs3_matched_workers.tsv", r04)
    write_tsv(args.output / "cs3_input_interventions.tsv", controls)
    write_tsv(
        args.output / "fixed_shape_status.tsv",
        [r for r in rows if r["backend"] == "fixed_shape"],
    )
    pairs = retry_pairs(rows, configs)
    write_tsv(args.output / "fixed_shape_retry_pairs.tsv", pairs)
    resources, resource_sources = retry_resources(args.output)
    diagnostic_summary, layouts, diagnostic_sources = diagnostics(rows, args.output)
    supporting_tables(rows, layouts, resources, args.output)
    findings(rows, r04, controls, diagnostic_summary, resources, args.output)
    sources = [
        args.index,
        ROOT / "records/raw_logs/hpcasia/input_pipeline/R04.tsv",
        Path(__file__),
    ]
    if args.excluded_index.exists():
        sources.append(args.excluded_index)
    sources += [Path(module.__file__) for module in parsers.values()]
    sources += diagnostic_sources
    sources += resource_sources
    provenance = {
        "input_records": len(rows),
        "recomputed_measurements": sum(r["throughput"] is not None for r in rows),
        "excluded_records": sum(r["excluded"] for r in rows),
        "sources": [
            {
                "path": (
                    str(p.resolve().relative_to(ROOT))
                    if p.resolve().is_relative_to(ROOT)
                    else str(p.resolve())
                ),
                "sha256": sha(p),
            }
            for p in sources
        ],
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
