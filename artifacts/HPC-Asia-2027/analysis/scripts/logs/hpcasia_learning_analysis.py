"""Rebuild final-campaign GraphSAGE learning evidence from the active raw logs.

Run from the repository root with .venv/bin/python.  Full-precision raw accuracy
records are authoritative; imported learning_curves.json is cross-checked when
present.  The retry's SDK summary is deliberately kept distinct from a campaign
result.  No checkpoint, human review decision, or process return code is invented.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hpcasia-learning-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from plot_accuracy import read_curve


ROOT = Path(__file__).resolve().parents[3]
LABELS = {"cs3": "CS-3 / 固定形状", "h100": "H100 / PyG (コンパイル有効)"}
FIGURE_LABELS = {"cs3": "CS-3 / Model Zoo", "h100": "H100 / PyG (compiled)"}
COLORS = {"cs3": "#0072B2", "h100": "#D55E00"}
NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tsv(path, rows):
    require(bool(rows), f"No rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_raw(log, platform):
    evaluations, losses, contracts = [], [], []
    eval_step, eval_loss, pending_accuracy = None, None, None
    gpu_policy = None
    completed = False
    for line_number, line in enumerate(log.splitlines(), 1):
        if line.startswith("GNN_INPUT_CONTRACT "):
            contracts.append(json.loads(line.split(" ", 1)[1]))
        if line.startswith("[GPU policy] "):
            gpu_policy = json.loads(line.split("] ", 1)[1])
        if platform == "h100":
            if line.startswith("[Learning] "):
                event = json.loads(line.split("] ", 1)[1])
                if event["event"] == "validation":
                    evaluations.append(dict(step=event["step"], accuracy=event["accuracy"],
                                            validation_loss=None, log_line=line_number))
                elif event["event"] == "train":
                    losses.append(dict(step=event["step"], loss=event["loss"],
                                       window_mean_loss=event["window_mean_loss"],
                                       optimizer_steps=event["optimizer_steps"],
                                       skipped_optimizer_steps=event["skipped_optimizer_steps"],
                                       all_losses_finite=event["all_losses_finite"],
                                       log_line=line_number))
            if "Training Completed. Total Steps:" in line:
                completed = True
        else:
            train = re.search(r"\| Train Device=CSX, Step=(\d+), Loss=(" + NUMBER + ")", line)
            if train:
                losses.append(dict(step=int(train[1]), loss=float(train[2]),
                                   window_mean_loss=None, optimizer_steps=None,
                                   skipped_optimizer_steps=None, all_losses_finite=None,
                                   log_line=line_number))
            match = re.search(r"\| Eval Device=CSX, GlobalStep=(\d+),", line)
            if match:
                eval_step = int(match[1])
            match = re.search(r"Avg Eval Loss: (" + NUMBER + ")", line)
            if match:
                eval_loss = float(match[1])
            match = re.search(r"eval/masked_accuracy = (" + NUMBER + ")", line)
            if match:
                require(eval_step is not None and pending_accuracy is None,
                        f"Accuracy missing evaluation step or completion at line {line_number}")
                pending_accuracy = dict(step=eval_step, accuracy=float(match[1]),
                                        validation_loss=eval_loss, log_line=line_number)
            if "Evaluation completed successfully!" in line:
                require(pending_accuracy is not None, "Completed evaluation missing accuracy")
                evaluations.append(pending_accuracy)
                eval_step, eval_loss, pending_accuracy = None, None, None
            if "Training completed successfully!" in line:
                completed = True
    require(completed and pending_accuracy is None, "Incomplete training/evaluation log")
    require(evaluations and losses, "Missing learning records")
    for name, records in [("evaluation", evaluations), ("training", losses)]:
        steps = [row["step"] for row in records]
        require(steps == sorted(set(steps)), f"Nonmonotone or duplicate {name} steps")
    require(all(math.isfinite(row["accuracy"]) and 0 <= row["accuracy"] <= 1
                for row in evaluations), "Invalid validation accuracy")
    require(all(math.isfinite(row["loss"]) for row in losses), "Nonfinite recorded loss")
    return evaluations, losses, contracts, gpu_policy


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}.{key}" if prefix else key))
    else:
        result[prefix] = value
    return result


def load_runs(index):
    all_rows = list(csv.DictReader(index.open(), delimiter="\t"))
    learning = [r for r in all_rows if r["purpose"] == "learning"]
    selected = [r for r in learning if r["run"].startswith("hpcasia_final/")]
    excluded = [dict(r, exclusion_reason="Exploratory baseline preceding final campaign")
                for r in learning if not r["run"].startswith("hpcasia_final/")]
    keys = [(r["platform"], r["dataset"], int(r["seed"])) for r in selected]
    expected = {(p, d, s) for p in LABELS for d in ("arxiv", "products") for s in (42, 43, 44)}
    require(len(keys) == len(set(keys)) and set(keys) == expected,
            "Final learning cohort must contain exactly one run per platform/dataset/seed")
    runs, curves, losses, configs, sources, policies, contracts = [], [], [], {}, {}, {}, {}
    for row in sorted(selected, key=lambda r: (r["dataset"], r["platform"], int(r["seed"]))):
        require(row["status"] == "completed", f"Incomplete selected run: {row['run']}")
        key = (row["platform"], row["dataset"], int(row["seed"]))
        paths = {field: ROOT / row[field] for field in ("params", "log", "result")}
        for path in paths.values():
            require("invalid" not in path.resolve().parts, f"Invalid input: {path}")
            sources[str(path.relative_to(ROOT))] = sha256(path)
        cfg = yaml.safe_load(paths["params"].read_text())
        configs[key] = cfg
        init, fit = cfg["trainer"]["init"], cfg["trainer"]["fit"]
        max_steps, frequency = init["loop"]["max_steps"], init["loop"]["eval_frequency"]
        require((max_steps, frequency) == ((500, 20) if row["dataset"] == "arxiv" else (1000, 40)),
                "Unexpected final-campaign learning budget")
        require(init["seed"] == key[2] == fit["train_dataloader"]["sampler_seed"], "Seed mismatch")
        require(fit["val_dataloader"]["sampler_seed"] == key[2], "Validation seed mismatch")
        require(fit["val_dataloader"]["split"] == "valid" and not init["loop"]["eval_steps"],
                "Expected uncapped valid-split evaluation")
        ev, train, contract, policy = parse_raw(paths["log"].read_text(), row["platform"])
        require([e["step"] for e in ev] == list(range(frequency, max_steps + 1, frequency)),
                f"Unexpected evaluation schedule: {row['run']}")
        log_steps = init["logging"]["log_steps"]
        require([t["step"] for t in train] == list(range(log_steps, max_steps + 1, log_steps)),
                "Incomplete training loss records or step budget")
        result = json.loads(paths["result"].read_text())
        if "runs" in result:
            require(result["runs"] and all(r["status"] == "success" for r in result["runs"]),
                    "SDK summary contains unsuccessful execution")
            completion_basis = "SDK summary success + raw final training/evaluation steps"
        else:
            require(result["status"] == "completed" and result["returncode"] == 0,
                    "Campaign did not complete successfully")
            completion_basis = "campaign completed, returncode=0 + raw final steps"
        saved_curve = paths["params"].parent / "learning_curves.json"
        if saved_curve.exists():
            saved = read_curve(saved_curve)
            require([(x["step"], x["accuracy"]) for x in saved] ==
                    [(x["step"], x["accuracy"]) for x in ev], "Raw/saved curve mismatch")
            sources[str(saved_curve.relative_to(ROOT))] = sha256(saved_curve)
        if row["platform"] == "cs3":
            require(len(contract) == 1 and contract[0]["source_seed_nodes"] ==
                    sum(contract[0]["seed_nodes_by_batch"]), "Bad input contract")
            contracts[key] = contract[0]
        else:
            require(policy is not None, "Missing GPU optimizer policy")
            require('[Compile] {"enabled": true, "dynamic": true}' in paths["log"].read_text(),
                    "Expected compiled PyG learning run")
            expected_optimizer = init["optimizer"]["AdamW"]
            for source, target in [("learning_rate", "lr"), ("eps", "eps"), ("betas", "betas"),
                                   ("weight_decay", "weight_decay")]:
                require(expected_optimizer[source] == policy["optimizer_defaults"][target],
                        f"Runtime optimizer mismatch for {source}")
            require(all(t["all_losses_finite"] for t in train), "GPU reported nonfinite loss")
            require(train[-1]["optimizer_steps"] + train[-1]["skipped_optimizer_steps"] == max_steps,
                    "Optimizer update accounting mismatch")
            policies[row["run"]] = policy
        best = max(ev, key=lambda e: e["accuracy"])
        base = {k: row[k] for k in ("platform", "dataset", "seed", "run")}
        runs.append(dict(base, status=row["status"], final_step=max_steps, eval_frequency=frequency,
                         evaluation_count=len(ev), first_accuracy=ev[0]["accuracy"],
                         final_accuracy=ev[-1]["accuracy"], best_accuracy=best["accuracy"],
                         best_step=best["step"], best_minus_final_pp=100*(best["accuracy"]-ev[-1]["accuracy"]),
                         optimizer_steps=train[-1]["optimizer_steps"],
                         skipped_optimizer_steps=train[-1]["skipped_optimizer_steps"],
                         completion_basis=completion_basis, curve_crosscheck=saved_curve.exists(),
                         log=row["log"], params=row["params"], result=row["result"]))
        curves.extend(dict(base, **e) for e in ev)
        losses.extend(dict(base, **t) for t in train)
    # This is a nominal pass count, explicitly distinguished from optimizer
    # updates and from device-consumption evidence across CS-3 evaluation loops.
    for run in runs:
        contract = contracts[("cs3", run["dataset"], int(run["seed"]))]
        run["source_train_nodes"] = contract["source_seed_nodes"]
        run["batches_per_full_pass"] = len(contract["seed_nodes_by_batch"])
        run["nominal_loader_passes"] = run["final_step"] / run["batches_per_full_pass"]
    differences = []
    for dataset in ("arxiv", "products"):
        for seed in (42, 43, 44):
            left, right = (flatten(configs[(p, dataset, seed)]) for p in ("cs3", "h100"))
            for field in sorted(left.keys() | right.keys()):
                a, b = left.get(field, "<absent>"), right.get(field, "<absent>")
                if a != b:
                    differences.append(dict(dataset=dataset, seed=seed, field=field,
                                            cs3=json.dumps(a), h100=json.dumps(b)))
                    require(not field.startswith(("trainer.init.model.", "trainer.init.optimizer.",
                                                  "trainer.init.loop.")),
                            f"Unmatched model/optimizer/step setting: {field}")
    return runs, curves, losses, excluded, differences, sources, policies


def summarize(runs):
    summary, paired = [], []
    for dataset in ("arxiv", "products"):
        for platform in LABELS:
            group = [r for r in runs if r["dataset"] == dataset and r["platform"] == platform]
            row = dict(dataset=dataset, platform=platform, n=len(group),
                       final_step=group[0]["final_step"], evaluation_count=group[0]["evaluation_count"])
            for metric in ("final_accuracy", "best_accuracy", "best_minus_final_pp"):
                values = [r[metric] for r in group]
                row[metric + "_mean"] = statistics.mean(values)
                row[metric + "_sample_sd"] = statistics.stdev(values)
                row[metric + "_min"] = min(values)
                row[metric + "_max"] = max(values)
            summary.append(row)
        for seed in (42, 43, 44):
            pair = {r["platform"]: r for r in runs if r["dataset"] == dataset and int(r["seed"]) == seed}
            paired.append(dict(dataset=dataset, seed=seed,
                               final_cs3_minus_h100_pp=100*(pair["cs3"]["final_accuracy"]-pair["h100"]["final_accuracy"]),
                               best_cs3_minus_h100_pp=100*(pair["cs3"]["best_accuracy"]-pair["h100"]["best_accuracy"])))
    return summary, paired


def save_figure(figure, output, name):
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"{name}.{suffix}", dpi=200, bbox_inches="tight",
                       metadata={"CreationDate": None, "ModDate": None} if suffix == "pdf" else None)
    plt.close(figure)


def plot(curves, output):
    # Individual trajectories carry the seed variation; endpoint summaries belong
    # in the table, so a second mean curve and SD band add no necessary evidence.
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    for axis, dataset in zip(axes, ("arxiv", "products")):
        for platform in LABELS:
            for seed in (42, 43, 44):
                rows = sorted((r for r in curves if r["dataset"] == dataset
                               and r["platform"] == platform and int(r["seed"]) == seed),
                              key=lambda r: r["step"])
                axis.plot([r["step"] for r in rows], [100*r["accuracy"] for r in rows],
                          color=COLORS[platform], linestyle="-" if platform == "cs3" else "--",
                          alpha=.8, lw=1.35, label=FIGURE_LABELS[platform] if seed == 42 else None)
        axis.set(title=f"ogbn-{dataset}", xlabel="Training step", ylim=(25, 95),
                 xlim=(0, 500 if dataset == "arxiv" else 1000))
        axis.set_yticks(range(30, 100, 10))
        axis.grid(axis="y", color=".9", lw=.6)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].tick_params(labelleft=True)
    dip = next(r for r in curves if r["dataset"] == "products" and r["platform"] == "h100"
               and int(r["seed"]) == 43 and r["step"] == 840)
    axes[1].annotate("H100: seed 43", xy=(dip["step"], 100*dip["accuracy"]),
                     xytext=(760, 59), ha="center", fontsize=9, color=COLORS["h100"],
                     arrowprops={"arrowstyle": "-", "color": COLORS["h100"], "lw": .8})
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.03),
                  ncol=2, frameon=False, handlelength=3)
    figure.tight_layout(rect=(0, 0, 1, .94), w_pad=2)
    save_figure(figure, output, "validation_curves")


def write_figure_tex(output):
    """An ACM figure fragment included from the HPC Asia manuscript's src dir."""
    manuscript_dir = ROOT / "paper/5.HPC-Asia/graphsage-cs3/src"
    figure_path = Path(os.path.relpath(output.resolve() / "validation_curves.pdf", manuscript_dir)).as_posix()
    caption = (
        "Validation accuracy of GraphSAGE on CS-3/Model Zoo and H100/PyG with compilation enabled. "
        "Each line represents one run (seeds 42, 43, and 44 for each path and dataset), "
        "with all 25 evaluation points connected without smoothing. "
        "Solid lines denote CS-3 and dashed lines denote H100. "
        "Both panels use the same vertical scale."
    )
    text = "\n".join([
        "% Include from paper/5.HPC-Asia/graphsage-cs3/src/main.tex.",
        r"\begin{figure*}[t]", r"  \centering",
        r"  \includegraphics[width=\textwidth]{" + figure_path + "}",
        r"  \caption{" + caption + "}",
        r"  \label{fig:hpcasia-validation}",
        r"  \Description{Two panels compare validation accuracy against training steps for three seeds per path. "
        "Both paths improve during training. The H100 products run with seed 43 briefly drops at step 840.}",
        r"\end{figure*}", "",
    ])
    (output / "validation_curves.tex").write_text(text)


def findings(summary, paired, runs):
    lines = ["# 最終キャンペーンの学習評価", "",
             "`RUNS.tsv` の最終キャンペーン12件を集計した。各経路・各データセットで seed 42・43・44 の3件が完了し、各実行の検証精度25点を生ログから抽出した。探索時の学習2件は `excluded_runs.tsv` に残した。", "",
             "最終stepの精度を主集計とし、25回の評価中の最高精度を感度確認として併記する。最高値は検証集合で選んだ観測値であり、別のテスト集合での精度は取得していない。実験前のcheckpoint選択規則や研究者の承認は、今回の集計では補作していない。", "",
             "| データセット | 経路 | n | 最終精度 (%) 平均 ± 標本SD | 最高精度 (%) 平均 ± 標本SD |", "| --- | --- | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| ogbn-{row['dataset']} | {LABELS[row['platform']]} | {row['n']} | {100*row['final_accuracy_mean']:.3f} ± {100*row['final_accuracy_sample_sd']:.3f} | {100*row['best_accuracy_mean']:.3f} ± {100*row['best_accuracy_sample_sd']:.3f} |")
    lines += ["", "## IA³ の主張に対する根拠", ""]
    for dataset in ("arxiv", "products"):
        vals = [r["final_cs3_minus_h100_pp"] for r in paired if r["dataset"] == dataset]
        lines.append(f"ogbn-{dataset} の同じseed番号での CS-3 − H100 最終精度差は平均 {statistics.mean(vals):+.3f} ポイント、標本SD {statistics.stdev(vals):.3f} ポイント、範囲 {min(vals):+.3f}〜{max(vals):+.3f} ポイントだった。同じseed番号は同じ初期重み・近傍列を保証しない。")
        lines.append("")
    lines += ["2データセットの固定形状 GraphSAGE が3 seedで学習を完了し、検証精度が上昇したという観測を、IA³ の単一実行から補強できる。等価性の許容幅を事前設定した検定は実施しておらず、平均と標本SDを記述的に示す。特に products は最終値と最高値の差、seed間の幅も含めて示す必要がある。", "",
              "## 学習量・評価点・設定の照合", "",
              "- arxiv は500 step、20 stepごとの25評価。products は1000 step、40 stepごとの25評価。全12件で設定、最終学習step、最終評価stepが一致した。",
              "- 学習batch sizeは4096、末尾batchを保持する。CS-3入力契約の対象数は arxiv 90,941（23 batch/周）、products 196,615（49 batch/周）。従って設定上の周回換算はそれぞれ500/23 = 21.739、1000/49 = 20.408。これはloaderの公称周回数であり、CS-3の評価を跨ぐ入力継続やデバイス消費件数の追加検証とは区別する。",
              "- H100は arxiv seed42 と products 全seedでAMP更新スキップが1回、arxiv seed43・44で0回。CS-3の同じ更新回数は原本に記録がなく空欄。step数の一致とoptimizer更新回数の一致を分けた。",
              "- H100の実行ログは `torch.compile` 有効、AdamW eps=1e-6、betas=[0.9, 0.999]、学習率0.003、重みへのdecay=0.0005・biasへのdecay=0、GradScaler初期値32768を記録する。CS-3とモデル寸法、fanouts [15,10,5]、公称optimizer設定、dropout、seed集合、学習step・評価頻度は一致する。",
              "- `configuration_differences.tsv` は全末端設定の差をseed別に記録する。H100のcache_fraction=0.0とCS-3のnullは双方未キャッシュ。PyGとCS-3ではoptimizerのeps位置、bias構成、AMP実装・clamp、サンプリング経路に差が残る。H100は独立したPyG実装として比較する。",
              "- 検証はvalid split、評価step数の上限なし。全対象ノードを評価する設定でも近傍はサンプリングするため、full-neighbor評価とは区別する。",
              "- 学習lossは10 stepごとの当該stepの記録。H100の窓平均は別列に保存。CS-3は表示桁までの精度で、全stepの有限性を確認したものではない。products の入力契約では末尾batchは7対象である。周期的な損失低下と末尾batch候補の対応は [損失診断](loss_diagnostics.md)、既知の正規化不具合との照合は [実装・コンパイル結果の監査](loss_source_audit.md) に示す。損失の原記録は保持する。",
              "", "## 原本と再生成", "",
              "`evaluations.tsv` は全300評価点と生ログ行番号、`training_losses.tsv` は記録loss、`runs.tsv` は実行別の最終・最高精度と設定/原本パス、`summary.tsv` は平均・標本SD・n、`paired_accuracy.tsv` は同じseed番号の差を保存する。`manifest.json` は原本・索引・集計コードのSHA-256とGPU実行方針を保存する。", "",
              "11件は保存済みlearning_curves.jsonと全精度・stepが一致した。products seed44の追加取り込みはlearning_curves.jsonを含まず、SDK summaryのstatus=successとtrain.logの1000 step・最終評価から確認した。SDK summaryのending_step=0を学習到達stepとして使っていない。", "",
              "```bash", ".venv/bin/python analyze/scripts/logs/hpcasia_learning_analysis.py", "```", "",
              "図は [検証精度の推移](validation_curves.pdf) の1点に絞った。2データセットを同じ縦軸範囲で並べ、実線はCS-3、破線はH100、各線は1 seedの全25評価点を示す。全実行の曲線を残し、平均線・標準偏差帯の重ね描きを省いた。図中の文字は英語に統一した。seed数・評価点数の説明は図外の [LaTeXキャプション](validation_curves.tex) に置く。この断片はHPC Asia原稿の `src` から読み込める。最終値・最高値は上の表と `runs.tsv` で確認できる。記録した損失は `training_losses.tsv` に保存し、本文用の別図は設けない。", "",
              "![CS-3とH100の検証精度の推移](validation_curves.png)", "",
              "## 残る疑問と範囲を広げる方向", "",
              "H100 products seed 43の840 stepでの検証精度は70.058%で、前の800 stepでの86.237%から一時的に低下した。全曲線を残すことで、この変動とseed間差を示した。入力順序・近傍の共有、bias構成、optimizer更新と評価手順の差を切り分けることが次の検証候補になる。固定形状H100の学習精度を揃えた比較が得られれば、実装差と実行系差の分離へ範囲を広げられる。今回の固定形状H100性能測定だけから学習精度の一致へ進めることはできない。", "",
              "学習と性能は別実行であり、性能測定の平均速度から同一精度への到達時間を推定していない。3 seedは初期値感度の観測を増やすが、別ジョブ・別環境での再現性やGraphSAGE以外への一般性は今後の評価対象となる。", "",
              "実装解釈の根拠: `context/fact/host-neighbor-sampling.md`、`context/fact/running-on-wse.md`。[共通設定・比較範囲](../../../../records/raw_logs/hpcasia/README.md#graphsageの共通設定と比較範囲) と [学習記録の形式](../../../../records/raw_logs/hpcasia/learning/README.md) から現在の記録を参照できる。PyGの入力反復・更新記録は `../gnn-modelzoo/src/cerebras/modelzoo/models/gnn/reference/pyg/train.py`、CS-3入力契約は各学習ログを参照する。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "records/raw_logs/hpcasia/RUNS.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/learning")
    args = parser.parse_args()
    runs, curves, losses, excluded, differences, sources, policies = load_runs(args.index)
    summary, paired = summarize(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [("runs",runs),("evaluations",curves),("training_losses",losses),
                       ("summary",summary),("paired_accuracy",paired),("excluded_runs",excluded),
                       ("configuration_differences",differences)]:
        write_tsv(args.output / f"{name}.tsv", rows)
    plot(curves, args.output)
    write_figure_tex(args.output)
    (args.output / "findings.md").write_text(findings(summary,paired,runs))
    manifest = dict(schema_version=1, script_sha256=sha256(Path(__file__)),
                    curve_validator_sha256=sha256(Path(__file__).with_name("plot_accuracy.py")),
                    matplotlib_version=matplotlib.__version__,
                    index_sha256=sha256(args.index), source_sha256=sources, gpu_policies=policies,
                    cohort="hpcasia_final, learning, CS-3/H100, arxiv/products, seeds 42/43/44",
                    run_count=len(runs), evaluation_count=len(curves),
                    standard_deviation="sample SD (ddof=1); independent unit is seed",
                    figures=["validation_curves.pdf", "validation_curves.png"],
                    figure_tex="validation_curves.tex",
                    figure_encoding="Each line is one seed; CS-3 solid, H100 dashed; no smoothing",
                    primary_accuracy="final configured evaluation", secondary_accuracy="max of 25 evaluations")
    (args.output / "manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(f"Validated {len(runs)} final runs, {len(curves)} evaluations, {len(losses)} recorded losses; wrote {args.output}")


if __name__ == "__main__":
    main()
