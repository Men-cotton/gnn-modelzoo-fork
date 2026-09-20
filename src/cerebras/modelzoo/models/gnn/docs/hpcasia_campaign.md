# HPC Asia の固定設定・3 seed 測定

## Pegasus: 30 runと既存worker比較54 run

[GPU比較方針と一次資料](gpu_comparison_policy.md)に従い、PyG標準のモデル・
サンプリング・AdamWを維持して、公称設定・biasの正則化対象・初期AMP scaleを揃える。
CS-3固有の更新式・固定近傍選択・二重biasには追従しない。完全に同じ学習軌道の比較とは区別する。

Pegasusで更新後、以下を実行する。`--dry-run`を付けると投入コマンドだけを表示する。
30 runは各runを独立したPBS jobにする。各jobの制限は4時間、クライアントは3時間55分。
`--hours 3`も選べる。待ち時間を含むSDK制限をGPU側に流用しない。

```bash
git pull --ff-only origin main
bash benchmark_scripts/pegasus/submit_hpcasia_nqsv.sh \
  --dataset all --hours 4 --compile \
  --output "model_dirs/hpcasia_final/pegasus_native_$(date -u +%Y%m%dT%H%M%SZ)"
```

2 dataset × 3 seed × (learning 1 + throughput 3 + cache 1) = 30 run。
productsの性能測定は40 step除外後の1600 step、arxivは800 step。
出力は `<output>/<dataset>/seed_N/<kind>_rN/` に分かれ、それぞれに
`campaign.json`、`params.yaml`、`train.log`、`result.json`、`summary.json`を保存する。
各jobのsummaryはそのrunだけを含み、30 job全体の自動集計ではない。
学習runには `learning_curves.json` も保存し、当該stepのlossと区間平均を分ける。

既存worker比較54 runの再実行は別の18 PBS job（各3反復）である。
新規30 runと両方実行する場合は計84 run。旧結果は上書きしない。

```bash
bash benchmark_scripts/pegasus/submit_gpu_input_sweep_nqsv.sh \
  --dataset all --backend pyg --phase workers \
  --workers '2 4 8 12 16 24 32 40 48' --compile \
  --output "model_dirs/hpcasia_gpu_input/pyg_native_$(date -u +%Y%m%dT%H%M%SZ)"
```

両方ともcompileを明示し、可変サイズ入力向けのdynamic shapeを使う。
30 run側は `--no-compile`、worker比較側は `--compile` の省略でeagerにできる。
混ぜて集計しない。worker比較の既存PBS上限は24時間のまま、各クライアントの既定は1800秒。
30 runの3/4時間指定とは独立している。実際のGPU性能は実行後のログで確認する。

各jobの失敗・timeoutや途中のqsub失敗では、まず投入済みjobと出力を確認する。
一括コマンドを新しい出力先で繰り返すと、投入済みjobも重複して投入される。

## CS-3: 既存の逐次campaign

```bash
bash benchmark_scripts/cerebras/run_hpcasia_campaign.sh
```

既存の tmux ランチャーで1つのドライバを起動する。既定の対象は ogbn-arxiv。
DataLoader の `num_workers: 4`、`prefetch_factor: 2`、
`persistent_workers: true`、CSX の入力 Worker replica 数1を固定する。
各クライアントは新規モデルから開始し、個別の設定・モデル出力先・ログを持つ。
スクリプトの作成・dry-run は実機ジョブを投入しない。

実行順は seed 42 の以下5実行、seed 43 の5実行、seed 44 の5実行、計15実行。
`--seeds 42 123 456` で後続のseedを変更できる。
モデル初期化とtrain/valid双方のsamplerに同じseedを設定する。

| 順序 | 実行 | 条件 |
| --- | --- | --- |
| 1 | `learning_r1` | 500ステップ学習、20ステップごとにvalid split全体を検証 |
| 2–4 | `throughput_r1`～`r3` | キャッシュ無効、40ステップ除外後に800ステップ計測 |
| 5 | `cache_r1` | 特徴量キャッシュ `cache_fraction: 1.0`、同じ区間を1回計測 |

学習曲線はIA³のaccuracy profileと同じ `eval/masked_accuracy` を使い、
検証step・平均loss・正解率、training loss、最終検証正解率を保存する。
arxivの500ステップ・20ステップ周期は既存の
`configs/params_graphsage_ogbn_arxiv.yaml` に対応する。
現在の `learning_campaign` と同じ収集処理を使い、trainの入力順序を継続し、
末尾バッチを含める。旧設定の `steps_per_epoch: 20` による打ち切りは採用しない。
正解率の高低では後続実行を止めず、欠測・非有限値・クライアント失敗時は停止する。

スループット測定は検証・checkpoint保存・詳細worker診断を無効にする。
生ログのstep40と840の時刻差を使い、学習対象頂点数/秒、教師信号のある対象数/秒、
公称バッチ枠数/秒を併記する。対象数は入力順序から算出し、デバイスカウンタ値と区別する。
前半・後半の差が2%を超えた場合も記録を保持し、完了実行の集計から除外しない。
キャッシュの効果は測定結果として報告し、小さい効果を合格条件にしない。
静的バッチ再利用はすべて無効にする。

`--dataset products` では1000ステップ学習・40ステップ周期の検証を既定とする。
`--learning-steps`、`--eval-every`、`--warmup-steps`、`--measure-steps` で変更できる。

## 出力と確認

既定の出力先は `model_dirs/hpcasia_final/arxiv_<UTC>/`。

- `campaign.json`: seed順の実行状態、環境・ソースの指紋、各設定・測定・ジョブID。
- `seed_42/learning_r1/learning_curves.json`: 検証精度とlossの曲線。後続seedも同形式。
- 各実行の `params.yaml`、`train.log`、`result.json`、`model/`。
- `summary.json`: seed別の最終検証精度、主スループット3反復の値・平均・標本標準偏差、
  キャッシュ有効時の値と同seedの主スループット平均に対する比。
- `launcher.json`、`driver.log`: tmux接続情報、ドライバの出力と終了状態。

同じseedの3反復と3つのseedの違いを分けて集計する。

```bash
bash benchmark_scripts/cerebras/run_hpcasia_campaign.sh \
  --dry-run --output /tmp/hpcasia-final-preview
```

15件の設定と `plan.json` だけを書き出す。測定には別の空の出力先を指定する。
`--foreground` は現在のシェルで実行する。
tmux接続と状態確認は [起動と監視](worker_launch.md) と共通。

合計予算は既定172800秒、クライアントごとの上限9000秒、SDK job上限7200秒。
合計予算が尽きた場合、同じ `--output` と条件で `--budget-sec` を増やすと、
完了済み実行をスキップして再開する。実装・環境・条件が変わった場合は新しい出力先を使う。
失敗・中断・timeout後は遠隔ジョブの状態を確認し、新しい出力先で開始する。
