# HPC Asia の固定設定・3 seed 測定

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
