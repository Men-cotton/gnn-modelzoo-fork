# Worker実験の起動と監視

`benchmark_scripts/cerebras/run_worker_sensitivity.sh` と
`run_worker_campaign.sh` は共通の起動処理を使う。通常の起動では専用tmux内で
実験を開始し、コマンドは端末へ戻る。起動したホストが動作している間は、
端末やSSH接続を閉じても実験を継続する。tmuxはホスト再起動後の自動復旧機能ではない。

投入元にはtmuxと、準備済みのPython 3.11 / Cerebras 2.10環境が必要である。
既存のtmux設定・環境に依存しない専用serverを出力先ごとに使うため、
接続には起動時に表示する `tmux -L ... attach -t ...` をそのまま使う。
既存のtmuxセッションから実行する場合も、この起動方法を使える。

## 起動する

worker数だけを比較する初回確認の例:

```bash
bash benchmark_scripts/cerebras/run_worker_sensitivity.sh \
  --dataset arxiv --wsc-workers 1 --cache none \
  --workers 2 4 --repeats 3 \
  --warmup-steps 40 --measure-steps 400 \
  --budget-sec 54010 \
  --output model_dirs/hpcasia_r04/arxiv_none_worker_check
```

6 runを逐次実行する。`54010`は全runのクライアント時間の合計予算であり、
予測時間ではない。出力先は新規にする。`--workers` を省略した従来の既定値には
40が含まれ、40 workerではOOMの実績がある。

入力条件を変える比較と詳細診断を含める場合:

```bash
bash benchmark_scripts/cerebras/run_worker_campaign.sh \
  --output model_dirs/hpcasia_r04/arxiv_input_campaign
```

campaignは通常の感度測定をsensitivityへ委譲し、介入・診断も一つずつ実行する。
内側のsensitivityは `--foreground` で完了を待つため、各段階が別々にdetachされて
同時投入になることはない。測定対象・順序・失敗時の扱いは、それぞれの
[sensitivity](worker_sensitivity.md)・[campaign](worker_campaign.md)の説明に従う。

両コマンドで次の起動オプションを使える。

- `--tmux-session NAME`: 英数字・`-`・`_`でセッション名を指定する。省略時は自動で決める。
- `--foreground`: tmuxを使わず、呼び出し元で終了まで待つ。
- `--dry-run`: 計画・設定の生成だけを前景で行い、tmuxや学習クライアントを起動しない。
- `--help`: 学習を起動せず、起動オプションと実験オプションを表示する。

## 経過と終了結果を見る

出力先の `driver.log` に実験ドライバの出力を追記し、`launcher.json` に
起動・終了の状態と終了コードを残す。端末への復帰は起動処理の終了であり、
実験の成功判定には `launcher.json` と `study.json`／`campaign.json` を使う。

```bash
tail -f model_dirs/hpcasia_r04/arxiv_none_worker_check/driver.log
cat model_dirs/hpcasia_r04/arxiv_none_worker_check/launcher.json
```

tmuxへ接続した後は `Ctrl-b d` でdetachする。実験を中断する場合は接続先で `Ctrl-C` を使う。
実験ドライバが終了すると専用セッションも
終了し、ログと状態ファイルが残る。再起動時は同じ出力先を指定すると既存の
study／campaign再開規則に従う。同じ出力先の実行中ドライバを重複起動できない。
campaign配下では `launcher.log` が内側の起動処理を、`driver.log` が感度測定を記録する。

## csctlのlabelで識別する

各trialの `trainer.init.backend.cluster_config.job_labels` に以下を付与する。
SDKがそのtrialのcompile／execute jobへ渡すため、投入後にjob IDを調べて
手動でlabelを付け直す操作は不要である。以下の8項目を実験条件に合わせて更新し、
それ以外の既存の独自labelは保持する。

| Label | 内容 |
| --- | --- |
| `gnn-model` | モデル名 |
| `gnn-dataset` | `ogbn-arxiv`／`ogbn-products`等のdataset名 |
| `gnn-cache` | `none`／`zero`／`partial-N`／`full`。GraphCacheを作らない条件と割合0の条件も区別する |
| `gnn-mode` | sensitivity／intervention／diagnostic等の実験種別 |
| `gnn-workers` | PyTorch DataLoader子worker数 |
| `gnn-repeat` | 反復番号 |
| `gnn-trial` | study内のtrial識別子。campaignでは段階も区別する |
| `gnn-study` | study／campaignの出力先から決める識別子 |

SDKのlabelはkey/valueとも1–63文字で、英数字で始まり終わる必要がある。
長い値や利用できない文字を含む値は、hash付きの安定した表現に変換する。
完全な出力パスや元の設定はローカルの計画・状態・YAMLへ保持する。
campaign配下は共通の `gnn-study` でまとめて追える。

```bash
csctl get jobs
csctl get jobs -a -l gnn-dataset=ogbn-arxiv,gnn-mode=sensitivity,gnn-workers=4
```

`gnn-study` は生成された `params.yaml`／`preview_w*.yaml` の値を使って絞り込める。
LABELS列と `-l key=value` による絞り込みは
[Cerebrasのcsctl仕様](https://training-docs.cerebras.ai/rel-2.10.0/cluster-monitoring/cerebras-job-scheduling-and-monitoring/cli-for-job-monitoring-csctl)
に従う。compile cache再利用時は新しいcompile jobを作らない場合があるため、
ジョブIDの個数と独立した学習run数は分けて数える。
