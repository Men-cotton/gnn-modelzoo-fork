# Worker実験の起動と監視

`benchmark_scripts/cerebras/run_worker_sensitivity.sh` と
`run_worker_campaign.sh`、`run_learning_campaign.sh` は共通の起動処理を使う。通常の起動では専用tmux内で
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
予測時間ではない。出力先は新規にする。worker数は投入元と実行Workerの
CPU・メモリ割当に合わせて明示する。

入力条件を変える比較と詳細診断を含める場合:

```bash
bash benchmark_scripts/cerebras/run_worker_campaign.sh \
  --output model_dirs/hpcasia_r04/arxiv_input_campaign
```

campaignは通常の感度測定を `autotune.py --mode sensitivity` へ委譲し、
介入・診断も一つずつ実行する。各段階を同じPythonで直接起動し、完了を待つ。
tmuxと起動状態の管理はcampaign全体に一つだけ設ける。測定対象・順序・失敗時の扱いは、それぞれの
[sensitivity](worker_sensitivity.md)・[campaign](worker_campaign.md)の説明に従う。

これらのコマンドで次の起動オプションを使える。

- `--tmux-session NAME`: 英数字・`-`・`_`でセッション名を指定する。省略時は自動で決める。
- `--foreground`: tmuxを使わず、呼び出し元で終了まで待つ。
- `--dry-run`: 計画・設定の生成だけを前景で行い、tmuxや学習クライアントを起動しない。
- `--help`: 学習を起動せず、起動オプションと実験オプションを表示する。

Pythonのドライバを直接呼ぶ場合は前景実行が既定で、`--detach` でtmux起動を選べる。
シェルスクリプトは環境の選択と既定オプションを指定する入口である。
入口で選んだPythonを環境検証・各trialまで使うため、準備済み環境で実行する。
前景実行では結果を端末と `study.json`／`campaign.json` で確認し、終了コードは呼出元へ返る。

## 経過と終了結果を見る

tmux起動では出力先の `driver.log` に実験ドライバの出力を追記し、`launcher.json` に
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
campaign配下の感度測定は各段階の `driver.log` と `study.json` に記録する。
起動管理の実装と新しいbenchmarkの追加方法は
[benchmarkの起動規約](../../../../../../benchmark_scripts/README.md)を参照する。

## csctlのlabelで識別する

各trialの `trainer.init.backend.cluster_config.job_labels` に `run` と `study` を付与する。
SDKがそのtrialのcompile／execute jobへ渡すため、投入後の手動操作は不要である。
以前の8つの `gnn-*` labelは置き換え、それ以外の独自labelは保持する。

例えば通常の感度測定は次のように表示する。

```text
run=sage-arxiv-sens-w4-pf2-persist-r1,study=0123456789
```

| 表記 | 意味 |
| --- | --- |
| `sage` | GraphSAGE |
| `arxiv` / `products` | ogbn-arxiv / ogbn-products |
| `sens` / `diag` | 感度測定 / 診断 |
| `workers` / `loader` / `confirm` | worker探索 / loader設定探索 / 確認測定 |
| `w4` / `pf2` / `persist` | worker数4 / prefetch factor 2 / workerを維持 |
| `nopersist` / `pf0` | workerを維持しない / prefetchなし |
| `r1` | 反復1 |
| `cache0` / `cache0.5` / `cache1` | GraphCacheの割合。省略時はGraphCache自体を作らない |
| `vs16` / `static1` | worker 16との比較組 / static batchを1個再利用 |

比較対象のworker数、baseline／selected、入力条件の介入も `run` に残す。
異なる段階の参照runを同じ条件名にまとめない。
`run` の値は60文字以内で、通常の条件は意味を読める短縮形で表す。
未知の長い名前やSDKで使えない文字だけ、末尾に識別hashを付けて収める。
完全な出力パスと設定は計画・状態・YAMLに保持する。
`study` は出力先の絶対パスから決める10桁の識別子で、campaign配下で共通である。

```bash
csctl get jobs
csctl get jobs -a -l study=0123456789
csctl get jobs -a -l run=sage-arxiv-sens-w4-pf2-persist-r1
```

実際の `study` は生成された `params.yaml`／`preview_w*.yaml` の値を使う。
LABELS列と `-l key=value` による絞り込みは
[Cerebrasのcsctl仕様](https://training-docs.cerebras.ai/rel-2.10.0/cluster-monitoring/cerebras-job-scheduling-and-monitoring/cli-for-job-monitoring-csctl)
に従う。compile cache再利用時は新しいcompile jobを作らない場合があるため、
ジョブIDの個数と独立した学習run数は分けて数える。
