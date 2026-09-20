# Benchmarkの起動

シェルの入口でPython環境を選び、実験ドライバが条件・順序・予算・再開を管理する。
tmuxが必要な実行では、外側の一箇所で
[`benchmark_launcher`](../src/cerebras/modelzoo/tools/benchmark_launcher.py)を使う。
共通launcherはコマンド、作業ディレクトリ、ログ、起動状態を扱う。

| 入口 | 実験の実行方法 |
| --- | --- |
| `cerebras/run_learning_campaign.sh` | 学習曲線を保存後、設定探索を逐次実行。精度の判定は研究者が行う。既定でtmux起動 |
| `cerebras/run_worker_sensitivity.sh` | 指定したworker数を反復測定。既定でtmux起動 |
| `cerebras/run_worker_campaign.sh` | 感度測定・入力条件の比較・診断を逐次実行。既定でtmux起動 |
| `cerebras/run_modelzoo.sh` | 選択した設定で一つの学習を前景実行 |
| `cerebras/run_worker_diagnostics.sh` | 保存した設定から診断を逐次実行 |
| `cerebras/run_non_gnn.sh` | 入力準備・条件ごとの反復測定・集計まで逐次実行。既定でtmux起動 |
| `pegasus/submit_*` | `qsub`で実行をスケジューラへ渡す |

Worker実験の既存コマンド、監視、CSX job labelは
[Worker実験の起動と監視](../src/cerebras/modelzoo/models/gnn/docs/worker_launch.md)を参照する。
Pegasusの独立した30 runと54-run worker再実行は
[HPC Asia測定](../src/cerebras/modelzoo/models/gnn/docs/hpcasia_campaign.md)、
揃える設定と残す実装差は
[GPU比較方針](../src/cerebras/modelzoo/models/gnn/docs/gpu_comparison_policy.md)を参照する。

## 別のbenchmarkをtmuxで実行する

任意のコマンドを `--` の後へ渡せる。新しい実験名を使う際にlauncherへの登録は不要である。
例えば、前景実行の単一Model Zoo設定をtmux内で実行するには次を使う。

```bash
uv run --no-sync -- python -m cerebras.modelzoo.tools.benchmark_launcher \
  --name modelzoo --output model_dirs/launches/modelzoo_01 \
  -- bash benchmark_scripts/cerebras/run_modelzoo.sh --config-index 1
```

launcherの `--output` はログ・起動状態の保存先である。
コマンド側の実験出力先とは独立して選べる。
`--` より後の引数はそのままコマンドへ渡し、呼出元の作業ディレクトリと環境を引き継ぐ。

起動時に表示される `Attach:` のコマンドで専用tmuxへ接続する。
保存先の `driver.log` と `launcher.json` で経過・終了コードを確認できる。
launcher自身の `--foreground` はtmuxを使わずにログと起動状態を記録する。
launcherは渡されたコマンドの終了を記録するので、コマンドがさらにジョブを投入する場合は
各ジョブの結果を実験側の記録で確認する。non-GNNでは `client_status.json`、
Pegasusではスケジューラのjob状態が該当する。

GNN campaign・Worker実験・non-GNNのCSXシェル入口は自動でtmuxを使う。通常はそのまま呼び出す。
外側から監視する必要がある場合は、入口に `--foreground` を渡して前景で完了を待つ。
PBSの実行はスケジューラが管理するため、PBS payloadは前景で実行する。

non-GNNの全条件は[一括実験](non_gnn/README.md)で準備・検証・反復・集計する。
一つのprepared runの実行処理をCSXとGPUで共用し、GPUは各PBS jobの終了時にも
campaignの集計を更新する。新規出力先を要求するnon-GNNでは、共通launcherの記録を
`<output-dir>.launcher/` に置いて実験データと分ける。

## 新しい測定項目を追加する

既存実験の条件追加は実験側の設定や計画生成へ追加する。Worker感度測定の探索・予算・再開は
`Study`、backend固有の設定・実行コマンド・測定値抽出は `autotune_backends.py` が担当する。
campaignの内部段階は同じ `sys.executable` でドライバを直接起動して完了を待つ。

独立したドライバを作る場合も、最初は上の汎用CLIで起動できる。
ドライバのCLIへtmux起動を組み込む場合は次の境界を保つ。

1. `benchmark_launcher.add_arguments(parser)` で起動オプションを追加し、ドライバ自身で引数を検証する。
2. `--detach` のときだけ、検証後の実験出力先とコマンドを `benchmark_launcher.launch()` へ渡す。
3. 子コマンドには `--foreground` と確定した出力先を渡す。helpとdry-runは前景で処理する。
4. `detach`・`tmux_session` は測定条件や再開の同一性判定から除外する。

launcherのlockは起動と監視プロセスを保護する。実験のlockとjournalはドライバが管理し、
直接実行でも同じ出力先を同時に更新できないようにする。
起動の仕組みと実験条件を分けることで、測定追加時の変更は実験側に収まる。

## 起動処理の検証

実tmuxを含むテストは専用の一時ディレクトリとsocketを使う。

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover \
  -s benchmark_scripts/tests -p 'test_launcher.py' -v
```

条件・予算・再開のテストは対応する実験のtest directoryで実行する。
