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

## JSON catalogからplanを作る

ジョブ一覧・set・PyG profile・HPC Asiaのseed/run・GPU input sweepの展開表は
[`jobs.json`](jobs.json)に置く。共通入口は既存launcherの `jobs` サブコマンドである。

```bash
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.tools.benchmark_launcher jobs --list
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.tools.benchmark_launcher jobs --job fixed-shape --help
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.tools.benchmark_launcher jobs \
  --set hpcasia-all --output /tmp/hpcasia-preview --dry-run
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.tools.benchmark_launcher jobs \
  --job gpu-input --output /tmp/input-preview --phase workers --workers '2 4' --json
```

`--dry-run` は検証済みargvを表示し、`--json` は同じplanをJSONで表示する。
いずれもschedulerへの投入や出力directory作成を行わない。これらの指定を外すと投入する。
全jobの設定・引数を先に検証し、shellを介さずargvで実行する。設定は実行側と同じ
`extends` 解決・重複YAMLキー検査を使う。qsubの `-v` 値にcomma/newline/NULを認めない。
fixed-shapeのconfig pathに含まれる空白は保持する。HPC Asia/sweepは従来どおり空白を拒否する。

既存のPegasus submitterはこの入口への互換wrapperである。non-GNNは既存run/campaignが
入力と設定を準備し、dry-runと実投入の両方で共通qsub builder/executorを使う。
CSXの3 campaign wrapperもcatalogのdriver定義を使い、相対出力pathは呼出元から解決する。
`run_modelzoo.sh` の8設定も `csx-config` としてcatalogに置く。診断worker一覧は
`presets.worker_diagnostics.workers` に置き、診断実行も共通launcherで監督する。
シェル入口の診断 `--dry-run` も固定sourceのplanだけを表示する。たとえば
`jobs --job csx-worker -- --foreground --dry-run ...` の `--` 以後はcampaignへ渡す。
campaign自身が研究条件・dry-run・予算・trial停止・lock・journalを管理し、
既存 `benchmark_launcher.launch` がtmuxを管理する。PBS payloadとrun.pyの学習条件は変えない。
HPC AsiaのPython planとPegasus planは同じcatalogのrun一覧を参照する。

新しいjob/setはcatalogへ追加する。新しい実行形式だけを `job_plan.py` へ追加し、
個別submitterへqsub処理を戻さない。catalogの未知field・未知job/set、不正設定、引数境界、
全plan検証後の投入は `benchmark_scripts/tests/test_job_plan.py` で検証する。

## run単位のsource固定

通常は現在のbranchの`HEAD`をrun開始時に一度だけfull commit hashへ解決する。
set内の全job・並列worker・内側のcampaignは同じhashとsourceを使う。
未commitの変更は実行sourceに入らない。過去sourceはrun-levelの
`--revision <commit-hash>`を明示した場合だけ選ぶ。branch名やjob定義内のrevision/hashは拒否する。

`--json`は複数jobでも単一JSON documentを返す。`resolved_commit`、`source_root`、
`manifest_path`、`jobs`、固定inputのdigestを含む。従来の単一job用`cwd`と`commands`も残す。
複数jobのcwdは`jobs[*].cwd`で確認する。previewはsource/manifest/outputを作らず、
表示した将来のpathはまだ存在しない。`--prepare-only`はsnapshotとmanifestを準備し、
実際のimport元を検証して終了する。実行するときは新しいrun directoryを使う。

```bash
# 検証済みplanのみ。HEADはこのrunの全jobで同一。
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.tools.benchmark_launcher jobs \
  --set hpcasia-all --output model_dirs/hpcasia_preview --json
# snapshot準備のみ。CSX clientやschedulerは起動しない。
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.tools.benchmark_launcher jobs \
  --job csx-config --config-index 1 --run-dir model_dirs/source_inspection_01 --prepare-only
# 明示的な過去commitのpreview。ここではsource固定実装のcommitを例示。
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.tools.benchmark_launcher jobs \
  --revision e02e1452237af6e77c0a6e34c737a23c90b7d424 \
  --job fixed-shape --config src/cerebras/modelzoo/models/gnn/configs/fixed_shape_gpu/arxiv.yaml --json
```

既定保存先は`model_dirs/job_runs/<UTC>-<unique>/`の一意なdirectoryで、`source/`は
`git worktree add --detach`で作る。元checkoutをcheckout/switch/resetしない。
tracked sourceだけを取り出し、sourceファイルをread-onlyにする。既存runは上書きしない。
manifestには全jobのhash/cwd/argv、実dispatch、準備失敗、input digest、bindingを記録する。
legacy source handoffは`exec`でpayloadへ移るため、完了状態は既存campaign/launcherのjournalも参照する。
必要なpayloadやhandoff APIが過去commitにない場合は検証で失敗し、最新sourceへfallbackしない。

PBSはsnapshotをqsubのcwdとして渡すので、`PBS_O_WORKDIR`とscriptが同じsourceを指す。
`PYTHONPATH`はsnapshotの`src`へ固定し、既存editable環境でも`__file__`を新規processで確認する。
`common.sh`はsourceを照合し、uvにはsnapshotのprojectと既存`.venv`を使わせる。
`UV_NO_SYNC=1`/`UV_OFFLINE=1`および`--no-sync`で依存更新を行わない。
CSX設定の`python_paths`/`mount_dirs`もsnapshotとinput/output bindingを参照する。
実schedulerではproject・run directory・依存環境・dataが計算nodeから同じpathで見える必要がある。
`/tmp`の共有は仮定しない。`--run-dir`を指定する場合も共有pathを選ぶ。

`.venv`と既存dataは可変な依存環境・入力として別記録し、source hashによる固定対象に含めない。
run内の`model_dirs`は専用outputへ向ける。non-GNNの既存data cacheと診断の保存済みbaseは
元projectから参照する。`--base`/`--base-config`/`--config`は別token形式も`=PATH`形式も
同じ処理で固定する。tracked YAMLとその`extends`は選択commitから読み、外部YAMLは解決済み内容を
`inputs/<sha256>.yaml`へ保存する。output/data引数は入力configとしてコピーしない。
`launch.json`を伴うprepared configや外側からの`--execute-config`は、新しいsource runへ
持ち越さず投入前に拒否する。pinned campaign内で再準備する通常のnon-GNN経路は維持する。

legacy non-GNN/diagnostics/sensitivity入口の保存先指定は`--source-run-dir`、過去sourceは
同じ`--revision`を使う。低水準Python driverを直接呼ぶ科学条件のdry-runには既存のconfig生成動作が
残るため、出力を書かない投入previewには`benchmark_launcher jobs`またはシェル入口を使う。
汎用launcherへ渡す任意の外部コマンドまでsource固定する機能ではない。

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

## Cerebras の GraphSAGE 表現比較

実行手順は既存の `cerebras/` 配下に置く。

- [Tree 特徴共有](cerebras/feature_sharing/README.md): dense/dictionary 入力の比較。
- [History と生特徴 mean 前集約](cerebras/history_embedding/README.md): 固定形状の学習・評価。

実行モジュールは `cerebras.modelzoo.models.gnn.tools` に置く。データ読み込みは既存の
`BaseGraphDataSource` を使い、未準備なら `download_datasets.py` の実行を案内して停止する。`csx-shared-tree`、`csx-history-pilot`、
`csx-history` は既存 catalog・source 固定・tmux 起動を使う。
ALCF の SDK 環境と既存のデータ準備処理を使い、別の cluster JSON は設けない。
