# 一度の起動で行う入力パイプライン調査

二重DataLoader修正後のarxivでは、2から4 workerで平均15.9%の増速があり、
4での飽和は未確認。一方、40 workerの診断はstreamerのOOMで停止した。
このコマンドは、既存のworker感度測定、入力条件を変える比較、詳細診断を
まとめて実行し、最後に結果をZIPへ保存する。

ALCF側に変更を反映したリポジトリの直下で実行する。

```bash
bash benchmark_scripts/cerebras/run_worker_campaign.sh
```

既定はarxiv、24時間の学習クライアント合計予算、4／8／12／16 worker。
毎回新しい `model_dirs/hpcasia_r04/campaign_<UTC>/` を作る。
共有領域にある既存のリポジトリとPython 3.11 / Cerebras 2.10環境を使う。
ターミナルを切断する場合は既存のtmuxセッション等から起動する。
実行中の追加操作は不要。学習クライアントは一つずつ動かす。
失敗・タイムアウトは記録して次へ進む。ただしクライアント終了だけでは遠隔ジョブの終了を
保証できず、残ったジョブとの競合が後続runに影響する可能性はある。

## 何を測り、何を判別するか

| 比較・観測 | 判別する対象 |
| --- | --- |
| 通常入力の4対8、4対12、4対16 | worker追加による速度の伸び、実行時刻による基準性能の変動 |
| 4 workerでprefetch_factor=1対通常の2 | 先読み量の速度・メモリへの寄与 |
| 4 workerでpersistent_workers=false対true | 23バッチごとの反復境界におけるworker作り直しの影響 |
| 4 workerでcache_fraction=1.0対null | ホストのGraphCache特徴量取得経路の影響 |
| 4 workerでstatic_batch_cache_size=1対0 | サンプリング・特徴量取得を毎回行うコストを除いた入力経路 |
| 各worker数の80 step診断 | 実効worker数、子PID、sampling/gather、親next、スレッドCPU、CPU quota |
| 診断中の独立観測プロセス | CPU・page fault、RSS/PSS、memory cgroup使用量・上限・OOMカウンタ、共有メモリ空き、CPU/メモリ/I/O pressure |

GraphCacheはホスト上の特徴量キャッシュで、WSE SRAM常駐とは区別する。
固定バッチは同じバッチを繰り返す原因切り分け用の実験であり、通常の学習精度の比較には用いない。
その性能差にはデータ再利用・メモリ局所性の変化も含まれる。
PSSは共有ページを各プロセスに按分した値。RSSの単純合計を実メモリ使用量にしない。
共有メモリの空きは `/dev/shm` マウント全体、host_pressureはホスト範囲であり、
そのrunだけの量と断定しない。cgroupの使用量には観測プロセス自体も含まれる。

## 既定の順番と測定区間

1. 通常の4 workerを3回。
2. 4種類の入力条件を各3回。反復ごとに順番を反転する。
3. 4 workerの診断を1回。
4. 4対8 workerを各3回、8 workerの診断を1回。
5. 4対12 workerを各3回、12 workerの診断を1回。
6. 4対16 workerを各3回、16 workerの診断を1回。

合計37 run（通常測定21、入力条件の比較12、診断4）。大きいworker数で失敗しても、
先に完了した対照・入力条件の結果が残る。各ペアの通常測定は既存の
`run_worker_sensitivity.sh` を呼び、4との実行順を反復ごとに反転する。
4種類の介入はworker数4、WSC入力Worker数1、batch_size=4096、fanouts=[15,10,5]に固定する。
診断を有効にするのは4つの診断runのみ。

通常・入力条件の比較は40 stepのウォームアップ後、step40–440の400 stepを測る。
診断はstep40–80の40 stepで、計時・PSS読み取り等の負荷を含む。
公称seed枠/秒はパディングを含む。全体速度と親next時間の差から、
純粋な通信時間やWSE計算時間を求めない。WIO時間の正規化には対象チャネル数の別確認が必要。

Grafanaの系列は実行後に別途取得する。対応するジョブID・時刻は生ログとSDK出力に残る。
実行中にしか採れないプロセス別PSSや入力生成の計時は、上記の診断runで保存する。

## 出力・失敗・再開

`plan.json` と各 `params.yaml` に計画、`campaign.json` に実行状態、
`runs.csv` と `summary.json` に通常・介入・診断を区別した測定値が残る。
各runには生ログ、SDK出力、設定、メタデータがある。診断では
`worker_diagnostics/loader-<host>-<pid>-<uuid>/` にプロセス別JSONLと
`resources-<parent-pid>.jsonl` ができる。投入元と遠隔Workerの記録はhostnameで区別する。
バッチ形状・dtype・論理bytesも記録するが、これは同時常駐量ではない。

resource観測はDataLoader反復開始後に独立プロセスで5秒間隔、最大120回。
親プロセスの終了・PID再利用でも終了する。バッチ取得が停止中でも採取できるが、
コンテナ全体がOOM killされる場合は観測プロセスも停止し得る。
読み取れないprocfs/cgroup項目はerrorを残す。隠れた祖先cgroupの上限は未確認として扱う。

通常終了・実験失敗・予算切れで集計し、出力ディレクトリの隣に時刻付きZIPを作る。
`--no-archive` で省略可能。予算は学習クライアント時間で、最後のZIP作成は別。
ジョブ上限7200秒、クライアント上限9000秒。総予算は所要時間予測ではない。

予算切れは同じ `--output` と設定で `--budget-sec` を増やして再開できる。
完了・失敗済みのrunを再投入しない。設定・ソース・環境が変われば新しい出力先を要求する。
失敗・タイムアウトしたrunは記録してスキップし、予定済みの後続runへ進む。
同じ条件の別反復も予定どおり実行する。通常の感度測定には内部で
`--continue-on-failure` を渡し、その段階内でも継続する。
失敗があった場合は全予定を消化しても状態を `completed_with_failures`、終了コードを2とし、
成功扱いにはしない。最後に集計・ZIPを保存する。
Ctrl-C／SIGTERMによる中断では後続を投入しない。中断した出力先の再開は拒否するため、
遠隔ジョブの状態を確認し、未実施範囲を選んで新しい出力先で起動する。
遠隔ジョブが終了したとの自動承認や、他のジョブのキャンセルは行わない。

## 事前確認とローカル検証

```bash
bash benchmark_scripts/cerebras/run_worker_campaign.sh \
  --dry-run --output /tmp/worker-campaign-preview
```

37 runの計画と設定だけを生成し、学習は投入しない。既存結果に重ならない出力先を使う。
`--workers 4 8` で探索範囲を絞れ、`--controls` の後に候補名を並べて介入を選べる。
候補名なしの `--controls` は介入を省略する。既定の40 workerへの再投入はない。

GNNディレクトリで次を実行する。データ取得やCSXは不要。

```bash
OUTDATED_IGNORE=1 uv run --no-sync python -m unittest discover -s tests -p 'test_worker_campaign.py' -v
OUTDATED_IGNORE=1 uv run --no-sync python -m unittest discover -s tests -p 'test_worker_diagnostics.py' -v
```
