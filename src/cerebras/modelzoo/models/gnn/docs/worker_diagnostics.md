# GNN入力Workerの診断

ModelZoo / Cerebras PyTorchのneighbor入力経路で、実効worker数、親子プロセス、
スレッド別CPU時間、可視範囲のCPU quota、入力生成・受け取り時間を記録する。
管理者権限や追加パッケージは不要。Linuxの`/proc`とcgroupを読み取る。
CSL SDK、Cerebras PyTorch本体、クラスタ設定には変更を加えない。

## R04の診断を一つのコマンドで実行する

診断コードを反映したALCF側のリポジトリ直下で実行する。

```bash
bash benchmark_scripts/cerebras/run_worker_diagnostics.sh
```

既定では `model_dirs/hpcasia_r04/arxiv_none/sensitivity_w02_p2_s1_r1/params.yaml`
を読み、2 workerと40 workerを各80 step、逐次実行する。ほかの場所に保存した設定を
使う場合は `--base /path/to/params.yaml` を付ける。相対パスは呼び出し元を基準にする。
`--dry-run` を付けると設定生成とコマンド表示まで行い、ジョブは投入しない。
dry-runでは投入元のCPU数による拒否も省略する。実行時は40 worker設定を事前検査する。

毎回新しい `model_dirs/hpcasia_r04/worker_diag_<UTC>_<unique>/` を作る。
設定の継承を解決した `base.yaml`、元設定のパス、コードrevision、投入元hostnameと、
`w02/params.yaml`、`w40/params.yaml` を保存する。変更する設定はmodel_dir、max_steps、
num_workers、worker_diagnosticsのみ。診断は各プロセスの最初の80バッチを計時し、
5秒以上の間隔で最大80回のスナップショットを採る。

各 `w*/` に `train.log`、SDK出力の `model/`、`worker_diagnostics/`、
step40から80の対象頂点数/秒と公称枠数/秒を集計した `throughput.json` が残る。
診断有効時のthroughputは追加の計測コストを含むため、原稿の主性能値とは分けて扱う。
起動直後を除く分析にはstep40以降の時刻に対応する記録を使う。
CSXでの診断出力には、既存の共有領域とコードがWorkerから見えることが必要である。

多数の条件とメモリ観測を一度に集める場合は [一括調査](worker_campaign.md) を使う。
既定は4／8／12／16 workerで、worker数の少ない条件から実行する。

クライアントまたは集計が失敗すると、その場で停止する。自動再開・再試行は行わない。
中断時は保存ログのjob IDでリモートジョブの状態を確認してから再実行する。
再実行では新しい出力先を作り、2 workerから始める。

## 有効化

既存の `trainer.fit.train_dataloader` に次を追加する。既定は無効。
同じ設定をnative `fixed_shape_gpu.py` でも使用できる。

```yaml
worker_diagnostics:
  enabled: true
  output_dir: /absolute/shared/path/worker-diagnostics/run-001
  max_batches: 16
  snapshot_interval_seconds: 1.0
  max_snapshots: 30
  resource_monitor: false
  resource_monitor_pss: false
```

`output_dir` は投入元とWorkerの双方から書ける既存の共有領域に置く。
CSXでは既存の `mount_dirs` / `python_paths` を通じ、このリポジトリの変更と
出力先がWorkerに見えるようにする。共有領域のマウント設定は通常の入力コードと同じ。
相対パスはfactoryを実行するプロセスの作業ディレクトリを基準とするため、絶対パスを推奨する。
full_graphモードでの有効化は設定エラーになる。

出力は `output_dir/loader-<hostname>-<factory-pid>-<uuid>/<pid>.jsonl`。
factoryごとにディレクトリを作り、プロセスごとに別ファイルへ追記する。
既存の出力を切り詰めない。出力先とPIDはログにも記録する。
出力不能ならそのプロセスで警告を一度出し、診断書き込みを停止して学習を継続する。
バッチ値、ラベル値、認証情報、環境変数全体は保存しない。

Cerebras PyTorch 2.10.0は、投入元で入力仕様を調べるDataLoaderを
`num_workers=0` に変更する。Workerでは入力factoryを別に実行する。
そのため、`loader_created` の設定と `iterator_started` の設定を照合し、
hostname、PID、実際の `worker_initialized` を使って遠隔Workerの記録を選ぶ。
投入元の0-worker記録だけで遠隔の実効値を判断しない。
GNNは通常のPyTorch DataLoaderを返し、TrainerがCerebrasラッパーを管理する。
遠隔の `iterator_started` と `worker_initialized` で、指定worker数に対応する
子PIDが実際に起動したことを確認する。

## 記録の読み方

| event | 記録するもの |
| --- | --- |
| `loader_created` | 生成時のnum_workers等、PyTorch/Cerebras PyTorch版、affinity、PyTorchスレッド設定、ソースファイルのパス・SHA-256、cgroup |
| `iterator_started` | 反復開始時の実効設定、PyTorch iteratorが保持する子PID、反復番号 |
| `worker_initialized` | 実際の子PID/PPID、worker ID、実効worker数、affinity、子のPyTorchスレッド設定 |
| `batch_generated` | Datasetのバッチindex、プロセス内の計測番号、生成開始・終了、wall/CPU時間。通常入力ではsampling/gatherの内訳も記録 |
| `batch_received` | 親の`next(iterator)`の開始・終了、wall/CPU時間、反復内の取得番号 |
| `snapshot` | 親と直接のDataLoader子のプロセス／スレッドCPUカウンタ、スレッド名・状態・待機先、cgroup、収集時間 |
| `batch_layout` | 各生成プロセスの最初のバッチのshape・dtype・論理bytes。テンソル値は保存しない |
| `resource_monitor_started` | 任意の独立観測プロセスのPID、出力先、実行ソースのSHA-256 |

各行にはhostname、PID、PPID、UNIX時刻ns、単調時計nsがある。
単調時計は同じホスト内で比較する。異なるホスト間では単純に差を取らない。
このloaderは順序通りにバッチを返すため、親の反復内取得番号と子のバッチindexを
時刻・worker IDと合わせて照合できる。prefetchやepochの切り替わりも含めて読む。

`max_batches` は親での計時回数、各生成プロセスでの計時回数のそれぞれの上限。
最初のバッチから計るため、起動直後のコストを含む。persistent workerではepochを
またいでも予算を引き継ぐ。非persistent workerは新しいPIDごとに記録を始める。
`static_batch_cache_size > 0` ではキャッシュからの取得を計り、`phases` は空になる。
静的バッチの事前生成時間は、このイベントの対象外。

`batch_received.wall_ns` には入力待ち、キュー受け渡し、CPU処理等が含まれる。
`process_cpu_ns` はそのプロセス全体のCPU時間差分で、子プロセスの時間は別記録。
これらの差だけで純粋なIPC待ち時間を確定することはできない。
GPU同期は追加しない。非同期デバイス処理がある場合、その完了時間まで計る指標ではない。

スナップショットは親のバッチ取得境界で、指定間隔以上空いた場合に最大
`max_snapshots` 回取得する。指定間隔は最小間隔であり、定期タイマーではない。
長い`next()`呼び出しの最中や学習の停止中には採取しない。
定常区間まで追いたい場合は間隔と回数を増やす。
上限は128プロセス、各256スレッド。省略時はtruncatedフラグを記録する。
`collection_wall_ns` はファイル読み取り等の収集時間で、JSON化・出力時間は含まない。

`stat` のuser_ticks/system_ticksを同じPID/TID・start_ticksの間で差分にし、
`clock_ticks_per_second` と経過秒で割れば平均CPU使用コア数になる。
プロセスとそのスレッドのCPU時間は重複するため、両方を足し合わせない。
プロセス直下のstatus/schedstatは主スレッドの項目を含む。
スレッドの `wchan` は読める場合のカーネル待機先で、0や未取得の場合もある。
関数別の実行プロファイルや、別コンテナのcs_worker_appの状態は取得しない。

cgroup v2ではcpu.max、cpu.stat、cpu.weight、cpu.pressure、cpuset.cpus.effectiveを、
v1では対応するquota/period/stat/shares/cpusetを読む。マウント内の可視祖先も記録する。
`quota_cores: null` はその階層のquota無制限を表す。読み取れない場合はerrorを記録し、
quota_cores自体を省略する。親階層、affinity、競合も実際のCPU取得量を制限し得る。
隠れた祖先は `hidden_ancestors_checked: false` とし、分かったことに含めない。
子の所属cgroupは各process記録にあり、親と異なる場合は親の制約だけでは評価できない。
nr_throttled等は区間差分で評価し、throttled時間を学習時間の損失率へ直接換算しない。

`resource_monitor: true` は、DataLoader iterator生成後にstdlibだけの独立プロセスを起動する。
親と子孫のCPUカウンタ・page fault・RSS、メモリcgroupの使用量・上限・OOMカウンタ、
`/dev/shm` の空き、pressureを `resources-<parent-pid>.jsonl` へ保存する。
`resource_monitor_pss: true` を併用すると `smaps_rollup` のPSSも読む。
バッチ取得が停止中でも観測できる。間隔はsnapshot_interval_seconds（最大60秒）、
回数はmax_snapshots（最小1回）で制限し、親終了・PID再利用・loader解放でも終了する。
PSS読み取りの負荷を含むので、通常の主性能測定ではこの機能を無効にする。

## オーバーヘッドと検証

無効時は従来のDataLoaderとDatasetをそのまま使う。診断実装のimport、ラッパー、
タイマー、`/proc`走査、ファイル出力、観測プロセスは反復経路に入らない。
設定解析とfactory内の有効化判定のみが増える。
テストでは無効のloader生成と2 epochの反復をPythonの関数呼出しフックで観測し、
診断モジュール内の関数呼び出しが0回であることを確認する。

有効時はファイル読み取り・JSON出力・計時のコストがある。ほぼ無視できるという性能保証は
置かず、回数制限のある診断runとして扱う。通常の性能測定は無効にして行う。
計時予算終了後も有効なラッパーの分岐は残る。resource_monitor無効時は追加プロセスを作らない。
有効時はloader解放時のfinalizerで観測プロセスを回収する。

GNNディレクトリで次を実行する。グラフのダウンロードやCSXジョブ投入は不要。

```bash
OUTDATED_IGNORE=1 uv run --no-sync python -m unittest discover -s tests -p 'test_worker_diagnostics.py' -v
OUTDATED_IGNORE=1 uv run --no-sync python -m unittest discover -s tests -p 'test_loader_settings.py' -v
OUTDATED_IGNORE=1 uv run --no-sync python -m unittest discover -s tests -p 'test_trainer_loader.py' -v
OUTDATED_IGNORE=1 uv run --no-sync python -m unittest discover -s tests -p 'test_fixed_shape_gpu.py' -v
```

fork/spawnで実際の子プロセスを生成し、persistent workers、静的バッチ、
Cerebras PyTorchのfactoryシリアライズ、v1/v2の可視階層、出力失敗を検証する。
診断オン／オフのバッチ一致とnative学習loss一致も確認する。CUDAが利用可能なら
native学習の比較をGPUでも実施する。DRAM帯域・instructions per cycleのPMU計測は含めない。
