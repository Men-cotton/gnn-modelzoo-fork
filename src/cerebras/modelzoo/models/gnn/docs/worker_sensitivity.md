# HPC Asia R04: num_workers と学習 throughput

## 一つのコマンドで全候補を実行する

リポジトリ直下で、準備済みの Python 3.11 / Cerebras 2.10 環境とデータセットを使う。
現在のGNN入力経路は WSC Worker replica を1に限定する。R02で採用する割当と照合し、
`--wsc-workers` を明示する。DataLoader の `num_workers` は各 WSC Worker 内の
CPU子プロセス数であり、replica数とは別である。

```bash
bash benchmark_scripts/cerebras/run_worker_sensitivity.sh \
  --dataset arxiv --wsc-workers 1 --cache none \
  --workers 40 4 8 10 12 16 20 --repeats 3 \
  --warmup-steps 40 --measure-steps 400 \
  --budget-sec 189010 \
  --output model_dirs/hpcasia_r04/arxiv_none
```

事前確認では別の `--output` に `--dry-run` を付ける。学習ジョブは投入されない。
`plan.json` と各候補の `preview_w*.yaml` を確認する。dry-run の commands は各候補の
起動形式を示すものであり、反復を含む投入総数は `planned_trials` に示す。

既定の候補は `40, 4, 8, 10, 12, 16, 20`。7条件×3独立runで21ジョブを逐次実行する。
初回選別や上位候補の追加反復はなく、全runが同じ測定区間を使う。
奇数ラウンドは指定順、偶数ラウンドは逆順に回す。各runは新しいモデルディレクトリと
新しい学習クライアントを使う。内部の起動は同じPythonによる `python -u -m cerebras.modelzoo.cli.main fit params.yaml
--target_device CSX --model_dir ...` で、クライアントの終了を待って次へ進む。
一つのクライアントがコンパイル用と学習用の複数wsjobを作る場合があるため、
独立run数はwsjob IDの個数から数えない。

`189010` 秒は21run×client上限9000秒＋終了用10秒の予算であり、所要時間の予測ではない。
各ジョブの上限は7200秒、client上限9000秒にはキュー待ち・コンパイルも含む。
まずarxivで確認し、productsでは `--dataset products` と別出力先を使う。
必要なら主要な少数候補だけを `--workers 40 8 20` のように選ぶ。

通常起動は共通launcherが専用tmux内で実行するため、端末・SSH切断後も継続する。
`--foreground` で前景実行を選べる。接続方法、ログ、終了コード、csctlのlabelは
[起動と監視](worker_launch.md)を参照する。
予算が尽きた場合は同じコマンド・同じ出力先で `--budget-sec` を増やして再開できる。
完了runを再投入しない。既定ではCSX clientが失敗・中断した場合はstudyを停止する。
保存したログとjob IDでリモートジョブの終了を確認したうえで
`--acknowledge-stopped-jobs` を付けて再開する。失敗runは記録に残り、自動再試行されない。

無人運用で失敗・タイムアウトをスキップして後続runへ進むには `--continue-on-failure` を付ける。
[一括実験](worker_campaign.md) は内部でこの指定を使う。失敗runは再開時にも再試行せず、
同じ条件の予定済みの別反復は行う。失敗は集計に残り、測定欠落があればドライバの終了コードは2となる。
tmux起動時のシェルは起動成功で0を返すため、実験の終了コードは `launcher.json` で確認する。
Ctrl-C／SIGTERMの中断と予算切れは、この指定でも停止する。
遠隔ジョブの終了を確認したことにはしない。残存ジョブとの競合が後続runへ影響する可能性がある。

起動側のCPU affinityに全候補が収まることを、投入前に検査する。40が収まらない場合、
実行は最初のジョブより前に停止する。dry-runは40を含む全設定を生成し、
`blocked_workers` に不足を記録する。リモートWSC WorkerのCPU・メモリ割当も別途確認する。

## 固定する条件と persistent_workers

R04モードは `persistent_workers=True`、`prefetch_factor=2` を既定にする。
共有の `configs/components/architectures/input_pipelines/neighbor.yaml` もTrueである。
ゼロworkerを明示した場合はPyTorchの仕様に合わせFalse／prefetch=Noneへ正規化する。
autotuneモードの既定値は [autotune](autotune.md) を参照する。

根拠となるコードは `data_processing/samplers/neighbor_tree.py` の
`_order_targets`、`_deterministic_choice`、`_create_dataloader`。
対象頂点順と近傍選択はseedと頂点・hopから決まり、workerの可変乱数状態を使わない。
Datasetの一巡後もworkerとそのDatasetを保持し、再反復時のプロセス生成を避ける設定にする。
ローカルSDKの `streamer/data_pipe.py` は入力を使い切ると再び `iter(source)` を呼び、
PyTorchの `DataLoader.__iter__` はTrueの場合に既存iteratorをresetする。
この動作は [PyTorchの説明](https://docs.pytorch.org/docs/2.14/data.html#torch.utils.data.DataLoader) とも一致する。
実機での短縮量は対象設定の測定後に判断する。

batch size 4096、fanouts `[15,10,5]`、model seedとsampler seed 42を固定する。
cacheは `--cache none`（CSXでGraphCacheを作らない）または `--cache full`（CPU上のGraphCache）を
study全体で指定する。R02の主比較とそろえ、途中で変更する場合は新しいstudyにする。
静的バッチの再利用・fake data・validation・checkpoint保存とautoloadは無効。
各runは40stepの後、step40と440の時刻差で400stepを測る。
現在のCSX主指標は、実際の入力バッチ順から求めた対象頂点数を `(t440 - t40)` で割った
`seed_nodes_per_second` である。末尾のパディングを除き、教師ラベルが無効な実頂点は含む。
教師ラベルも有効な頂点数/秒と、公称枠数/秒を併記する。
入力生成時の `GNN_INPUT_CONTRACT` が欠ける測定は新しいstudyから除外する。
導出条件は [autotune](autotune.md) を参照する。
setupとcompilationはこの区間の外に置く。

worker数を変えるとprefetchの総枠数（2×num_workers）も変わる。
観測差はこの入力経路設定全体への感度として説明し、CPU計算能力だけへ帰属させない。
WSC replica数は要求値を各params.yamlへ保存する。実際の配置・割当はSDK/jobの記録と照合する。

## 実行ログから集計する

実行の `train.log`、SDKの `model/`、`study.json` / `result.json` / `params.yaml` を使う。
入力生成時にバッチ別の対象頂点数を記録し、学習中は既存のstepログを使う。
`GNN_INPUT_CONTRACT` とstepログが揃ったrunだけを集計する。

各runの更新時に次の派生ファイルを再生成する。

- `sensitivity_runs.csv`: 全予定runの値、状態、trial ID、job IDs、失敗理由。
- `sensitivity_summary.csv`: worker数ごとの有効測定数、未実施数、不安定run数、平均、標本標準偏差（ddof=1）。
- `sensitivity.json`: 上の内容と測定設定・study状態。

反復内の多数のstepを独立標本として扱わず、各runのthroughputを等しい重みで集計する。
半区間差が2%を超えた有限値のrunも平均へ含め、不安定フラグを残す。
失敗・不完全・非有限値の測定と未実施runは数値から除外し、その件数と理由を表示する。
有効runが1件の場合、標準偏差は空欄とする。最速値の選別や勝者設定の出力は行わない。
平均だけで安定性を判断せず、各runと不安定判定を併読する。

`performance.json` の集約値だけでは任意の測定区間を復元できないため、
`train.log`、入力カウント、解決済み設定を保持する。
CPUのsampling/gathering/packing個別計時（R05）はこの集計の対象外である。

## Workerの実効設定とCPU活動を調べる

[Worker診断](worker_diagnostics.md) の `worker_diagnostics` を対象の入力設定で有効にすると、
実際の子PID、反復開始時のnum_workers、CPU quota、スレッド別CPUカウンタ、
sampling/gatherと親の取得時間を記録できる。既定は無効で、通常のR04測定には
診断処理を追加しない。診断runは出力先を分けて実行する。
