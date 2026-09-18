# HPC Asia R04: num_workers と学習 throughput

## 一つのコマンドで全候補を実行する

リポジトリ直下で、準備済みの Python 3.11 / Cerebras 2.10 環境とデータセットを使う。
次は WSC Worker replica を1に固定した実験の例である。R02で採用する割当と照合し、
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
新しい学習クライアントを使う。内部の起動は `uv run --no-sync -- cszoo fit params.yaml
--target_device CSX --model_dir ...` で、クライアントの終了を待って次へ進む。
一つのクライアントがコンパイル用と学習用の複数wsjobを作る場合があるため、
独立run数はwsjob IDの個数から数えない。

`189010` 秒は21run×client上限9000秒＋終了用10秒の予算であり、所要時間の予測ではない。
各ジョブの上限は7200秒、client上限9000秒にはキュー待ち・コンパイルも含む。
まずarxivで確認し、productsでは `--dataset products` と別出力先を使う。
必要なら主要な少数候補だけを `--workers 40 8 20` のように選ぶ。

シェルは前景で動くので、長時間実行には既存の持続セッションを使う。
予算が尽きた場合は同じコマンド・同じ出力先で `--budget-sec` を増やして再開できる。
完了runを再投入しない。既定ではCSX clientが失敗・中断した場合はstudyを停止する。
保存したログとjob IDでリモートジョブの終了を確認したうえで
`--acknowledge-stopped-jobs` を付けて再開する。失敗runは記録に残り、自動再試行されない。

無人運用で失敗・タイムアウトをスキップして後続runへ進むには `--continue-on-failure` を付ける。
[一括実験](worker_campaign.md) は内部でこの指定を使う。失敗runは再開時にも再試行せず、
同じ条件の予定済みの別反復は行う。失敗は集計に残り、測定欠落があれば終了コードは2となる。
Ctrl-C／SIGTERMの中断と予算切れは、この指定でも停止する。
遠隔ジョブの終了を確認したことにはしない。残存ジョブとの競合が後続runへ影響する可能性がある。

起動側のCPU affinityに全候補が収まることを、投入前に検査する。40が収まらない場合、
実行は最初のジョブより前に停止する。dry-runは40を含む全設定を生成し、
`blocked_workers` に不足を記録する。リモートWSC WorkerのCPU・メモリ割当も別途確認する。

## 固定する条件と persistent_workers

R04モードは `persistent_workers=True`、`prefetch_factor=2` を既定にする。
共有の `configs/components/architectures/input_pipelines/neighbor.yaml` もTrueである。
ゼロworkerを明示した場合はPyTorchの仕様に合わせFalse／prefetch=Noneへ正規化する。
既存のautotuneモードとインポート済みの過去設定は従来の既定値を保持する。

根拠となるコードは `data_processing/samplers/neighbor_tree.py` の
`_order_targets`、`_deterministic_choice`、`_create_dataloader`。
対象頂点順と近傍選択はseedと頂点・hopから決まり、workerの可変乱数状態を使わない。
Datasetの一巡後もworkerとそのDatasetを保持し、再反復時のプロセス生成を避ける設定にする。
ローカルSDKの `streamer/data_pipe.py` は入力を使い切ると再び `iter(source)` を呼び、
PyTorchの `DataLoader.__iter__` はTrueの場合に既存iteratorをresetする。
この動作は [PyTorchの説明](https://docs.pytorch.org/docs/2.14/data.html#torch.utils.data.DataLoader) とも一致する。
実機での短縮量は測定後に判断する。過去のYAMLにTrueと書かれていても、当時の実装が
引数を渡していたかは別に確認するため、過去runとの同条件性はYAMLだけで認定しない。

batch size 4096、fanouts `[15,10,5]`、model seedとsampler seed 42を固定する。
cacheは `--cache none`（CSXでGraphCacheを作らない）または `--cache full`（CPU上のGraphCache）を
study全体で指定する。R02の主比較とそろえ、途中で変更する場合は新しいstudyにする。
静的バッチの再利用・fake data・validation・checkpoint保存とautoloadは無効。
各runは40stepの後、step40と440の時刻差で400stepを測る。
CSXの指標は `4096 * 400 / (t440 - t40)` の **nominal seed-node slots/s** で、
末尾のpaddingを含む。setupとcompilationはこの区間の外に置く。

worker数を変えるとprefetchの総枠数（2×num_workers）も変わる。
観測差はこの入力経路設定全体への感度として説明し、CPU計算能力だけへ帰属させない。
WSC replica数は要求値を各params.yamlへ保存する。実際の配置・割当はSDK/jobの記録と照合する。

## 既存ログから集計する

通常の `train.log`、SDKの `model/`、既存の `study.json` / `result.json` / `params.yaml` を使う。
R04のための学習中の追加計時やログ出力は加えていない。

各runの更新時に次の派生ファイルを再生成する。

- `sensitivity_runs.csv`: 全予定runの値、状態、trial ID、job IDs、失敗理由。
- `sensitivity_summary.csv`: worker数ごとの有効測定数、未実施数、不安定run数、平均、標本標準偏差（ddof=1）。
- `sensitivity.json`: 上の内容と測定設定・study状態。

反復内の多数のstepを独立標本として扱わず、各runのthroughputを等しい重みで集計する。
半区間差が2%を超えた有限値のrunも平均へ含め、不安定フラグを残す。
失敗・不完全・非有限値の測定と未実施runは数値から除外し、その件数と理由を表示する。
有効runが1件の場合、標準偏差は空欄とする。最速値の選別や勝者設定の出力は行わない。
平均だけで安定性を判断せず、各runと不安定判定を併読する。

既存 `artifacts` の調査では、2026-06-22収集の
`arxiv_graphsage_wse_not2.log` にstep40/240/440の時刻と正常終了があり、
追加記録なしで同じ定義のthroughputを再計算できる。
`trainer_params.yaml` にはseed/cache/worker設定も残る。
`performance.json` の集約値だけでは任意の測定区間を復元できないので、run.logを併せて保持する。
元ログを集計できることと、旧runを新studyの独立反復に含められることは別の確認事項である。
WSCの実際の割当、旧コードの挙動、設定、測定区間が一致する場合に限って旧runを採用する。
CPUのsampling/gathering/packing個別計時（R05）はこの集計の対象外である。

## Workerの実効設定とCPU活動を調べる

[Worker診断](worker_diagnostics.md) の `worker_diagnostics` を対象の入力設定で有効にすると、
実際の子PID、反復開始時のnum_workers、CPU quota、スレッド別CPUカウンタ、
sampling/gatherと親の取得時間を記録できる。既定は無効で、通常のR04測定には
診断処理を追加しない。診断runは出力先を分けて実行する。

## 二重DataLoaderラッパー修正後の再測定

2026-09-16の診断では、GNNが返すCerebrasラッパーをTrainerがさらに包むため、
遠隔の実効num_workersが0になっていた。修正後はTrainerがラッパーを管理する。
まず `bash benchmark_scripts/cerebras/run_worker_diagnostics.sh` を再実行し、
遠隔の実効2／40と子PIDを確認する。診断は毎回新しい出力先へ保存される。

通常の感度測定も同じスクリプト・測定条件を使えるが、`--output` は新しくする。
既存studyはコードrevisionを含むfingerprintが変わるため再開できない。過去の結果を
消さず、修正前後を別studyとして保存する。旧30 runと同じ指定値で測る例:

```bash
bash benchmark_scripts/cerebras/run_worker_sensitivity.sh \
  --dataset arxiv --wsc-workers 1 --cache none \
  --workers 40 2 4 6 8 10 12 16 20 24 --repeats 3 \
  --warmup-steps 40 --measure-steps 400 --budget-sec 270010 \
  --output model_dirs/hpcasia_r04/arxiv_none_loader_fix
```

通常runでは診断は無効。実効40プロセスとprefetchが有効になるので、修正前と比べて
Workerのメモリ使用や処理速度が変わり得る。短い診断の結果を確認して本測定へ進む。
