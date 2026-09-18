# 学習曲線を保存してから入力設定を探索する

```bash
bash benchmark_scripts/cerebras/run_learning_campaign.sh --dataset arxiv
```

一つの tmux ドライバが、実データの学習と検証精度の記録、入力設定の探索、
上位候補の反復測定、選択した設定の保存を順に実行する。
products は `--dataset products` で別の出力先に実行する。
一つのドライバ起動に対して SDK は複数の wsjob を作る。
単一の CSX 予約・wsjob 内で全設定を切り替える仕組みではない。

学習成立の判断は研究者が行う。精度の閾値、精度向上量、loss の減少率による
自動判定は行わず、結果に `review_status: pending_human_review` を記録する。
学習クライアントが正常終了し、予定した有限値の loss・検証記録が揃えば、
人の入力を待たずに探索へ進む。低い検証精度でも探索を中止しない。
失敗、非有限値、記録の欠損・重複、遠隔ジョブの終了が未確認の場合は停止する。

## 学習と探索の条件

最初に arxiv は 500 step、products は 1000 step の実データ学習を行い、
それぞれ 100／200 step ごとに valid split 全体を評価する。
train と valid を明示し、末尾の実ノードを捨てず、static batch の再生も行わない。
既定の DataLoader worker 数は 2。`--learning-steps` と `--eval-every` で変更できる。
少なくとも 2 回の検証を行い、両方とも 10 step のログ周期に合わせる。
複数 seed による最終精度の比較は、この一回の記録とは別に行う。

次に同じモデル、AdamW、精度設定、batch size 4096、fanouts [15,10,5] を保ち、
worker 数 2／4／8／12／16、上位候補の prefetch factor 1／2、persistent workers
false／true を調べる。AdamW の eps=1e-6 と betas=[0.9,0.999] は GPU 側にも明記する。
CSX の入力 Worker replica 数は 1 に固定する。
40 step のウォームアップ後に 400 step を測り、上位 2 候補を 800 step ずつ 3 回確認する。
確認では候補の実行順を反転し、全反復で正常かつ安定した候補だけを順位付けする。
探索と確認に検証、checkpoint 保存、詳細 worker 診断は混ぜない。

`--repeat-learning-with-selected` を付けると、選択後にも同じ長さの学習曲線を保存する。
この追加記録にも精度による自動合否はない。既定では選択後の再学習は実行せず、
後から実行できる `handoff/selected_learning.yaml` を保存する。

## 出力と GPU 比較への引き継ぎ

`baseline/learning_curves.json` に全 training loss と検証 step・loss・accuracy、
`baseline/train.log` に生ログ、`baseline/params.yaml` に実行設定を保存する。
`tuning/study.json` には候補、失敗、測定区間、分子の定義、反復値、順位が残る。
`learning_campaign.json` は全体の状態、ソース・環境の指紋、消費予算を記録する。

| 出力 | 用途 |
| --- | --- |
| `handoff/selected_csx.yaml` | 選択された CSX のスループット測定設定 |
| `handoff/selected_fixed_shape_gpu.yaml` | 同じ固定形状・sampler を使う GPU 実装への開始設定 |
| `handoff/pyg_reference.yaml` | 通常の PyG 実装への開始設定。固定形状経路とは別に報告する |
| `handoff/selected_csx_diagnostics.yaml` | 80 step の追加診断。worker の PID、入力生成、資源利用を取得する |
| `handoff/selected_learning.yaml` | 選択した入力設定で学習曲線を追加取得する設定 |
| `handoff/selection.json` | 順位と学習曲線への対応、研究者の確認が未実施であること |

GPU 向け設定の worker 数は CSX の選択値を初期値として含む。GPU での最適値や
学習成立を保証するものではなく、GPU 側で独立に探索する。
共通 [autotune](autotune.md) の `--backend fixed_shape --base-config
handoff/selected_fixed_shape_gpu.yaml` で、引き継いだモデル・optimizer・精度設定を保ったまま
GPU 側の入力設定を探索できる。通常の PyG は別に `--backend pyg --base-config
handoff/pyg_reference.yaml` を使う。
特徴量 cache は固定形状 GPU では `null` でバイパスし、PyG では `0.0` で
GPU cache を無効にする。PyG の `null` は GPU 自動 cache を意味するため使わない。
出力先のパスは実行ホスト用なので、GPU 実行時は別の空の model/output ディレクトリを指定する。

最終的な GPU 対 CSX の性能比には、winner 選択に使った反復を再利用しない。
両側の入力設定を固定してから、新しい出力先に同じ step40–840 の区間を各 3 回以上記録する。
共通 tuner の sensitivity モードで候補を一つに絞る方法と、prefetch／persistent workers を
選択値に保つ指定は [autotune](autotune.md) を参照する。
この反復は選択後に独立して取得し、失敗・欠測も保存する。

性能比較では、パディングを含む公称 seed 枠/秒と実 seed 数/秒を混ぜない。
CSX の実 seed 数が sampler の宣言したスケジュールからの算出である場合は、
デバイスで直接観測した数とは区別する。固定形状 GPU と通常 PyG も別の実装として記録する。
精度の判断、同等な学習条件の確認、分子・測定区間を揃えた最終比較は、これらの生データを使って行う。

詳細診断の procfs/cgroup は実行ユーザーが読める項目だけを取得する。
Grafana cookie、運営の管理権限、追加のサービス許可は必要ない。
読めない項目は欠測として扱い、診断ありの測定値を通常のスループット反復に混ぜない。

## 事前確認と再開

```bash
bash benchmark_scripts/cerebras/run_learning_campaign.sh \
  --dataset arxiv --dry-run --output /tmp/learning-preview-arxiv
```

設定と計画だけを保存し、ジョブを投入しない。
通常実行は `model_dirs/hpcasia_learning/<dataset>_<UTC>/` に出力する。
tmux の接続と終了コードの確認は [起動と監視](worker_launch.md) と共通である。

既定の合計予算はクライアント経過時間 86400 秒、各クライアント上限 9000 秒、
各 SDK job 上限 7200 秒。合計予算が尽きた場合は、同じ引数と `--output` を指定し、
`--budget-sec` を増やして再開する。完了した学習・探索は繰り返さない。
ソース、設定、環境が変わった場合は別の出力先を使う。
停止・失敗した遠隔クライアントを自動的に再投入したり、終了確認を代行したりしない。
遠隔ジョブの状態を確認してから新しい出力先で再実行する。
