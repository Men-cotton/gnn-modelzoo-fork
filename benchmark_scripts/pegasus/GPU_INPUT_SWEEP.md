# Pegasus: ogbn-arxiv / ogbn-products入力設定の探索

`run_gpu_input_sweep.sh` は既存の `tools/autotune.py` を逐次実行する。
`--backend both`（既定）は固定形状GPU経路、その後に通常のPyG経路を実行する。
`--backend fixed_shape` / `--backend pyg` で個別にも実行できる。
固定形状側は2026-09-18開始のCS-3学習キャンペーンとR04のモデル・表現を継承する。
PyG側はデータセット別の独立した既存設定とNeighborLoader/GraphSAGE実装を使う。
学習済み重みは読み込まず、各試行でモデルを初期化する。

CS-3キャンペーンの `handoff/selected_fixed_shape_gpu.yaml` をPegasusへ配置し、
`--base-config` に指定する。2026-09-18のarxiv条件はseed 42、GraphSAGE 3層、
hidden_dim 1024、fanouts [15,10,5]、batch_size 4096、dropout 0.5、
AdamW lr 0.003 / weight_decay 0.0005 / eps 1e-6、FP16である。
固定形状側はモデルとseedを指定YAMLから引き継ぐ。データセットは `--dataset arxiv|products`（既定arxiv）で選ぶ。
PyG側の既定はデータセットに対応する
`configs/autotune/arxiv_w40.yaml` / `products_w40.yaml`（継承を解決して使用）で、
`--pyg-base-config PATH` で独立に変更できる。`--backend pyg` では
`--base-config` は不要。データセットは事前配置する。

```bash
# リポジトリルート。設定ファイルのパスは実際の配置先へ置き換える。
./benchmark_scripts/pegasus/run_gpu_input_sweep.sh \
  --base-config model_dirs/arxiv/handoff/selected_fixed_shape_gpu.yaml \
  --output /tmp/arxiv_gpu_preview --compile --dry-run

# qsubコマンドの確認（投入しない）
./benchmark_scripts/pegasus/submit_gpu_input_sweep_nqsv.sh \
  --base-config model_dirs/arxiv/handoff/selected_fixed_shape_gpu.yaml \
  --output model_dirs/hpcasia_gpu_input/arxiv_s42 --compile --dry-run
```

投入時はsubmitコマンドの `--dry-run` を外す。既存のAC2/gpu、1ノードの
NQSV様式を使い、PBS payloadは前景で動く。全GPU試行は逐次実行する。
既にGPU割当内ならrunスクリプトを直接使える。`--compile` は全段階で
固定する条件で、省略すると両経路ともeager実行になる。
PyGは既存実装に合わせてNO_COMPILEを設定/解除する。外側のNO_COMPILEより
このフラグを優先する。compiled/eagerは別の出力先で測る。

| 段階（`--phase`） | 条件 | CS-3側との対応 |
|---|---|---|
| `tune` | worker探索、上位2 workerでprefetch 1/2/4 × persistence off/on、最終上位2候補を3回確認 | learningのworkers/loader/confirm。prefetch 4はGPU側の追加 |
| `workers` | 全workerをp2/persistent onで各3回、反復ごとに順序を反転 | R04のworker感度比較 |
| `prefetch1` | w4、prefetch 1、persistent on、3回 | R04 control_prefetch1 |
| `persistent-off` | w4、prefetch 2、persistent off、3回 | R04 control_persistent_off |
| `feature-cache` | w4、prefetch 2、persistent on、cache_fraction 1.0、3回 | R04 control_feature_cache |

既定の `all` は表の順に実行する。worker候補は
`2 4 8 12 16 24 32 40 48 64`。既存autotunerの規則により40を先に測る。
全worker数の基準はprefetch 2、persistent on、特徴量キャッシュ無効。
固定形状GPUのキャッシュはCS-3と同じCPU GraphCacheである。
PyGはキャッシュ無効を0.0、有効を1.0に設定する（nullはGPU自動キャッシュを
意味するため使用しない）。PyGのGPU特徴量キャッシュは別の機構として報告する。
静的バッチ再生は対象に含めない。

ウォームアップ40ステップを除外し、全段階（探索・感度・入力設定比較・最終候補確認）で
以下の区間を計測する。arxivの初期探索も従来の400から800計測ステップへ延長する。

| データセット | 総ステップ数 | 計測区間 | 計測ステップ数 |
|---|---:|---|---:|
| arxiv | 840 | step 40→840 | 800 |
| products | 1,640 | step 40→1,640 | 1,600 |

productsの固定形状側にはproducts用の解決済みYAMLを指定する。
データセットとYAMLの不一致は既存autotunerが拒否する。

```bash
./benchmark_scripts/pegasus/submit_gpu_input_sweep_nqsv.sh \
  --dataset products --backend both \
  --base-config model_dirs/products/handoff/selected_fixed_shape_gpu.yaml \
  --output model_dirs/hpcasia_gpu_input/products_s42 --compile --dry-run
```

全ての試行で検証とcheckpointを無効化し、
同一経路の指定seedで独立に初期化する。同期したGPUの計測端点間で、消費した実対象頂点数を
割った速度を用いる。半区間差2%の安定性判定、不正値やAMP更新スキップの拒否は
既存autotunerに従う。`unstable` も結果に残る。詳細な入力計時は性能測定へ追加しない。

既定の全探索には64以上のCPU割当が必要である。実行前にCPU affinityを確認し、
不足していれば投入されたpayloadは測定前に終了する。割当に合わせる場合は
run/submitの両入口で `--workers '2 4 8 12 16 24 32 40'` のように明示する。
GPUメモリとホストメモリへの適合は実行結果で確認する。CS-3のOOMを理由に
GPUの40 workersを除外しない。dry-runは現在のCPU割当を反映するため、
小さいCPU割当ではtuneの高worker候補が省かれる旨を表示する。

各段階の出力は `<output>/<backend>/<phase>/`。`study.json`、解決済みYAML、試行ログ、
`result.json`、GPU metricsを既存形式で保存する。経路ごとの `tune/best.yaml` が選択結果、
`workers/sensitivity_summary.csv` 等が反復集計である。独立した本番測定では、
選択設定を固定して別の出力先を用いる。設定探索結果と最終性能評価を区別する。通常のPyGは動的な入力表現・独自の
GraphSAGEパラメータ化を持つ実用経路の比較、固定形状GPUはCS-3と表現・モデル条件を
揃えた比較として、別々の結果表とランキングを使う。

各段階の予算は既定10800秒、試行timeoutは1800秒。run入口では
`--budget-sec` と `--trial-timeout-sec` で変更できる。各予算は経路・段階ごとで、both/allの最大予算合計は30時間になる。
PBS上限は24時間で、長い測定は `--backend` / `--phase` で分けて投入する。

コンパイル等により未完了になる場合は同一条件・同一出力先で再開する。
予算不足や中断時には後続段階へ進まない。感度測定のOOM/timeoutは記録して次の
試行へ進み、欠測を含めて完了した段階の後も独立した比較を実行するが、全体の
終了コードは2にする。スケジューラにジョブ全体を停止された場合の再開は、
既存autotunerの停止確認手順に従う。ソース・設定を変えた場合は新しい出力先を使う。
