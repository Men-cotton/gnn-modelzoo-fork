> 自動探索の実装・実行・再開手順は [docs/autotune.md](docs/autotune.md) を参照してください。
> 以下は ZIP 同梱の過去の検討記録です。現在は prefetch / persistent worker を接続済みで、
> 12設定には以前の実効値（prefetch 2、persistent false。worker 0 は null / false）を明示しています。

# CS-3 の warm-up と入力 worker 数の比較

対象コミット: [3dbe09aa22b8e10d7f6348190807d987b43bfbca](https://github.com/Men-cotton/gnn-modelzoo-fork/commit/3dbe09aa22b8e10d7f6348190807d987b43bfbca)

## 推奨する測定

１回の連続した train loop の中で、最初の 40 step を warm-up として除外し、続く 200 step を測る。合計は `max_steps: 240` とする。40 回のジョブ投入や 40 epoch を意味しない。評価や checkpoint 保存を挟まない。

公開された arxiv/products × cached/uncached の４ログについて、進捗ログの step と timestamp から再集計した。下表の長区間は、既存論文の集計と同じく arxiv が step 20〜500、products が step 40〜1000 である。

| データセット・条件 | 長区間の slots/s | step 40〜240 の slots/s | 長区間との差 |
| --- | ---: | ---: | ---: |
| arxiv・uncached | 1212.484 | 1213.358 | +0.0721% |
| arxiv・cached | 1241.021 | 1242.899 | +0.1514% |
| products・uncached | 592.847 | 592.763 | −0.0142% |
| products・cached | 590.843 | 590.194 | −0.1098% |

arxiv の最初に観測できる区間 step 20〜40、products の step 40〜80 も、長区間との差は 0.6% 未満だった。したがって、既存条件では長い warm-up が必要という兆候はない。ただしログはそれぞれ 20/40 step 間隔なので、必要最小 step 数を特定した結果ではない。同一実行の部分区間を比較しており、独立した再実行の精度や、新しい worker 数の定常性を保証する値でもない。

集計式は `4096 * (end_step - start_step) / (t(end_step) - t(start_step))` とする。step 40 の完了時刻を始点にするので、測るのは step 41〜240 の 200 step になる。これは論文に合わせた名目上の seed-node slot 数であり、パディングを除いた実ノード数ではない。新しい YAML でも batch size=4096 とサンプリング条件を固定する。

`Rate` の単発値や累積の `GlobalRate` は順位付けに使わない。GlobalRate が上昇し続けても、それだけでは warm-up が続いていると判断できない。コンパイル費用は集計対象にせず、そのための記録項目も設けていない。

## 内容と配置

ZIP を `src/cerebras/modelzoo/models/gnn/` に展開する。既存の設定を上書きする必要はない。

- `configs/autotune/arxiv_w00.yaml` など: データセット２種類 × worker 数 `0, 4, 8, 16, 32, 40` の12設定。
- `tools/measure_window.py`: Python 標準ライブラリだけで動く、完了済みログの集計スクリプト。
- `autotune_evidence.json`: 上表、40→80 の除外境界変更、測定区間の延長を検討した再計算結果と元ログへのリンク。

まず `w40` を基準として再測定する。その後 `w08`、`w16` などを１試行ずつ実行する。小さい worker 数ほど速いと予測しているわけではない。CPU 数の検査で許容され、入力 worker に割り当てられた CPU・メモリ内で動く候補だけを使う。候補を一括でキューに入れない。

## 実行例

リポジトリで用意済みの uv 環境を使う。モデルディレクトリで実行する。

```bash
cd /path/to/gnn-modelzoo-fork/src/cerebras/modelzoo/models/gnn
unzip /path/to/cs3_autotune_overrides.zip

trial_id="arxiv_w08_r1"
mkdir -p "autotune_runs/${trial_id}"
set -o pipefail
uv run --no-sync -- cszoo fit configs/autotune/arxiv_w08.yaml \
  --target_device CSX \
  --model_dir "autotune_runs/${trial_id}/model" \
  2>&1 | tee "autotune_runs/${trial_id}/train.log"
```

正常終了した試行のログを集計する。

```bash
uv run --no-sync tools/measure_window.py \
  "autotune_runs/${trial_id}/train.log" \
  --start-step 40 --end-step 240 --batch-size 4096 \
  > "autotune_runs/${trial_id}/measurement.json"
```

products の場合は `products_w08.yaml` と別の trial_id を使う。繰り返す際も trial_id を変え、１ファイルに複数の実行ログを連結しない。checkpoint の自動読み込みと `fit.ckpt_path` は明示的に無効にしてある。

`job_time_sec: 7200` は一試行のジョブに指定する実行上限であり、複数試行全体の時間制限ではない。タイムアウトした試行を「遅いが正常終了した測定」として採用しない。既存速度が維持される場合、200 step の測定区間は arxiv で約11分、products で約23分となる。

## 定常性と候補の確定

新しいログは `log_steps: 10` なので、集計スクリプトは step 40〜140 と 140〜240 の速度も出力する。両者の差を平均値で割った値が 2% 以下かを、粗探索用の目安として報告する。この値は定常性の証明や信頼区間ではない。

差が 2% を超える、継続的な増減がある、入力待ちが続く、といった場合は、その値を直ちに順位付けに使わない。warm-up 境界を 80 に延ばして 200 step 測るなら、`max_steps: 280` と `--start-step 80 --end-step 280` を組み合わせる。周期的な遅延がある場合は、遅い区間だけを削除するのではなく、測定区間を延長する。

有望な候補は `max_steps: 440` とし、同じ warm-up 40 step の後で400 step を測る。別々のジョブで最低３回再測定し、実行順を交互にして中央値とばらつきを確認する。

```bash
uv run --no-sync tools/measure_window.py \
  /path/to/finalist.log --start-step 40 --end-step 440
```

必要な step 境界がログにない場合、スクリプトは補間せずエラーにする。公開 products ログには step 140 がないため、40〜240 の全体集計はできても半分ずつの検査は行わない。その場合は `half_window_check: null` と理由を出力する。

## 対象コミットに合わせた注意点

- この YAML は実際に DataLoader に渡る `num_workers` を探索する。`prefetch_factor`、`persistent_workers`、`pin_memory` は設定クラスにはあるが、対象コミットの neighbor DataLoader には設定値が渡っていないため、ここでは探索しない。worker 数を変えることと、Cerebras の `num_workers_per_csx` を変えることも区別する。
- `cache_fraction: null` は GraphCache を生成しない。`0.0` は「０ノードをキャッシュする GraphCache」を生成するため、同じコード経路ではない。この提案では通常の uncached 経路を明示的に選ぶ。
- 公開ログの cached 条件は `cache_fraction: 1.0` による特徴量キャッシュであり、少数の完成バッチを繰り返す `static_batch_cache_size` とは異なる。後者はこの提案では常に０とする。cached 条件を別に比較したい場合は `cache_fraction: 1.0` にしたコピーを作り、同じ worker 数・測定窓・別 trial_id で実行する。CSX 経路の GraphCache の配置先は CPU である。
- `warmup_steps` は Trainer のネイティブな設定項目ではないので YAML に追加していない。240 step の学習自体には最初の40 step も含まれ、計測スクリプトだけがその部分を除外する。optimizer の learning-rate warm-up とも無関係である。
- アーカイブの `trainer_params.yaml` は古いキーやサニタイズ済みパスを含む実行記録である。新しい実行用 YAML は、リポジトリ内の現行設定を `extends` で継承している。

## 検証の範囲

対象コミットの設定ローダーで12設定の継承結果を検証し、集計スクリプトを４つの公開ログと不正入力で確認した。CS-3 への投入、Cerebras SDK を使った完全な設定検証、実機でのスループット測定は行っていない。ALCF で使う既存の uv 環境、ソース・データのマウント、および import path が利用できることが前提となる。

## 参照

- [論文用の集計スクリプト](https://github.com/Men-cotton/gnn-modelzoo-fork/blob/3dbe09aa22b8e10d7f6348190807d987b43bfbca/artifacts/IA3-2026/scripts/training_loop_throughput.py)
- [公開アーティファクトの説明](https://github.com/Men-cotton/gnn-modelzoo-fork/blob/3dbe09aa22b8e10d7f6348190807d987b43bfbca/artifacts/IA3-2026/README.md)
- [DataLoader の実装](https://github.com/Men-cotton/gnn-modelzoo-fork/blob/3dbe09aa22b8e10d7f6348190807d987b43bfbca/src/cerebras/modelzoo/models/gnn/data_processing/samplers/neighbor_tree.py)
- [Trainer の loop の実装](https://github.com/Men-cotton/gnn-modelzoo-fork/blob/3dbe09aa22b8e10d7f6348190807d987b43bfbca/src/cerebras/modelzoo/trainer/callbacks/loop.py)
- [Cerebras による Rate / GlobalRate の説明](https://training-docs.cerebras.ai/rel-2.5.0/fundamentals/measure-throughput-of-your-model)
- [ALCF のジョブ運用案内](https://docs.alcf.anl.gov/ai-testbed/cerebras/running-a-model-or-program/)
