# GNN入力設定の探索

[学習曲線の取得と探索](docs/learning_campaign.md) は、一つのドライバで実データ学習の
loss・検証精度を保存してから入力設定を探索する。学習成立の判断は研究者が行う。

[autotune](docs/autotune.md) は CSX、固定形状 GPU、通常 PyG の worker 数・prefetch・
persistent workers の探索、予算、再開、独立反復を扱う。
[測定指標](docs/throughput.md) の分子と計測区間を揃え、候補選択後に最終比較を別途行う。

`configs/autotune/` は arxiv/products の実行設定で、`measure_batch_accounting` を有効にする。
CSX は `GNN_INPUT_CONTRACT` と完了stepから対象頂点数を計算し、GPU は消費した対象頂点数を記録する。
出力には分子の定義、解決済み設定、ソースと環境、各独立runの値を保存する。

近傍サンプリングの訓練入力では、`shuffle: true` により入力を1周するたびに対象を
`sampler_seed + epoch`（初回はepoch 0）で並べ替えてからbatchを構成する。
初回の順序、固定形状、近傍選択、小さい末尾batchの保持は従来どおりである。
Model Zooの `GptHDF5MapDataProcessor` → `HDF5Dataset` → SDK `ShuffleSampler` が
周回ごとにseedを更新する方式を参照した。GNNではdataset要素が既にbatchなので、
外側でbatchの順序だけを並べ替えず、samplerから周回番号をworkerへ渡して対象を再配置する。
これによりpersistent workersと通常のworkersで同じseed・周回の対象順序が一致する。
`shuffle: false`、検証入力、`static_batch_cache_size > 0` の固定入力診断は反復を固定する。

周回はSDK Repeaterがtorch loaderを使い切った時点で進む。消費側が同じiteratorから
読み続ける場合は途中の停止を跨いで継続する。新しいloaderの作成やcheckpointからの
再開で入力位置を復元する機能はなく、`GNN_INPUT_CONTRACT.restartable` は引き続きfalseである。
CS-3の学習・評価切替を跨ぐ実消費順序は、実機の入力記録で確認する必要がある。

入力契約のversion 2は対象・ラベルのhashを初回周回のものと明記し、対象数の周期と
対象IDの順序を区別する。splitに無視ラベル（-100）が混在する場合、再shuffle後の
batchごとの教師対象数は初回と変わり得るため、周期表からの性能計算は拒否する。
固定形状GPUの計測では、消費した教師対象数のカウンタを集計し、範囲と合計の整合性を検証する。
旧version 1の固定順序の記録は引き続き読める。
