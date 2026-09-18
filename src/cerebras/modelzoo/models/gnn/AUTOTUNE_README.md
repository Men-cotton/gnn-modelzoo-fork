# GNN入力設定の探索

[学習曲線の取得と探索](docs/learning_campaign.md) は、一つのドライバで実データ学習の
loss・検証精度を保存してから入力設定を探索する。学習成立の判断は研究者が行う。

[autotune](docs/autotune.md) は CSX、固定形状 GPU、通常 PyG の worker 数・prefetch・
persistent workers の探索、予算、再開、独立反復を扱う。
[測定指標](docs/throughput.md) の分子と計測区間を揃え、候補選択後に最終比較を別途行う。

`configs/autotune/` は arxiv/products の実行設定で、`measure_batch_accounting` を有効にする。
CSX は `GNN_INPUT_CONTRACT` と完了stepから対象頂点数を計算し、GPU は消費した対象頂点数を記録する。
出力には分子の定義、解決済み設定、ソースと環境、各独立runの値を保存する。
