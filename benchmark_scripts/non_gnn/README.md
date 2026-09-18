# 非 GNN モデルの CS-3 / Pegasus 比較

CS-3 は Model Zoo 2.10.0 の `cszoo fit` 相当の CLI，Pegasus は既存例と同じ
NQSV の `AC2` / `gpu` / 1ノード / 2時間の PBS 設定を使う。
各コマンドは明示した1条件だけを実行・投入する。`--dry-run` は設定とコマンドを
表示し，ファイル作成，コンパイル，ジョブ投入を行わない。

## 比較条件

| `--profile` | 系列長 | 有効バッチ | GPU マイクロバッチ | GPU 勾配累積 |
| --- | ---: | ---: | ---: | ---: |
| `bert_large_msl128` | 128 | 256 | 16 | 16 |
| `bert_large_msl512` | 512 | 64 | 4 | 16 |
| `llama3p2_1b_msl1024` | 1024 | 32 | 1 | 32 |
| `llama3p2_1b_msl2048` | 2048 | 16 | 1 | 16 |

設定の入口は [profiles.yaml](profiles.yaml)。Model Zoo の既存 YAML を読み，実行用の
完全な `params.yaml` を生成する。両環境で BF16，seed 42，同じ有効バッチ，
AdamW，勾配ノルム上限1を使う。CS-3 は1台でマイクロバッチ自動選択。
全条件とも1更新あたり32768の予約トークン位置を持つが，有効トークン数や演算量は異なる。

既定はランダム初期化から200回の optimizer 更新を行うスループット実験。
BERT の学習率は一定の `1e-4`，Llama は `3e-4`。
検証とチェックポイント保存は無効。事前学習の収束や精度同等性の評価は含まない。
BERT は両系列長とも512位置の埋め込み表を維持する。Llama は元の1Bモデルの
層数，隠れ次元，GQA，RoPE 設定，埋め込み共有を保って系列長を短くする。
表のバッチは調整開始点であり，H100 のメモリ使用量や CS-3 のコンパイル成立は実機で確認する。

## 環境とデータ

既存の `./setup.sh --target-env csx` / `./setup.sh --target-env gpu` で作成する
リポジトリの `.venv` を使う。GPU ジョブは既存の `gpu_env.sh` で CUDA module と
toolkit を確認する。`setup.sh` がダウンロードする GNN データとは別に，次を準備する。

- BERT: `BertCSVDynamicMaskDataProcessor` 用 CSV と `meta.dat`。
  CSV は `tokens`, `segment_ids`, `is_random_next` を持つ。
  系列長128／512用に前処理したデータをそれぞれ指定する。NSP を有効にしたまま，
  512のサンプルを128として読み込むことはできない。
  語彙は同梱の uncased BERT 語彙を既定とし，`--vocab-file` でも指定できる。
  前処理例: `src/cerebras/modelzoo/data_preparation/nlp/bert/README.md`。
- Llama: Llama 3系の tokenizer で事前処理した HDF5。
  既定の `--llama-data-format sample` は `data` の形が `(N, 3, 系列長)`，
  順に `input_ids`, 損失マスク，シフト済み `labels`。
  **8Kのファイルを指定しても自動で1024／2048には短縮しない。** 全ファイルの形を投入前に検査する。
  `--llama-data-format corpus` なら1次元のトークン列を読み，loader が指定系列長で
  切り出して正解ラベルを作るため，同じコーパスを両系列長に使用できる。
  前処理例: `src/cerebras/modelzoo/data_preparation/README.md`。

同じ条件の両環境には，同じ前処理済みデータと語彙を配置する。CS-3 側のデータと
語彙は Worker コンテナからも参照できるパスを使う。必要な追加 mount は
`--mount-dir /absolute/path` で指定する。スクリプトはデータの転送やダウンロードを行わない。

GNN 向け setup で省かれる NLP 依存は [requirements.txt](requirements.txt) に記載した。
設定検証は，使用しない入力 processor も Model Zoo registry 経由で import するため，
`datasets` と `torchvision` も必要になる。PyTorch 2.4.0 のビルドに合わせて，
次の追加セットアップを行う（GPU の例。CPU版を使う CS-3 user node では末尾を `/cpu` にする）。

```bash
uv pip install --python .venv/bin/python -r benchmark_scripts/non_gnn/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

GPU 実装は Transformers 4.57.3 と PyTorch 2.4 の API を使用する。

## CS-3：各コマンドを Cerebras user node で実行

リポジトリのルートから実行する。`/data/...` はその環境の実パスへ置き換える。
最初は末尾に `--dry-run` を付けて確認する。

```bash
./benchmark_scripts/cerebras/run_non_gnn.sh --profile bert_large_msl128 --data-dir /data/bert/train_msl128
./benchmark_scripts/cerebras/run_non_gnn.sh --profile bert_large_msl512 --data-dir /data/bert/train_msl512
./benchmark_scripts/cerebras/run_non_gnn.sh --profile llama3p2_1b_msl1024 --data-dir /data/llama/train_msl1024
./benchmark_scripts/cerebras/run_non_gnn.sh --profile llama3p2_1b_msl2048 --data-dir /data/llama/train_msl2048
```

foreground のクライアントを保持する既存例と同じ方式。設定だけを保存・検証する場合は
`--prepare-only --output-dir /path/to/new/run` を付ける。

## Pegasus：各コマンドをログインノードで実行

```bash
./benchmark_scripts/pegasus/submit_non_gnn_nqsv.sh --profile bert_large_msl128 --data-dir /data/bert/train_msl128
./benchmark_scripts/pegasus/submit_non_gnn_nqsv.sh --profile bert_large_msl512 --data-dir /data/bert/train_msl512
./benchmark_scripts/pegasus/submit_non_gnn_nqsv.sh --profile llama3p2_1b_msl1024 --data-dir /data/llama/train_msl1024
./benchmark_scripts/pegasus/submit_non_gnn_nqsv.sh --profile llama3p2_1b_msl2048 --data-dir /data/llama/train_msl2048
```

GPU の既定経路は [gpu/](gpu/) 内のネイティブ PyTorch 学習ループと
Transformers の BERT／Llama，SDPA，fused AdamW。モデル重みのダウンロードは不要。
入力処理は Model Zoo の既存 processor と native DataLoader を再利用する。
GPU のモデル計算と学習ループは Model Zoo Trainer を経由しない。

- `--gpu-implementation modelzoo`: 同じ生成設定を Model Zoo Trainer で実行する補助比較。
- `--compile`: ネイティブ GPU モデルに `torch.compile` を適用する。
- `--gradient-checkpointing`: ネイティブ GPU の activation checkpointing を有効にする。
- `--gpu-micro-batch-size N`: GPU マイクロバッチを変更。有効バッチの約数が必要。
- `--effective-batch-size N`: 有効バッチを変更。対応する CS-3 実行にも同じ値を指定する。
- `--csx-micro-batch-size N`: CS-3 の内部マイクロバッチを指定。既定は `auto`。
- `--max-steps N`, `--num-workers N`, `--seed N`: 両環境に適用。
- `--warmup-steps N`: native GPU の計時から除く先頭更新数。既定20，`max-steps` 未満。

`--compile` や大きいマイクロバッチによる速度改善は実測で判断する。
再実行は新しい出力ディレクトリを用い，同じディレクトリへの上書き・自動再開は行わない。
PBS の時間上限を変更する場合は `run_non_gnn_nqsv.pbs` を調整する。

## 損失と測定範囲

BERT は masked positions を gather して語彙 head を適用し，Model Zoo と同じ
MLM の重み（128: 0.058，512: 0.019）と NSP を用いる。
Llama の HDF5 の `attention_mask` は損失マスクで，モデルへの attention mask には使わない。
ラベルは既にシフト済みなので，Transformers の内部ラベルシフトは使わない。
native GPU では累積対象全体の有効トークン数で損失を正規化する。
初期化の乱数列や異なる実装の演算順序は一致を保証しない。同じ seed は同じ重みを保証しない。

実行ごとに `model_dirs/non_gnn/` 以下へ次を保存する。

- `params.yaml`, `launch.json`: 完全な設定，元設定の SHA256，revision，dirty 状態，起動引数。
- `console.log`: 学習ログ。Pegasus の投入結果は `qsub.log`。
- native GPU の `gpu_environment.json`: ソフトウェア版，GPU，パラメータ数，実装設定。
- native GPU の `metrics.jsonl`: 更新ごとの損失と時間，ウォームアップ後の平均処理速度とメモリ使用量。

GPU 計時はデータ取得，転送，forward/backward，optimizer を含み，初期化とウォームアップを除く。
予約トークン位置数による `nominal_tokens_per_second` と，マスクによる有効数を区別する。
BERT の有効数は入力の非 padding 位置，Llama は損失対象位置であり，両モデル間では意味が異なる。
CS-3 側は progress log の同じ更新区間から `更新差 × 有効バッチ / 経過秒` を算出する。
比較時は両側の区間を揃え，native GPU の区間平均と CS-3 の単発 `Rate` を混在させない。
短い実行のウォームアップ不足，実行間のばらつき，GPU のメモリ適合性は実機で確認する。

## ローカル検証

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s benchmark_scripts/non_gnn/tests -v
```

テストは小さい CPU モデルと一時データで，損失の値と勾配，入力契約，設定，
`qsub` の引数境界と dry-run を確認する。実機のコンパイル・実行性能の検証は別途必要。
