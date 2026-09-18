# 非 GNN モデルの CS-3 / Pegasus 比較

CS-3 は Model Zoo 2.10.0 の `cszoo fit` 相当の CLI，Pegasus は既存例と同じ
NQSV の `AC2` / `gpu` / 1ノード / 2時間の PBS 設定を使う。
引数なしで，公開 WikiText の取得・前処理から下表の4条件×3反復の実行・集計まで行う。
前処理済みデータを検証できた場合は，取得・整形をスキップして実行・投入へ進む。
`--dry-run` は予定を表示し，取得，ファイル作成，ジョブ投入を行わない。

CSXはGNNと同じ[共通launcher](../README.md)で既定でtmux起動する。
データ準備から全runの完了・集計まで，端末・SSH切断後も継続する。
`--foreground` で前景実行を選べる。tmuxは起動ホストが動作している間の継続を担う。

## 一括実行

環境をセットアップした後，リポジトリのルートで実行する。

```bash
# Cerebras user node: 入力準備 → 全設定の検証 → 4条件×3反復を逐次測定 → 集計
./benchmark_scripts/cerebras/run_non_gnn.sh

# Pegasus login node: 前処理またはキャッシュ再利用 → 4条件×3反復をqsub
./benchmark_scripts/pegasus/submit_non_gnn_nqsv.sh

# native / Model Zooを両方比較する場合は計24ジョブ
./benchmark_scripts/pegasus/submit_non_gnn_nqsv.sh --gpu-implementation both
```

既定の入力は [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
の `wikitext-103-raw-v1` / train。空行と見出しを除いた先頭10000段落を使う。
モデル重みは取得しない。Llama の tokenizer は
[meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
を使うため，初回は Hugging Face でのアクセス承認とログイン（または `HF_TOKEN`）が必要。
取得済み tokenizer は `--llama-tokenizer /path/to/tokenizer` または
`NON_GNN_LLAMA_TOKENIZER` で指定できる。BERT の語彙はリポジトリ同梱のものを使う。

データは既定で `model_dirs/non_gnn/data/<条件のハッシュ>/` に保存する。
両環境とも最初に `manifest.json` の前処理条件，必要ファイル，サイズ，SHA256 を確認する。
一致すれば，公開コーパスの問い合わせ・取得，tokenizer のロード，CSV/HDF5 の再作成を
すべて省く。学習のバッチやステップ数，GPU 実装の変更ではデータを再作成しない。
前処理条件の変更や破損があれば再作成し，破損した旧ディレクトリは `.invalid-*` として残す。
生成途中のディレクトリはキャッシュとして採用せず，同じ条件の並行前処理はロックで直列化する。

既定の revision `main` は初回取得時にコミットSHAを解決して manifest に記録する。
キャッシュ再利用時にはリモート更新を確認しない。別の版を使う場合は
`--dataset-revision` / `--tokenizer-revision` に明示した版を指定する。
両環境で同じ入力を保証するには，この版を揃えるか，キャッシュをディレクトリごと転送する。

```bash
# 投入予定だけを表示
./benchmark_scripts/cerebras/run_non_gnn.sh --dry-run

# データと設定を用意するだけ
./benchmark_scripts/pegasus/submit_non_gnn_nqsv.sh --prepare-only

# BERTだけ選択。--onlyは繰り返せる
./benchmark_scripts/pegasus/submit_non_gnn_nqsv.sh \
  --only bert_large_msl128 --only bert_large_msl512

# 任意のUTF-8テキストに切り替え（空行で文書を区切る）
./benchmark_scripts/cerebras/run_non_gnn.sh --raw-text /data/corpus.txt
```

`--data-root` はキャッシュの置き場所，`--max-documents` は使用段落数を変更する。
`--output-dir` はその実行の新しい出力先を指定する。再実行ではデータを再利用し，
学習ジョブは新しく投入する。既存の出力先への重複投入は拒否する。
生成設定がsourceの内容hashを変えないよう，出力先は `src/` と `benchmark_scripts/` の外に置く。
`--only` で選ぶモデル群が変わると，必要な前処理条件も変わるため別キャッシュになる。
`--repeats N` で反復数を変える。各反復は同じseedから新しいモデル・optimizerを作り，
性能の実行間変動を測る。奇数反復はprofileの指定順，偶数反復は逆順で実行する。
初回の動作確認を各条件1回にする場合は `--repeats 1` を指定する。

BERT は段落を文書として句読点で文分割し，WordPiece と NSP の文対をCSVにする。
MLM は既存 processor が学習時に生成する。Llama は文書ごとの BOS/text/EOS を
1次元HDF5に保存し，loader が文書境界をまたいで1024／2048に切り出す。
これはスループット測定用の入力であり，元論文の事前学習データ処理の完全再現ではない。

全設定の生成・検証に成功してから投入を始める。`campaign.json` に全条件と反復を保存する。
CSXは各クライアントの終了を待ち，結果を集計して次のrunへ進む。
失敗・タイムアウト・測定欠落も保存して後続を実行し，campaignの終了コードを非ゼロにする。
明示的な中断では後続を起動しない。既定のクライアント上限は `--trial-timeout-sec 9000`，
CSX側のjob上限は `--job-time-sec 7200`。クライアント時間はキュー待ち・コンパイルも含む。
ローカルクライアント終了後の遠隔job状態は保存したjob IDで確認する。

GPUは既存のPBSに `qsub` し，応答を各runの `qsub.log` に保存する。
各PBS jobは同じprepared-run実行処理で結果を保存し，終了のたびにcampaignの集計を更新する。
投入済み・実行完了・測定成立は区別して記録する。
GPU module等の起動前検査で失敗した場合も，実行用Pythonが利用可能なら同じ失敗記録を保存する。
元の診断出力はPBSのstdout/stderrに残る。

起動時に表示する `Attach:` のコマンドで専用tmuxへ接続する。`Ctrl-b d` でdetach，
`Ctrl-C` で中断する。起動成功でシェルへ戻った時点では実験は進行中である。
実験出力を `--output-dir /path/to/campaign` と指定した場合，監視記録は隣の
`/path/to/campaign.launcher/driver.log` と `launcher.json` に保存する。
実験ディレクトリを新規にする規則と，共通launcherの重複起動防止を両立するための分離である。
前景実行ではドライバの終了コードを直接受け取る。

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
toolkit を確認する。一括実行では以下のデータを自動生成する。既存の前処理済みデータを
指定して1条件だけ実行する場合は，`--profile` と `--data-dir` を併用する。

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
語彙は Worker コンテナからも参照できるパスを使う。CSX の生成設定には，
リポジトリの絶対パスを `cluster_config.mount_dirs`，その `src` を
`cluster_config.python_paths` として含める。入力ワーカー用イメージの構築が失敗して
仮想環境のマウントへ切り替わった場合も，ワーカーがソースを import できるようにする。
リポジトリ外のデータや語彙に必要な追加マウントは
`--mount-dir /absolute/path` で指定する。環境間のデータ転送は自動では行わない。
既存の `params.yaml` は自動更新しないため，修正後に新しい実行ディレクトリへ設定を生成する。

`./setup.sh` は [requirements.txt](requirements.txt) の NLP 依存も導入する。
以前作成した `.venv` は，両環境とも次のコマンドで更新する。仮想環境の再作成や
GNN データの再取得は行わず，インストール済み PyTorch のバージョンと CPU/CUDA ビルドを
固定して，対応する torchvision と datasets 等を追加する。

```bash
bash ./benchmark_scripts/non_gnn/setup.sh
```

起動時には `datasets`, `transformers`, `torchvision`, `h5py`, `filelock` の import を
データ取得・tokenizer ロード・ジョブ投入より前に確認する。不足やバイナリの不整合があれば，
使用中の Python と失敗した import，修復コマンドを表示して停止する。
CS-3／Pegasus の起動は `uv run --no-sync --project <repo> python ...` に統一する。
環境の有効化と `.venv/bin` の探索パスは uv に任せ，起動スクリプトでは `PATH` を変更しない。
`--no-sync` によりジョブ起動時の依存再解決・環境更新を省き，依存追加は上記の
セットアップ内の `uv pip install` で行う。CSX はデータ取得前とクライアント起動時に
`torch-cirh-opt` の存在を確認し，見つからなければ具体的な起動方法を表示して停止する。
Python ファイルを直接実行する場合も `uv run --no-sync` を付ける。
Model Zoo の設定検証は他の入力 processor も import するため，ローカルテキストや
前処理済みデータを使う場合にも `datasets` と `torchvision` が必要になる。
`--dry-run` は追加依存の確認を省いて予定だけを表示する。

GPU 実装は Transformers 4.57.3 と PyTorch 2.4 の API を使用する。

## 既存データで1条件のみ：CS-3

リポジトリのルートから実行する。`/data/...` はその環境の実パスへ置き換える。
最初は末尾に `--dry-run` を付けて確認する。

```bash
./benchmark_scripts/cerebras/run_non_gnn.sh --profile bert_large_msl128 --data-dir /data/bert/train_msl128
./benchmark_scripts/cerebras/run_non_gnn.sh --profile bert_large_msl512 --data-dir /data/bert/train_msl512
./benchmark_scripts/cerebras/run_non_gnn.sh --profile llama3p2_1b_msl1024 --data-dir /data/llama/train_msl1024
./benchmark_scripts/cerebras/run_non_gnn.sh --profile llama3p2_1b_msl2048 --data-dir /data/llama/train_msl2048
```

この `--profile` モードは1条件・1runを実行し，既定でtmuxを使う。設定だけを保存・検証する場合は
`--prepare-only --output-dir /path/to/new/run` を付ける。

## 既存データで1条件のみ：Pegasus

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
- `--warmup-steps N`: 両環境の計時から除く先頭更新数。既定20，`max-steps` 未満。
  CSX／Model Zoo GPUはログ端点が必要なため1以上。native GPUは0も指定できる。

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

campaignでは `r01/<profile>/` のように反復ごとの新しいディレクトリを作る。
各runには次を保存する。

- `params.yaml`, `launch.json`: 完全な設定，元設定と実行設定のSHA256，revision，dirty source内容hash，
  入力データと語彙の内容hash，準備manifest，起動引数。
- `environment.json`, `client_status.json`: 実行ホスト・パッケージ・CPU割当，開始・終了時刻，
  準備確認時間・クライアント時間・全体時間，終了コード，失敗・タイムアウト・中断。
- `console.log`: 学習ログ。Pegasus の投入結果は `qsub.log`。
- native GPU の `gpu_environment.json`: ソフトウェア版，GPU，パラメータ数，実装設定。
- native GPU の `metrics.jsonl`: 更新ごとの損失・消費数・時間，ウォームアップ後の処理速度とメモリ使用量。
- `result.json`, `step_metrics.csv`: 全更新の損失・時刻，job ID，測定成立の判定と理由，
  正確な計測区間・分子・秒数・処理速度，ローカルSDK成果物の一覧。

prepared runの実行直前に設定・source・入力・語彙の内容を再確認する。
キュー待ちの間に変更された場合は，準備時の条件による測定として集計せず，失敗理由と観測した差を保存する。

campaign直下の `summary.json` と `runs.csv` には全予定runを残し，条件ごとに独立した値，
測定成立数，失敗・中断・欠測・不安定件数，平均と標本標準偏差を保存する。
1runだけの標準偏差と未測定値は `null` にする。前半・後半の速度差は安定性の参考値として
記録し，不安定でも完全な有限測定は集計に残す。
GPUジョブの結果は完了時に自動反映する。保存済みデータから再集計する場合は次を使う。

```bash
uv run --no-sync -- python benchmark_scripts/non_gnn/measurements.py \
  --campaign /path/to/campaign
```

CSX jobはGNNと同じ `run`・`study` の2つのlabelで識別する。
例えば `run=bert-large-s128-b256-w4-r1` はBERT Large，系列長128，有効バッチ256，
DataLoader worker 4，反復1を表す。Llamaは `llama3.2-1b-s1024-...` と表示する。
`run` の値は60文字以内，`study` は出力先から決まる10桁の識別子で，同一campaign内で共通。
`csctl get jobs -a -l study=<識別子>` でcampaignを絞り込める。
完全な条件は `params.yaml` と `launch.json` に保持する。

GPU 計時はデータ取得，転送，forward/backward，optimizer を含み，初期化とウォームアップを除く。
予約トークン位置数による `nominal_tokens_per_second` と，マスクによる有効数を区別する。
BERTは非padding入力位置，MLM対象位置，NSP例数を分け，Llamaは損失対象位置を記録する。
CSXで実測していない有効token数は欠測とする。
CS-3 側はprogress logの厳密な端点 `(warmup_steps, max_steps]` から
`更新差 × 有効バッチ / 経過秒` を算出する。既定はstep20→200の180更新である。
同じ分子・区間からupdates/s，sequences/s，予約token位置/sを保存する。
native GPUはウォームアップ終了後に同期した計時開始点を明示し，summaryの時計を採用する。
比較時は両側の区間を揃え，native GPU の区間平均と CS-3 の単発 `Rate` を混在させない。
短い実行のウォームアップ不足，実行間のばらつき，GPU のメモリ適合性は実機で確認する。
保存した学習損失は有限性と経過を示す。精度・収束の判断は別の検証実験で行う。
既存のSDK成果物は保持して一覧化し，compilerの見積もりと実機カウンタを区別する。
詳細profilingは通常の反復測定と別に実行する。

## ローカル検証

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync -- python -m unittest discover -s benchmark_scripts/non_gnn/tests -v
```

テストは小さい CPU モデルと一時データで，損失の値と勾配，入力契約，設定，
`qsub` の引数境界と dry-run を確認する。実機のコンパイル・実行性能の検証は別途必要。
