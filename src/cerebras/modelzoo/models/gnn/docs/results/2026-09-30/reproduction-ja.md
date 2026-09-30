# ジョブ投入統一とローカルGPU測定（2026-09-30）

実装・単体テスト・RTX 4060 Laptop GPU実測を完了した。今回のsynthetic代表形状では、topology cacheによって測定区間の時間が61.856から21.273 ms/stepへ短縮した（2.908倍）。depth-wise linear fusion単独のend-to-end改善は確認できなかった。これは現行fixed-shape coreの逐次実行測定であり、実データの通常training throughputの実証ではない。

## 保存先とsource

- cwd: `/home/mencotton/research/gnn-modelzoo`。開始HEAD: `390c31772d97f2105b98df77429e27354cef9b69`。
- jobs commit: `3443d83`。GPU実装・benchmark commit: `74d10627ff4c18770dc9f34fac6d66c113e63509`。最終測定JSONのsource HEADも後者である。
- [最終代表結果](representative-final.json)、[最終smoke](smoke-final.json)、[集計](summary.json)、[環境](environment.json)、[filesystem](storage.txt)を同梱した。runtime source SHA-256は各JSONにあり、保存時に現ファイルと照合済み。
- 元のbenchmark出力は `model_dirs/gpu_optimizations_20260930/`、独立ログは `/tmp/gnn-modelzoo-handoff-20260930/`。source・data・TMPDIR・結果・cacheはWSL内。`/mnt/c`、UNC、Windows経由の測定I/Oは使っていない。
- OFFSET-GNN/は読まず変更していない。CSLリポジトリは対象外。外部job投入、download、install、push、既存commitの書換えは行っていない。

## 投入経路の変更

`benchmark_scripts/jobs.json` がjob/set、PyGの11 profile、HPC Asiaのdataset/seed/run、input sweepの展開表、CSXの8設定、診断worker一覧を保持する。`benchmark_launcher jobs` → `tools/job_plan.py` がplan生成・事前検証・argv実行を担当する。

既存Pegasus submitterは互換wrapperとなり、non-GNNのsingle/campaignも同じqsub builder/executorを使う。CSXの3 campaign wrapperと`run_modelzoo.sh`も共通入口を使い、診断実行は既存launcherで監督する。schedulerはrepo root、campaignは呼出元、単一CSX設定は従来のモデルdirectoryをcwdとする。campaign内の研究条件生成、予算、lock、journal、trial停止、PBS payload、run.pyの条件は維持した。HPC AsiaのPython planとPegasus planは同じrun一覧を読む。

全planの設定を投入前に検証する。未知job/set/option、schema/type不正、YAML継承・重複キー、PBS path、qsubのcomma/newline/NULを検査する。argvはshellを経由しない。fixed-shapeの空白入りconfig pathは従来どおり許可し、wrapperの`--`境界でselectorの上書きを防ぐ。scheduler不在では新しい出力先を作らない。`jobs --dry-run`/`--json`は投入も出力directory作成もしない。既存診断wrapperのdry-runだけは従来どおりconfigを生成する。

主な変更ファイルは `benchmark_scripts/{jobs.json,README.md,tests/test_job_plan.py}`、`src/cerebras/modelzoo/tools/{job_plan.py,benchmark_launcher.py}`、既存10 shell wrapper、`benchmark_scripts/non_gnn/campaign.py`、`tools/hpcasia_campaign.py`。完全な一覧は `git show --stat 3443d83` で確認できる。

## sampler・mask・最適化の契約

現行samplerはtargetを`seed + epoch`で並べ替え、近傍は`(seed,node_id,hop)`から決まるrotationを使う。近傍自体はepoch間で固定である。hop特徴は`[B,p_l,D]`、node maskは`[B,p_l]`、neighbor maskは`[B,p_l,fanout]`。mean分母は有効neighbor数を1以上へclampした値であり、masked child featureもゼロにする。loss/accuracyはtarget maskと`label != -100`を併用する。

| 候補 | 実装と再利用条件 | 制限 |
| --- | --- | --- |
| topology cache | node/hopごとのsampled ID/maskを前計算。targetは毎epoch再配列し、features/labelsは毎batch取得する。graph/seed/fanouts/pad/node数を固定する | tableは`9*N*sum(fanouts)` bytes。上限を超えれば構築前に拒否。graphをin-place変更したらloaderごと再構築。scratchやworker複製はtable上限に含まれない |
| depth linear fusion | 同一layerの各depthで集約した後、self/neighbor Linearをまとめて実行。split後のactivation/dropoutは元と同じdeepest→root順・shape | concatと追加bufferの費用がある。GEMMの丸め順は変わり得る。dropout/learned transformを越えて集約しない |
| raw-feature mean前計算（検討のみ） | raw featureが不変かつ非学習で、graph/samplingも固定ならfirst layerの入力集約を再利用可能 | learned state/dropout出力のcacheは不可。deepest payloadを置換しない限りH2D削減にならず、今回は実装対象にしなかった |

native `fixed_shape_gpu.py` の `--topology-cache-mb N`、`--fuse-depth-linear`で明示的に有効化する。既定およびCerebras経路はVanillaを維持した。PBS wrapperから新flagの転送は未対応。`static_batch_cache_size`は使っておらず、引き続き診断専用である。

変更は `gpu_optimizations.py`、samplerのoptional hook、native runner、`tests/test_gpu_optimizations.py`、`tools/benchmark_gpu_optimizations.py`、`docs/fixed_shape_gpu.md`。最後の文書では古い「target順序も固定」の記述を現コードに合わせた。

## 容量見積りと実測条件

現configのB4096・fanouts `[15,10,5]`・D128・H1024では、FP32のraw featuresだけで1.789 GiB、最初のnode-mask適用で同容量、first hidden statesで2.594 GiBになる。さらに最深hopのmasked multiply temporaryは約1.465 GiB、backward保存、勾配、dropout、optimizer、workspaceが必要になる。これはlive peakの厳密な和ではないが、8GB機で巨大OOMを試す根拠にはしない。B4096は実行せず、明示synthetic graphの小形状を選んだ。OOMは発生していない。

| 条件 | smoke | 代表形状 |
| --- | --- | --- |
| B / fanouts | 8 / `[3,2,2]` | 64 / `[5,3,2]` |
| graph nodes / D / H / layers | 127 / 16 / 32 / 3 | 509 / 64 / 128 / 3 |
| warmup / measured steps / repetitions | 3 / 4 / 3 | 10 / 20 / 4 |
| dropout / precision | 0.5 / FP32 | 0.5 / FP32 |
| workers / torch CPU threads | 0 / 1 | 0 / 1 |

Python3.11.13、torch2.4.0+cu121、PyG2.7.0、CUDA12.1、RTX4060Laptop（API報告8,585,216,000 bytes）を使用。PyGの演算経路は今回使っていない。TF32/AMP/compileは無効。AdamWはlr=.003、eps=1e-6、betas=(.9,.999)、weight_decay=.01、bias減衰0であり、現native runnerのparameter groupingを再利用した。全variantは同じ初期state、target/features/masks、dropout seed、optimizer条件を使う。

測定は現行`GraphSAGE`、sampler、masked lossを直接使うcore microbenchmark。batch sampling/gather → H2D → forward/backward/AdamWを逐次実行し、各境界でCUDA同期する。`e2e_ms`はこの区間のwall time。dataset/index/model構築、cache構築、warmup、seed設定、結果digest/集計/保存は区間外である。DataLoader prefetch、task adapter、validation、checkpointを含む通常runner全体のthroughputではない。

`device_stage_wall_ms`にはPython dispatchと同期を含む。CUDA eventは同一streamの経過時間で、host dispatch待ちを含み得る。kernel時間の総和やPCIe実転送量を意味しない。代表形状は4方式の順を反復ごとに回転させ、各方式が各順番を1回ずつ担当する。clock/powerは固定していない。peak VRAMはwarmup、model、AdamW stateを含むPyTorch allocatorの値で、whole-device使用量ではない。

## 結果

値は「各反復のstep平均」をさらに平均した値 ± 反復間sample SD、単位ms。最終代表結果のみを主結論とする。

| 方式 | host | H2D wall | device stage wall | E2E | peak allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 44.075 ± 1.914 | 3.199 ± 1.143 | 14.582 ± 3.383 | 61.856 ± 5.894 | 22.383 / 28 |
| topology | 0.989 ± 0.048 | 3.317 ± 0.747 | 16.967 ± 2.993 | 21.273 ± 2.971 | 22.383 / 28 |
| fusion | 45.803 ± 1.416 | 3.179 ± 0.353 | 14.045 ± 2.510 | 63.027 ± 3.342 | 23.249 / 30 |
| combined | 0.981 ± 0.029 | 3.404 ± 0.534 | 17.334 ± 0.949 | 21.718 ± 0.515 | 23.249 / 30 |

| 方式 | H2D CUDA event | device CUDA event |
| --- | ---: | ---: |
| baseline | 2.812 ± 1.046 | 14.143 ± 3.101 |
| topology | 2.898 ± 0.670 | 15.912 ± 2.475 |
| fusion | 2.708 ± 0.353 | 13.137 ± 2.266 |
| combined | 3.011 ± 0.521 | 16.466 ± 1.030 |

Topologyの短縮はhost sampling部分に現れた。H2D payloadとGPU演算量を減らしたという主張ではない。fusion単独はE2E平均が1.89%遅く、combinedもtopology単独より平均2.09%遅い。ばらつきがあるため、fusionの追加効果は実証していない。小さいsmokeのE2Eはbaseline15.101±3.872、topology11.619±3.342、fusion13.456±1.163、combined8.824±1.376 msだったが、この短いcaseを主結論にはしない。

代表形状のcache tableは45,810 bytes、構築費用はtopology54.112 ms、combined53.596 ms。`build_ms / (baseline_ms - topology_ms)`から約1.33 step、整数なら約2 stepが今回の算術上の償却点となる。大規模graphの全node前計算やworker複製には外挿できない。featuresが更新されてもこのtopology cacheは再利用できるが、sampling方式やgraphを変える場合は再構築が必要である。

代表形状の全16測定行でpayload SHA-256が一致した。各反復は1,274 valid targets / 1,280 slotsで、epoch再配列と2回のpadded tailを含む。topologyのloss差は0。fusion/combinedの30更新中、保存した測定区間の最大loss差は`5.734e-5`だった。数学的同値性とbitwise同一性を区別する。CPU/CUDAの単一stepテストではlogits/loss、全parameter gradient、AdamW state/更新、accuracyを許容誤差つきで比較し、native runnerの5-step checkpoint比較も通過した。

## 検証と再現コマンド

repo root、既存`.venv`、GPUとローカルIPCが見えるWSL shellから実行する。GPU不可時は失敗として保存し、CPUに切り替えない。下記の出力名は今回使用した名前なので、再実行時は未使用の名前へ変える。各benchmarkは内部150秒・外側180秒の制限がある。

```bash
PYTHONPATH=src TMPDIR=/tmp/gnn-modelzoo-handoff-20260930/tmp timeout 180s \
  .venv/bin/python -m cerebras.modelzoo.models.gnn.tools.benchmark_gpu_optimizations \
  --case smoke --timeout-sec 150 --output model_dirs/gpu_optimizations_20260930/smoke-final.json
PYTHONPATH=src TMPDIR=/tmp/gnn-modelzoo-handoff-20260930/tmp timeout 180s \
  .venv/bin/python -m cerebras.modelzoo.models.gnn.tools.benchmark_gpu_optimizations \
  --case representative --timeout-sec 150 --output model_dirs/gpu_optimizations_20260930/representative-final.json
```

単体テストはunittestを使用。29 jobテスト、62 runtime/campaignテスト、24 non-GNN campaignテストが通過した。その後の限定変更には13 HPC Asia、2 non-GNN、最終7 optimizationテストを再実行し、すべて通過した。最終7件にはCPU/CUDA数値比較、persistent workerでのepoch再配列、native runnerのopt-inとcheckpoint比較を含む。重複するテスト群なので件数を足し合わせない。

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src \
  UV_CACHE_DIR=/tmp/gnn-modelzoo-handoff-20260930/uv-cache timeout 60s \
  .venv/bin/python -m unittest discover -s benchmark_scripts/tests -v
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src:src/cerebras/modelzoo/models/gnn/tests \
  UV_CACHE_DIR=/tmp/gnn-modelzoo-handoff-20260930/uv-cache timeout 120s \
  .venv/bin/python -m unittest test_hpcasia_campaign test_learning_campaign \
  test_worker_campaign test_fixed_shape_jobs test_neighbor_shuffle test_gpu_optimizations test_fixed_shape_gpu -v
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src \
  UV_CACHE_DIR=/tmp/gnn-modelzoo-handoff-20260930/uv-cache timeout 120s \
  .venv/bin/python -m unittest discover -s benchmark_scripts/non_gnn/tests -p test_campaign.py -v
PYTHONPATH=src TMPDIR=/tmp/gnn-modelzoo-handoff-20260930/tmp timeout 90s \
  .venv/bin/python -m unittest discover -s src/cerebras/modelzoo/models/gnn/tests -p test_gpu_optimizations.py -v
```

zero degree、duplicate（既にundirectedとされる入力と暗黙undirectedの双方）、invalid parent、不正padding ID、padded target、ignored label、全target無効、dropout 0/.5、train/eval、mean/sum/maxを確認した。Black、Ruff、変更shellの`bash -n`、コード・文書の`git diff --check`も通過した。診断dry-runの原文ログには表示由来の行末空白2行を保持した。qsub/CSXはmockまたはdry-runだけで検証した。native runnerへの新flag接続は実CUDAのtiny graphでも確認した。

Job planの再現例:

```bash
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.tools.benchmark_launcher jobs --list
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.tools.benchmark_launcher jobs --job fixed-shape --help
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.tools.benchmark_launcher jobs --set hpcasia-all --output /tmp/hpcasia-preview --json
benchmark_scripts/pegasus/submit_gpu_input_sweep_nqsv.sh --output /tmp/input-preview --phase workers --workers '2 4' --dry-run
benchmark_scripts/cerebras/run_modelzoo.sh --config-index 1 --dry-run
```

## 失敗・予備測定・未検証範囲

初回sandboxでは`/dev/dxg`が見えず、CUDA初期化は`Found no NVIDIA driver`で失敗した（[ログ](gpu-probe.log)）。sandbox外WSLの同じ`.venv`でRTX4060を確認し、以後は実GPUで測定した（[確認ログ](gpu-unsandboxed-probe.log)）。外側primaryへの初期probe依頼はこの確認で解消した。GPU unavailableをCPU性能として扱っていない。

初回jobテストはtmuxのsandbox IPCで1件失敗し、同条件のsandbox外再実行で通過。non-GNN初回はuv cache書込みとcompiler PATHの環境条件で5件失敗し、WSL `/tmp` cacheと既存venv PATHで24件通過した。最初のcampaignテスト起動はtest helper探索path不足で失敗し、`PYTHONPATH`へ既存test directoryを追加して再実行した。失敗ログも同梱した。Blackはsandboxのmultiprocessing待ちで停止し、今回起動したPIDだけを終了してsandbox外で通過した。

`smoke-pilot.json`は短い初回確認で、biasにもdecayする暫定optimizer設定だったため最終比較から除外する。`representative-pilot.json`は短いwarmup/3反復で、formatter終了待ちと時間が重なる予備測定として残す。主結論はcommit済みsource・現native optimizer policy・4順序均等化で取り直した`*-final.json`のみを使った。

`8daed847dd9c`は現repositoryに存在せず、`8d1457914cfb`は存在した。指定の`artifacts/IA3-2026/MANIFEST.json`と`scripts/reproduce_metrics.py`は存在しないため、その`--check`は実行できない。他repositoryは探索していない。過去H100/PyG値は形式が異なるためbaselineにしていない。

実データ、大規模graph、B4096、FP16/BF16、compile、通常trainingのworker/prefetch overlap、複数GPU、実scheduler、physical CSXは未検証。今回のbounded実装・ローカル測定に残るblockerはない。上記範囲の性能・学習精度への一般化には別の測定が必要である。
