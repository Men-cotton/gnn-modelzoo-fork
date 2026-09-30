# ジョブ投入統一とローカルGPU測定（2026-09-30）

実装・単体テスト・RTX 4060 Laptop GPU実測を完了した。今回のsynthetic代表形状では、topology cacheによって測定区間の時間が61.856から21.273 ms/stepへ短縮した（2.908倍）。depth-wise linear fusion単独のend-to-end改善は確認できなかった。これは現行fixed-shape coreの逐次実行測定であり、実データの通常training throughputの実証ではない。

ジョブsource固定の追加実装・再起動後の検証は末尾の「追加: 1 run / 1 commit」に記録した。先行節は531c92bまでの記録であり、GPU結果と実行条件は変更していない。

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

## 追加: 1 run / 1 commit

source固定実装は`8c4ab0777f0091fc141f0a125f894f72bf3d04cb`、input/dispatch修正は
`e02e1452237af6e77c0a6e34c737a23c90b7d424`、等号形式・prepared契約・campaign mock境界の修正は
`1be3291f083abde7dfcc03b5b1adeaa9c2c02513`。先行commitは書き換えていない。

通常のrun開始時に現在branchのHEADを一度だけfull hashへ解決し、全jobで共有する。
過去commitはrun-level `--revision`にhashを指定する場合だけ使用し、job-level hash/revisionは拒否する。
`SourceRun`がtracked sourceだけのdetached worktreeをrunごとに作る。元checkoutを切り替えず、
別runのsource/manifestも上書きしない。dirty tracked sourceと既存untracked内容はsnapshotへ入らない。
旧commitに必要なpayload、driverのhandoff APIがなければ投入前に失敗する。

PBSの実cwdと`PBS_O_WORKDIR`、script/config、`PYTHONPATH`をsnapshot側へ揃える。
既存editable `.venv`を使う新規processで実際の`cerebras.modelzoo.__file__`を確認する。
`common.sh`/uv project検出、CSXの`python_paths`/`mount_dirs`も固定sourceを指す。
`.venv`は依存環境、既存dataは入力として可変bindingを別記録し、source hashと区別する。
run内の`model_dirs`は専用outputへ向ける。既存non-GNN cacheと診断baseの参照を保った。
インストール、依存sync、download、認証・設定変更はこの検証では行っていない。

`--json`は複数jobでも単一documentを返す。単一job用の`cwd`/`commands`は維持し、
`jobs[*]`にjobごとのcwd/argv/hashを記録する。異なるjobを含むsetを`json.loads`で消費する回帰を追加した。
`--json`/共通入口の`--dry-run`は出力を書かず、`--prepare-only`はsnapshot/manifest/import検証までで終了する。
低水準Python driverの科学条件dry-runには既存のconfig生成動作が残る。
manifestの`jobs`/`nested_jobs`/`dispatches`と既存campaign journalでdispatchを追跡する。
legacy handoffは`exec`で移るため、manifestの`running`だけを完了証明には使わない。

独立監査はlegacy `--base-config`がdirty checkoutを読む問題を小Git fixtureで実証した。
`e02e145`の別token形式向けfreezeを再利用し、`1be3291`では共通jobs driverとlegacyの両方で
`--base`/`--base-config`/`--config`の別token・`=PATH`形式を共通処理にした。
tracked YAMLと継承元は選択commitから読み、外部YAMLは解決済み内容とSHA-256をrun inputへ保存する。
無害な実payloadで、旧/新commit、dirty内容、plan後の変更、継承元、import元を比較した。
output/data引数はconfigとしてfreezeせず、呼出元からのpath意味を保つ。
`launch.json`を伴うprepared configと外側からの`--execute-config`は、契約を欠落させて持ち越さず
投入前に拒否する。pinned campaign内で再準備する通常経路は維持する。

### 保存snapshotとログ

再起動前の`--prepare-only`は次を作成した。再起動後にGit hash・実import元・固定input digestを再確認済み。

- manifest: `model_dirs/source_version_validation_20260930/manifest.json`
- source: `model_dirs/source_version_validation_20260930/source/`
- resolved commit: `e02e1452237af6e77c0a6e34c737a23c90b7d424`
- status: `prepared`。scheduler/CSX clientはこの準備では起動していない。
- [再確認結果](source-version-snapshot-check.json)

```bash
PYTHONPATH=src timeout 60s .venv/bin/python -m cerebras.modelzoo.tools.benchmark_launcher jobs \
  --job csx-config --config-index 1 \
  --run-dir "$PWD/model_dirs/source_version_validation_20260930" --prepare-only
```

この保存先は既に存在するため、再実行には新しい`--run-dir`が必要。
既定snapshot先は既存project内の`model_dirs/job_runs/<UTC>-<unique>/source`であり、
実schedulerではsource、run inputs/output、venv、dataの同一pathが計算nodeから見えることが条件となる。
`/tmp`の共有を仮定していない。remoteからの可視性とphysical CSX実行は未検証。

WSL再起動で`/tmp/gnn-modelzoo-handoff-20260930`は消失した。repo内に保存済みのGPU証拠と上記snapshotは残った。
再起動前のjobs 42、autotune 22、non-GNN 58、fixed-shape wrapper 2の通過はsession記録・独立引継ぎの結果であり、
失われたraw logを復元できたとは扱わない。GPU測定は再実行していない。
新しい生ログは`model_dirs/handoff_recovery_20260930/source-final-verification/`に保存した。
TMPDIR/cacheの実体も同じWSL ext4内にある。長いTMPDIRがUnix socketのpath制限を超えたため、
`/tmp/modelzoo-test-tmp-20260930`を永続`model_dirs/handoff_recovery_20260930/tmp`への短いsymlinkとして使用した。
初回のpath長エラー、fixture引数不足、alias表記差の失敗ログも生ログdirectoryへ残した。

### 再起動後の最終検証

| 対象 | 結果 | 保存ログ |
| --- | --- | --- |
| jobs/launcher/source fixture全体 | 47件 PASS、9.156秒 | [log](source-version-jobs.log) |
| 上記のjob plan/source部分だけ | 28件 PASS、5.990秒（47件に含む） | 永続生ログ `plan-source-complete.log` |
| HPC Asia/learning/worker campaign | 35件 PASS、6.553秒 | [log](source-version-campaign.log) |
| non-GNN互換の対象4件 | 4件 PASS、14.234秒 | [log](source-version-legacy.log) |
| fixed-shape qsub wrapper | 2件 PASS、1.770秒 | [log](source-version-fixed-shape-jobs.log) |

Black、Ruff、`git diff --check`も通過した。commit後のwrapper検証は`1be3291`のsourceで実行した。
Git fixtureはdefault HEAD一回解決、同一run全jobの同一hash、並列run独立性、run途中のHEAD更新、
dirty source/dirty symlink除外、旧commit、必要API欠如、準備失敗時のsubmitゼロを検証する。
config payload試験では3種類のflag×2形式×2revisionの12条件で、実際に選択sourceのYAMLを読み、
loss/速度の測定ではなく入力・provenanceの一致を確認した。

再現コマンド（cwdはrepository root）。短いWSL内aliasを安全に準備する:

```bash
python3 - <<'PYTHON'
from pathlib import Path
alias = Path('/tmp/modelzoo-test-tmp-20260930')
target = Path('model_dirs/handoff_recovery_20260930/tmp').resolve()
target.mkdir(parents=True, exist_ok=True)
if alias.is_symlink():
    assert alias.resolve() == target
elif alias.exists():
    raise SystemExit('Unexpected existing TMPDIR path')
else:
    alias.symlink_to(target, target_is_directory=True)
PYTHON
```

テスト本体:

```bash
TMPDIR="$PWD/model_dirs/handoff_recovery_20260930/tmp" PYTHONPATH=src timeout -k 5s 90s \
  .venv/bin/python -m unittest discover -s benchmark_scripts/tests -v
TMPDIR=/tmp/modelzoo-test-tmp-20260930 PYTHONPATH=src timeout -k 5s 90s \
  .venv/bin/python -m unittest discover -s src/cerebras/modelzoo/models/gnn/tests -p 'test_*campaign.py' -v
TMPDIR="$PWD/model_dirs/handoff_recovery_20260930/tmp" PYTHONPATH=src timeout -k 5s 60s \
  .venv/bin/python -m unittest discover -s src/cerebras/modelzoo/models/gnn/tests -p test_fixed_shape_jobs.py -v
PATH="$PWD/.venv/bin:$PATH" TMPDIR="$PWD/model_dirs/handoff_recovery_20260930/tmp" \
  UV_CACHE_DIR="$PWD/model_dirs/handoff_recovery_20260930/uv-cache" \
  PYTHONPATH=src:benchmark_scripts/non_gnn/tests timeout -k 5s 120s .venv/bin/python -m unittest \
  test_non_gnn.BenchmarkTests.test_dry_run_never_calls_qsub_or_writes \
  test_non_gnn.BenchmarkTests.test_qsub_preserves_config_path_with_spaces \
  test_campaign.CampaignTests.test_default_wrappers_dry_run_without_download_or_writes \
  test_shell_launch.ShellLaunchTests.test_gpu_module_failure_records_result_without_starting_training -v
```

### CSX client起動の検証事故と限界

再起動前、campaignの新しい`pin_entry`再実行境界が既存単体テストのmockを越え、
CSX clientが1回起動してcluster接続を試みた。これは「mockのみ」という指示に反する検証不備だった。
当時確認したログは`Failed to connect to cluster. Unable to start workflow.`、launcher exit codeは2で、
workflowは開始できなかった。専用tmux sessionの終了も当時確認した。失われた`/tmp` raw logについては
現在のsession記録以上の証拠を主張しない。

3 campaign TestCaseの`setUp`で再実行境界をmockし、client/launcherには想定外呼出しを即失敗にする
既定guardを追加した。その上で各テスト固有のmockを使って35件を完走した。
再起動後の検証ではCSX client、remote/有料job、GPU測定を起動していない。
今回のsource固定を実scheduler/remote workerで実証した結果ではなく、Git fixture、ローカルpayload、
import検証、dry-runとmockによる確認である。依存環境・大規模dataの内容固定も別途必要。

主な追加変更は`tools/job_source.py`、`tools/job_plan.py`、既存wrapper/common.sh、campaignのhandoff/dispatch、
`benchmark_scripts/tests/test_job_source.py`/`test_job_plan.py`とcampaign mockテスト。
正確なファイル一覧は`git show --stat 8c4ab07 e02e145 1be3291`で確認できる。
終了前のプロセス確認で、今回のclient/test/専用tmuxの残存は0件だった。
OFFSET-GNN/とCSL repoには触れず、pushと既存commitの書換えを行っていない。
