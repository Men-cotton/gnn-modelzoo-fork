# HPC Asia results requiring revalidation

This is an open research dependency, not a completed-work log. Retain it until the manuscript's results and reproduction instructions are updated. Checked on 2026-09-18 against `../Wafer-GNN-manager/paper/5.HPC-Asia/graphsage-cs3/src/main.tex`; `W/` below means that sibling repository. Existing manuscript and artifacts have not been rewritten.

## What the saved results establish

The public artifact snapshot `3dbe09aa22b8e10d7f6348190807d987b43bfbca` reproduces the displayed accuracy, throughput and WIO summaries with `artifacts/IA3-2026/scripts/reproduce_metrics.py --check`. Its `MANIFEST.json` identifies logs, configurations, job IDs and hashes. Recompute from that revision, not an arbitrary current artifact directory.

The current paper does not exercise ModelZoo's GPU cache/pinning path, multiple CSX input Worker replicas, full_graph, papers100M, distributed partitions or MAG preparation. Saved CSX runs have one Worker replica; H100 runs are single-process, ordinary PyG loaders. `drop_last` is False. In all six historical H100 profiles, `fit.val_dataloader` equals `validate.val_dataloader`, with split `valid`; `task.to_float16=True` agrees with enabled float16 precision. The mean aggregator matches the historical implementation, and the AdamW profiles contain no explicit eps/betas settings that the old runner would discard. The corresponding extension defects do not invalidate these observations.

However, all six paper CSX training graphs divide masked loss and gradients by fixed4096. This affects training semantics, especially padded tails. Accuracy uses its own valid-target count: the arxiv evaluation `executors/000002/cirh.mlir:196–207` confirms masked correct/count accumulation. Thus the saved accuracy is an observed result of the historical training objective; it is not evidence for the corrected objective, nor is its arithmetic itself shown to be wrong.

## Exact CSX runs to revisit

Under `W/artifacts/model_dirs/<directory>/cerebras_logs/<timestamp>/`, inspect `executors/000001/cirh.mlir` and the matching `wsjob-<ID>-cluster-details.json`. All six training IRs contain the fixed denominator at lines65–66 and loss/backward arithmetic at lines315–321. Each cluster-details record has one `WRK.taskMap` entry and one `WSE.taskMap` entry.

| Condition | Directory | Timestamp | Execute job ID, without wsjob- |
| --- | --- | --- | --- |
| arxiv accuracy | ogbn_arxiv_graphsage_accuracy | 20260624_190727 | mswfcbcppuzmiuapx9psr6 |
| products accuracy | ogbn_products_graphsage_accuracy | 20260628_041034 | ndve7gpbywsrd33jccuxnb |
| arxiv cached throughput | ogbn_arxiv_graphsage_throughput_cache | 20260624_141307 | v7j6tkhq7nfcrdburs7sui |
| products cached throughput | ogbn_products_graphsage_throughput_cache | 20260624_151908 | kqpue4uyt2s3wobgpkpw2c |
| arxiv uncached throughput | ogbn_arxiv_graphsage | 20260616_142904 | dw436fv2uoch2jqzzmfvco |
| products uncached throughput | ogbn_products_graphsage | 20260616_152437 | s4ghbzereuamjqqdqtzg7w |

Official train split sizes read from local gzip CSVs are arxiv90941 and products196615. With B4096, the tails contain829 and7 valid targets, every23 and49 batches respectively. The arxiv accuracy log `W/artifacts/raw_logs/2026-06-22/arxiv_graphsage_wse_accuracy_eval.log:1038,1082,1126` shows losses0.99537/0.19923/0.97547 at steps440/460/480, consistent with the tail effect. The exact scaling interpretation follows the ordinary masked CE semantics; no new numerical execution of historical CIRH was performed.

## Input parallelism and throughput definitions

The public snapshot nests a Cerebras loader inside the Trainer's wrapper. SDK2.10 local loader inspection sets `num_workers=0`; the second wrapper serializes the changed configuration. The removal is commit `55b4cbc` on September17, after the paper runs. The saved code, SDK behavior and reproduced old path support impact on the June runs, but there is no June child-PID record directly proving their effective worker count. The paper's configured40 must not be read as measured effective40.

The public manifest does not pin a source revision for each June CSX run; its H100 entries do identify a source revision. The artifact branch base is not proof of the exact source executed by the CSX jobs. The nested-loader impact is inferred from saved configurations, published source, SDK behavior and fix history.

The public reproduction script's CSX numerator is `4096 * delta_steps`; historical H100 `reference/pyg/train.py` counts `batch.batch_size`, the actual seeds. For normal continuous dataset traversal, subtracting tail padding from the same historical CSX time windows gives the following conditional reaggregation. These are not measurements of the corrected code.

| CSX condition | Interval | Nominal slots/s | Conditional valid targets/s |
| --- | --- | ---: | ---: |
| arxiv uncached | 20–500 | 1212.4838 | 1170.1737 |
| arxiv cached | 20–500 | 1241.0210 | 1197.7151 |
| products uncached | 40–1000 | 592.8472 | 580.5173 |
| products cached | 40–1000 | 590.8431 | 578.5549 |

The numerator decreases by3.4895% for arxiv and2.0798% for products. The four training windows contain no logged evaluation. Record both nominal slots and valid targets in replacement measurements, with warmup/evaluation/checkpoint boundaries stated.

To reproduce that conditional calculation, let `B=4096`, `N` be the train split size and `q=ceil(N/B)`. The interval `(s0,s1]` contains `floor(s1/q)-floor(s0/q)` tails; subtract that count times `q*B-N` from `B*(s1-s0)`. The arxiv intervals contain 21 tails / 68,607 padded slots; products contains 20 / 81,780. Recorded elapsed seconds are arxiv uncached/cached 1621.531/1584.244 and products uncached/cached 6632.670/6655.168.

`W/artifacts/r04_archive/20260918/arxiv_none_loader_fix_w02_w04/summary.json` contains post-loader-fix arxiv uncached results only: revision55b4cbc, 2/4workers, three repetitions each, interval40–440, evaluation disabled, loss still uncorrected. Those runs cannot replace the paper's accuracy, products or cached conditions.

## Reproduction conditions to correct

- The paper's H100 revision `8d1457914cfb8edfad40734500c663a483d13d90` defaults to8workers in `configs/components/architectures/input_pipelines/neighbor.yaml`. Actual six H100 logs record a dirty override and40workers. The appendix checkout command requires that override to reproduce the recorded condition.
- The appendix lists appliance-client1.14.0 and cluster/job-operator1.20.2. Matching execution entries in the six saved `cerebras_logs/run_meta.json` files instead report appliance-client2.10.0 and cluster/job-operator `3.2.1-202605191112-863-addb76f2+addb76f210`; for arxiv accuracy see lines834–843.
- AdamW defaults differ: CSX saved resolved config uses eps1e-6; PyTorch's historical default is1e-8. Matching learning rate and weight decay alone does not make all optimizer settings identical. Explicitly record the intended eps/betas for replacement runs.
- Fixed-shape CSX batches repeat deterministic target order/neighborhoods; ordinary PyG sampling does not establish the same optimization trajectory. Runtime revision `327b56a` also honors configured precision during PyG evaluation. When changing these comparison conditions, remeasure H100 as well.
- Compiler performance summaries are estimates, not physical end-to-end samples/s. H100 breakdown plots average logged cumulative averages; avoid interpreting them as an exact additive causal decomposition over one common window.

## Required closure

Re-run the two CSX accuracy conditions and four throughput conditions using runtime revision `327b56a` or a descendant with the corrected loader and explicit masked loss. Verify real remote child workers and the actual compiled loss/count before interpreting results. Preserve source revision, dirty diff if any, resolved configuration, job/runtime metadata and exact metric definitions. H100 must be re-run if optimizer, precision or sampling conditions change. No corrected accuracy or physical CSX throughput has yet been measured for this runtime revision.
