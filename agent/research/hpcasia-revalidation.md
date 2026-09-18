# HPC Asia measurement requirements

The physical CSX learning and throughput measurements for the current implementation remain to be acquired. Local tests and SDK lowering checks establish their tested contracts; they do not establish device accuracy, throughput or scaling.

## Learning evidence

Run [learning curves followed by tuning](../../src/cerebras/modelzoo/models/gnn/docs/learning_campaign.md) separately for arxiv and products. Record complete training-loss curves and full valid-split loss and accuracy, source identity, resolved configuration, environment, and execution job IDs. Researchers decide whether learning is adequate. The driver records `pending_human_review` and uses execution integrity, not an accuracy threshold, to decide whether tuning may proceed.

Verify the real compiled training loss and gradients use an explicit FP32 valid-target count, including padded tails and ignored labels. Evaluate accuracy over real supervised validation targets. Host oracles and the authored [loss fixture](../reference/fixtures/masked-loss.mlir) are supporting checks; collect the actual model IR and device results as separate evidence.

## Throughput evidence

Use one input streamer and a fresh uninterrupted training executor. CSX logs must include `GNN_INPUT_CONTRACT`, the ordered target/label digest, per-batch real seed counts, supervised counts and exact step timestamps. Compute rates over completed batches `(start_step, end_step]`. GPU logs must record actual consumed seed/supervision counts and synchronized endpoint times. Reject missing contracts, duplicate runs, nonfinite losses and overlapping evaluation/checkpoint work.

Report real seed occurrences/s, supervised target occurrences/s, and nominal fixed slots/s with their respective counts. Padding is part of nominal slots only. CSX counts are derived from the recorded sampler schedule and completed steps; GPU counts are consumed-batch measurements. Neither metric is a physical device-utilization counter.

First select input parameters independently per platform. Then acquire at least three fresh runs per selected configuration over the same declared window, normally step40–840. Retain each independent value, mean, dispersion, failed/unstable counts and settings. Winner-selection trials are not the final comparison repetitions.

## Matched conditions and representation

Preserve dataset/split identity, batch size, fanouts, seeds, precision and explicit AdamW learning rate, weight decay, eps and betas. Use the fixed-shape GPU implementation to compare the same model and sampler representation. Report ordinary PyG as a separate baseline because its sampling and model parameterization differ. A single seed's observed accuracy does not establish accuracy equivalence.

GraphCache is a host feature cache on CSX and fixed-shape GPU. Static batch replay is an input-mechanism experiment, not ordinary training. DataLoader child workers and CSX input Worker replicas are separate quantities. Verify effective remote worker settings and child PIDs from diagnostic records.

## Mechanism and provenance

Collect worker sampling/gathering time, parent loader wait, logical payload sizes, process CPU/RSS/PSS, cgroup limits/pressure and GPU allocator statistics through ordinary user-readable interfaces. Store unavailable fields as missing. Run detailed diagnostics separately from final throughput repetitions because their overhead changes timing. Host wait, transfer and compute times may overlap and do not form an additive utilization decomposition.

Preserve the exact source revision and dirty-source hash, resolved YAML, package/runtime versions, model and dataset identifiers, execution job IDs, raw measurements and derived summaries. Compiler performance estimates, physical measurements and local functional tests must remain distinct in the manuscript. Extending claims to other datasets, models or input-streamer counts requires corresponding measurements.
