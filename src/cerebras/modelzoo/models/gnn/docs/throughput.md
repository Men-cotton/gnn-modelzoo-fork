# Measure Model Throughput

## GNN experiment metrics

For new GNN comparisons, use explicit interval measurements instead of the
SDK's smoothed `Rate` or cumulative `GlobalRate`. Both CSX `measure_window.py`
and native GPU `measure_fixed_shape.py` use exact logged endpoints and count
completed batches in `(start_step, end_step]`. The GPU endpoints synchronize
CUDA. Endpoint elapsed time includes logging; setup and warmup are excluded,
and overlapping evaluation or checkpoint activity is rejected. The native
runner's final summary instead sums windows that exclude logging and must not
be silently substituted for this common comparison boundary.

New CSX Study runs enable `measure_batch_accounting` on the input loader. This
logs `GNN_INPUT_CONTRACT`, containing actual per-batch seed and supervised-label
counts derived from ordered split IDs and labels, their digest, batch size,
drop-last behavior and traversal scope. The parser verifies one fresh train
loop beginning at global step 1 and one input streamer. It rejects checkpoint
restore, earlier evaluation/executor restart, static replay and missing or
conflicting accounting metadata in strict mode. Counts are exact for that
recorded deterministic schedule; they are not device performance counters.
ModelZoo's SDK Repeater and MegaBatcher preserve order within that single
executor. A resumed or multi-executor run requires a different accounting
contract and cannot use the fresh-run formula.

Report these three numerators over the same elapsed seconds:

- `seed_nodes`: actual unpadded target occurrences (`target_mask`), including
  repeated visits on later epochs; not unique graph nodes.
- `supervised_targets`: seed occurrences whose label is not `-100`.
- `nominal_slots`: fixed batch slots, including padding.

For example, B=4 with 9 targets emits counts `[4, 4, 1]`. Steps `(2, 5]`
consume `[1, 4, 4]`: 9 seeds and 12 nominal slots. Ignored labels reduce the
supervised count independently. Do not compare CSX nominal slots/s directly
against GPU seed nodes/s. Both rates and their counts are retained in new Study
results, and ranking uses seed nodes/s. The same representation and the same
optimizer/precision settings still need to be verified separately; ordinary
PyG uses a different sampling representation and is a separate baseline.

Standalone use:

```bash
PYTHONPATH=src .venv/bin/python -m cerebras.modelzoo.models.gnn.tools.measure_window \
  train.log --start-step 40 --end-step 240
```

Runtime accounting metadata and one fresh train executor are mandatory.
Static replay uses the separate `--static-replay-probe` measurement path and
reports `probe_nominal_slots_per_second`; it is not an ordinary training rate.

GPU `--measure-input` and shared `worker_diagnostics` provide logical payload
bytes, loader/sampler timing, process resources and CUDA allocator peaks using
ordinary user access. Keep diagnostic overhead separate from final repeated
throughput measurements. GPU stream timings, CPU wait and worker phase times
overlap and are not an additive device utilization decomposition.

Learn how to measure the training throughput of your model to evaluate performance and optimize efficiency.
It is often desirable to measure the throughput of a model. In order to provide this Notermation out of the box, Cerebras Model Zoo runs print a couple of throughput metrics to the console as well as sending them to events files that are then viewable in TensorBoard. This section describes what these metrics are, how they are calculated, and intricacies to be aware of when interpreting them.
​
## Instant Throughput Measurement with Cerebras Model Zoo
There are two throughput metrics that are printed to the console by default when running a model using Cerebras Model Zoo. These are Rate and GlobalRate. It is important to note that these metrics are what’s measured by the user node. While they are useful for getting an overall picture of throughput, they are not exact measurements of throughput seen by the Cerebras Wafer-Scale Cluster. This is due to the asynchronous nature of execution on Cerebras Wafer-Scale, where input workers stream data to the wafer quasi-independently of the user node that’s receiving the outputs.
​
## GlobalRate
GlobalRate measures average throughput over the training run, excluding the time spent writing the final checkpoint. In contrast, total_time includes this final checkpoint time. As a result, dividing total_samples by total_time may yield a lower value than global_rate, which is based on the time at which the last sample was processed It does so by dividing total samples for which outputs have been received by total time since the executable was fully loaded to the Wafer-Scale Engine.
GlobalRate is also logged to events files and is viewable in TensorBoard as avg_samples_per_sec.
​
## Rate
Rate measures a smoothed out version of GlobalRate, where at each sampling interval (i.e. logging step), a smoothing factor (default of 0.4) is applied to the previously calculated Rate and added to the local throughput since the last sampling point.
Rate is also logged to events files and is viewable in TensorBoard as local_samples_per_sec.
While Rate is more susceptible to spikes than GlobalRate, it is more representative of the current throughput measured by the user node.
​
## Ephemeral throughput spikes after checkpointing
During a checkpointing step, wafer stops processing samples until a checkpoint is taken. Once checkpointing on the Wafer-Scale Cluster is complete, streaming samples from input workers is immediately resumed. However, while the WSE is processing samples post the checkpoint step, user node may still be downloading the checkpoint from the Wafer-Scale Cluster, which could take some time, especially for large checkpoints. As such, the WSE may have computed outputs for a number of samples but user node will only fetch those outputs once it has fully downloaded the checkpoint. As a result, once checkpointing on the user node is complete and it starts fetching outputs from the Wafer-Scale Cluster, a large number of outputs may be readily available for fetching, which could result in a large spike in local throughput (i.e., Rate) seen. Once user node catches up to the wafer, throughput will stabalize and return to normal.
This effect is less pronounced in GlobalRate when doing long training runs since it’s amortized over the entire training duration.
​
## Throughput in Weight Streaming execution
In Weight Streaming execution, outputs (such as losses, summaries, etc.) are received as soon their values have been computed by the wafer (except after a checkpointing step, as described above). As such, there’s a close one-to-one correspondence between the throughput achieved by the wafer vs. what the user node sees (i.e., Rate and GlobalRate). Having said that, the first few logging steps may present outlier throughputs due to difference in when the clock is started on the user node vs. when the wafer actually starts processing data. This effect is short-lived and steady-state throughput is achieved quickly thereafer.
​
## Conclusion
Accurately measuring the throughput of a model using the Cerebras Model Zoo provides vital insights into the performance and efficiency of the model on the Wafer-Scale Engine. The two key metrics, Rate and GlobalRate, offer a snapshot of instant and average throughput, respectively, aiding in the overall assessment of the model’s execution speed. It’s crucial to understand the nuances of these measurements, including their susceptibility to spikes during checkpointing and the influence of weight streaming on throughput visibility. By monitoring these metrics, users can gain a comprehensive understanding of their model’s performance, facilitating optimizations and ensuring effective utilization of the Wafer-Scale Engine’s capabilities.
