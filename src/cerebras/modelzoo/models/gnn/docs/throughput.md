# Measure Model Throughput

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