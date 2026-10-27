# History embeddings for fixed-shape GraphSAGE

This prototype evaluates a sampled history variant with host mean aggregation. The reference architecture has the same two biased linear transformations, ReLU placement, dropout placement and classifier as the existing `gnn-modelzoo` GraphSAGE. The runtime lives in this ModelZoo fork; Wafer-GNN-manager owns the saved-result analysis.

`cerebras.modelzoo.models.gnn.tools.history_embedding.core` extends native GraphSAGE with a single-owner history store, fixed-shape batches, recursive expansion and exact layerwise refresh. `run.py` compares native CPU/CUDA learning. `hardware.py` and `campaign.py` implement a separate Cerebras SDK path. Local CUDA timing is not a prediction of CSX throughput.

## What is compared

| Mode | Boundary representations | Update policy |
|---|---|---|
| `tree` | Recomputed recursive tree | Exact gradients on the sampled graph |
| `tree_mean` | Only raw-feature means precomputed | Same forward/gradients; no history approximation |
| `snapshot1` | Cached hidden representations | All-node refresh each epoch |
| `snapshot5` | Cached hidden representations | All-node refresh every five epochs |
| `snapshot32` | Cached hidden representations | All-node refresh every 32 optimizer steps and at epoch starts |
| `online1` | Cached hidden representations | Training-root writes each step, plus all-node refresh each epoch |

The frozen sampled graph chooses up to F unique neighbors per node uniformly without replacement. The same graph is used at every depth. This makes the tree and fresh history forward computations comparable. It differs from the native `[15,10,5]` sampler; native depth-dependent neighbor contexts cannot be represented by one context-free history per node without changing the approximation. Full-neighbor validation uses the original undirected graph, with fresh weights and no historical embeddings. Training labels are never read from validation/test nodes. All-node refresh uses their features and edges, as in transductive node classification.

For layer l, the host sends `boundary_mean[l] = sum(history[neighbors outside batch]) / degree`. The model adds the current in-batch neighbor sum divided by the same degree, then applies its learned transformation. This preserves in-batch gradients and stops gradients through boundary histories. Even an exactly fresh history does not restore the omitted boundary gradient. Mean aggregation is essential to this representation; GAT attention and nonlinear per-edge message functions require another derivation.

Snapshot runs cache both histories and their neighbor means. Online writes invalidate the cached means. Padded roots never update node zero. Versions record the weights used to produce an embedding; age counts completed optimizer updates, and does not measure the recursively inherited approximation error. Refresh runs with dropout disabled. Cached state is not a trainable parameter or an optimizer embedding table.

## Execution

See [real-data deployment](README.md) for preparation, direct CSX execution,
tmux, numerical replay and full-neighbor checkpoint evaluation. Runtime entrypoints use the normal `PYTHONPATH=$PWD/src`; ALCF jobs use the existing job catalog and pinned source.

For local CUDA reproduction, use the fork environment and existing OGB data:

```bash
PYTHONPATH="$PWD/src" uv run --no-sync -- python -m cerebras.modelzoo.models.gnn.tools.history_embedding.run \
  --data-dir src/cerebras/modelzoo/models/gnn/data/datasets --device cuda \
  --output /tmp/history-arxiv-comparison --epochs 10 --seeds 42 43 44 \
  --modes tree tree_mean snapshot1 snapshot5 snapshot32 online1
```

Default learning uses H128/L3/F10/B256, dropout 0.5, FP32 and native AdamW
lr0.003 with zero weight decay. SDK runs use cstorch.optim.AdamW; optimizer
implementations have different epsilon/bias-correction placement. The SDK
numerical checker uses the SDK optimizer too. Timing includes host batches,
transfers, updates, history returns and refresh; native CUDA validation and
checkpoint serialization are outside its training timer.

Plotting and capacity analysis remain in Wafer-GNN-manager. There is no runtime import from that repository.

## Sources

- [GNNAutoScale paper](https://arxiv.org/abs/2106.05609)
- [GAS source](https://github.com/rusty1s/pyg_autoscale)
