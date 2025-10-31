# AGENT.md: GNN Enhancements

## Next Goal

## Recommended Local Commands

Run these from `src/cerebras/modelzoo/models/gnn`:

```bash
# Ensure datasets are present inside the repo sandbox
UV_CACHE_DIR=.uv_cache uv run download.py --root data/datasets

# CPU validation of the dense GCN path
UV_CACHE_DIR=.uv_cache uv run run.py CPU -p configs/params_gcn_pubmed.yaml -m train_and_eval
```

---

## Current Capabilities

- **GraphSAGE baseline:** `model.py` provides a static-shape `GraphSAGE` core with configurable depth, hidden size, dropout, and aggregator (`mean`, `sum`, `max`).
- **Static neighbor sampling:** `data.py` and the neighbor sampling pipeline expose deterministic, padded mini-batches for PubMed and Reddit. Use dataset aliases (`data.dataset`) or explicit names; profiles live in `configs/`.
- **Dataset preparation:** `download.py` fetches PubMed and Reddit with checksum validation and resumable downloads so `run.py` always sees processed artifacts.
- **Experiment configs:**
  - `configs/params.yaml` — PubMed GraphSAGE defaults.
  - `configs/params_graphsage_reddit.yaml` — tuned GraphSAGE-on-Reddit recipe and logging defaults (`model_dir_gnn/reddit_graphsage`).

## Operational Playbook

- **PubMed sanity check:** Use the PubMed configuration to confirm accuracy improves monotonically and metrics remain deterministic for repeated runs with the same seed.

- **Reddit throughput check:** Run the Reddit configuration to exercise the large-graph neighbor sampler; monitor `model_dir_gnn/reddit_graphsage/latest_run.log` for throughput and accuracy (~0.94 after warm-up).

- **Download refresh:**
  ```bash
  uv run download.py --root data/datasets
  ```
  Safe to rerun; resumes partial downloads and revalidates checksums.

## Development Guardrails

1. **Scope containment:** Only touch files under `src/cerebras/modelzoo/models/gnn/**` plus configuration assets in the same subtree.
2. **Static-shape contract:**
   - Pad node/edge IDs using `pad_node_id` from config; never branch on dynamic neighbor counts.
   - Maintain `batch_mask` semantics so the model can filter padded entries without altering tensor ranks.
3. **Determinism:** Seeding `sampler_seed` and `trainer.init.seed` must yield identical batches; add regression tests or log checksums if behavior changes.
4. **Config ergonomics:** Keep dataset-specific overrides in `dataset_profiles` so users can switch datasets via `data.dataset=<alias>` without duplicating knobs. Document new keys.
5. **Logging:** Ensure new pipelines emit steps/sec, train loss, validation metrics, and (when applicable) test metrics to `model_dir_gnn/<experiment>/metrics.jsonl`.

## Future Enhancements

- Optional PyTorch Geometric backend (`backend=pyg`) mirroring the static batch contract for fast prototyping while keeping `backend=vanilla` as the default.
- Extended benchmarking (e.g., ogbn-arxiv) once static padding requirements are mapped to larger OGB graphs.
