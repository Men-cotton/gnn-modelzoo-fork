# GPU measurement window study

The initial PyG tuner defaults are **40 warm-up steps + 400 measured steps**
(440 total), followed by **40 + 800 steps**, repeated at least three times for
finalists. CSX defaults remain 40+200 and 40+400. GPU settings can be changed
with `--warmup-steps`, `--measure-steps` and `--confirm-steps` in a new study.

These choices use eight completed GraphSAGE single-GPU throughput logs in
`~/research/Wafer-GNN-manager/artifacts/raw_logs/`, from the January 30 and
June 22 buckets. The June metadata identifies an NVIDIA H100 PCIe, worker count
40, batch size 4096 and compilation enabled. Cached conditions are sensitivity
checks; the tuner starts from uncached input. Evaluation runs, GCN runs and the
`invalid/2026-06-19` bucket are excluded. Source logs are read without changes.

For every run, subtract exact logged `Wall` values at the endpoints. The reference
is step 40 to the last step (500 for arxiv, 1000 for products). Rates below use
4096 nominal batch slots per step because historical logs do not record actual
seed counts. First/second-half difference is `200*abs(a-b)/(a+b)` percent.

| Log bucket | Dataset / cache | Seconds, 40–440 | Rate difference from reference | Half difference |
| --- | --- | ---: | ---: | ---: |
| 2026-01-30 | arxiv / uncached | 13.870 | −0.436% | 0.487% |
| 2026-01-30 | arxiv / cached | 9.651 | −0.888% | 2.081% |
| 2026-01-30 | products / uncached | 88.378 | +0.132% | 0.906% |
| 2026-01-30 | products / cached | 56.023 | +2.453% | 1.690% |
| 2026-06-22 | arxiv / uncached | 13.825 | −0.528% | 0.231% |
| 2026-06-22 | arxiv / cached | 9.777 | −0.465% | 1.342% |
| 2026-06-22 | products / uncached | 86.848 | +0.706% | 0.517% |
| 2026-06-22 | products / cached | 57.690 | +1.486% | 0.057% |

The 200-step arxiv windows are only about 5–7 seconds long. Three of the four
arxiv logs exceed the 2% half-window check over 40–240 (2.13%, 3.20%, 3.65%).
Extending to 400 measured steps brings both uncached logs and the June cached
log below 2%. January cached arxiv still fails slightly, and January cached
products differs from its reference by 2.45% despite passing the half check.
These negative results remain in the evidence. Half agreement alone does not
establish agreement with a longer run.

Moving the warm-up boundary from 40 to 80 is not uniformly helpful: January
cached arxiv's 80–480 half difference is 5.31%. The evidence supports lengthening
the measurement before asserting that more warm-up solves the variability.
No minimum necessary warm-up can be determined from 20/40-step progress logs.

The 800-step confirmation length is a proposal for new trials. Arxiv logs end
at 500 steps, so it is not historically verified there. The reference and short
windows overlap and do not estimate independent-run uncertainty. Changed worker
counts, software, compilation, hardware, logging frequency or actual seed-count
accounting require new measurements. The new tuner records actual seeds/s and
keeps a 2% half-window gate; a failing repeat disqualifies that candidate instead
of discarding the inconvenient interval. No GPU speedup is claimed here.

Reproduce the table and additional windows from the GNN directory:

```bash
uv run --no-sync tools/study_pyg_windows.py \
  --artifacts ~/research/Wafer-GNN-manager/artifacts \
  --output docs/pyg_window_evidence.json
```

[pyg_window_evidence.json](pyg_window_evidence.json) records the eight relative
source paths, SHA-256 hashes, cache/provenance markers, reference rates, all
examined windows and unavailable endpoints. It can be regenerated without
copying the raw logs into this repository.
