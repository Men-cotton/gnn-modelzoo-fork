# SDK benchmark analogues

This directory contains ModelZoo validation models inspired by selected
`csl-extras` SDK benchmarks. They are PyTorch tensor analogues, not direct
wrappers around CSL kernels or exact models of PE routing, memcpy modes, colors,
or timestamp timing.

Use the trace-external correctness gate before treating a benchmark config as a
working analogue:

```bash
uv run python scripts/verify_sdk_benchmark_correctness.py
```

The gate loads every `sdk_benchmark/*` registry entry, builds its configured data
processor, runs CPU `forward` under `torch.no_grad()`, and asserts that:

- `forward` returns a dict with `loss`;
- all floating tensor outputs are finite;
- `loss` and returned error metrics stay within tolerance;
- the returned `output`/`reference` pair, or FFT real/imag pairs, agree within
  tolerance.

Solver residual metrics are checked for finiteness but are not required to be
zero, because the solver analogues intentionally use fixed iteration counts.
