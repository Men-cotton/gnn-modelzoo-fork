# Cholesky SDK Benchmark

This reimplements the CSL `examples/benchmarks/cholesky` computation as a
ModelZoo validation benchmark. The source WSE3 script uses `P=10,Nt=4`, giving
a dense matrix size of `N=40`, and distributes the lower triangle over a tiled
PE grid.

The PyTorch version builds the same sized deterministic positive-definite
matrix and calls `torch.linalg.cholesky`. It does not model triangular PE
tiling, route reconfiguration, or the right-looking CSL communication schedule.
Those differences should be kept in mind when comparing performance.

Run locally with:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/cholesky/configs/params_cholesky.yaml --target_device=CPU
```
