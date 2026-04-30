# Power Method SDK Benchmark

This is a ModelZoo validation benchmark for the CSL `power-method` example. It
implements the 7-point-stencil sparse matrix and fixed-count normalization loop
as PyTorch tensor operations.

The forward path uses static shapes and a fixed iteration count. It returns the
approximate dominant eigenvector, precomputed reference, maximum error, and the
Rayleigh quotient after the final iteration.

Run locally with:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/power-method/configs/params_power_method.yaml --target_device=CPU
```
