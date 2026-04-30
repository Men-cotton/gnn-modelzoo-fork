# Conjugate Gradient SDK Benchmark

This is a ModelZoo validation benchmark for the CSL `conjugate-gradient`
example. It reimplements the 7-point-stencil sparse matrix and fixed-count CG
iteration in PyTorch instead of invoking CSL kernels.

The traced forward path uses static tensor shapes, fixed unrolled iterations,
and tensor reductions only. It returns the fixed-iteration solution, a
precomputed PyTorch reference, the maximum output error, and the final residual
norm squared.

Run locally with:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/conjugate-gradient/configs/params_conjugate_gradient.yaml --target_device=CPU
```
