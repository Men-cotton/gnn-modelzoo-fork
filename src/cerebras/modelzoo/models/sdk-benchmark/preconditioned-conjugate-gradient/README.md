# Preconditioned Conjugate Gradient SDK Benchmark

This is a ModelZoo validation benchmark for the CSL
`preconditioned-conjugate-gradient` example. It keeps the 7-point-stencil matrix
and Jacobi preconditioner, with the diagonal coefficient matching the source
benchmark's `6 + column` center value.

The PyTorch forward path uses static shapes, fixed unrolled iterations, and no
residual-driven Python loop. It returns the fixed-iteration solution,
precomputed reference, maximum error, and residual norm squared.

Run locally with:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/preconditioned-conjugate-gradient/configs/params_preconditioned_conjugate_gradient.yaml --target_device=CPU
```
