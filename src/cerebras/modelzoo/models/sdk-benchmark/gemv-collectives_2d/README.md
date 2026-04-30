# GEMV Collectives 2D

Source CSL benchmark:
`csl-extras-202412111454-4-fd53c7c9/examples/benchmarks/gemv-collectives_2d`.

The source benchmark computes `y = A @ x + b` in FP32. Its default command uses
a `4 x 4` PE rectangle, `M = 32`, and `N = 16`; each PE owns an `8 x 4` tile of
`A`. The northwestern PE starts with full `x` and `b`, then the collectives
library scatters, broadcasts, reduces, and gathers the result.

This ModelZoo version keeps the same dense GEMV shape and FP32 dtype:

```text
output = torch.matmul(A, x) + b
loss = mean((output - reference)^2)
```

The data processor emits deterministic fixed-shape tensors via
`SampleGenerator`. The forward path is a static tensor graph with no CSL
runtime calls and no eager reads of traced tensor values.

Semantic mismatch: PyTorch does not expose the explicit `collectives_2d`
scatter, broadcast, row reduction, or final gather. This benchmark is therefore
the math-equivalent validation path, not a communication-topology clone.
