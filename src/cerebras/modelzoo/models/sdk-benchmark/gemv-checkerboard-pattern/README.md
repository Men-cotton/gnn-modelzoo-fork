# GEMV Checkerboard Pattern

Source CSL benchmark:
`csl-extras-202412111454-4-fd53c7c9/examples/benchmarks/gemv-checkerboard-pattern`.

The source benchmark computes `y = A @ x + b` on a `4 x 4` PE rectangle with
`M = 32`, `N = 16`, FP16 values, and an `8 x 4` local matrix chunk per PE. It
streams `x` from the north edge, streams `b` from the west edge, and performs a
checkerboard chain reduction across columns.

This ModelZoo version keeps the same dense GEMV shape and dtype, but expresses
the math as a PyTorch validation step:

```text
output = torch.matmul(A, x) + b
loss = mean((output - reference)^2)
```

The data processor returns deterministic fixed-shape tensors through
`SampleGenerator`. The forward path contains no CSL runtime calls, eager tensor
reads, assertions, or Python control flow based on tensor values.

Semantic mismatch: this version validates the GEMV math lowered from PyTorch.
It does not expose the CSL benchmark's explicit checkerboard routing, streaming
colors, or per-PE chain-reduction timing.
