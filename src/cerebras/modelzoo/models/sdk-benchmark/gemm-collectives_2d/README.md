# GEMM Collectives 2D

Source CSL benchmark:
`csl-extras-202412111454-4-fd53c7c9/examples/benchmarks/gemm-collectives_2d`.

The source benchmark implements SUMMA on a `P x P` PE grid. The WSE-3 command
uses `P = 4`, `Mt = 14`, `Kt = 14`, and `Nt = 14`, so the full dense GEMM is
`A[56, 56] @ B[56, 56] -> C[56, 56]` in FP32. Each SUMMA step broadcasts one
panel of `A` along rows and one panel of `B` along columns, then each PE
accumulates `C_tile += A_panel @ B_panel`.

This ModelZoo version keeps the same dimensions and fixed `P = 4` step count.
The forward path uses a static Python loop over the configured number of SUMMA
panels:

```text
for step in range(P):
    output += torch.matmul(A[:, :, k0:k1], B[:, k0:k1, :])
loss = mean((output - reference)^2)
```

The data processor emits deterministic fixed-shape FP32 tensors through
`SampleGenerator`.

Semantic mismatch: this validates the tiled GEMM computation through PyTorch
lowering. It does not model explicit collective routes, RPC launch behavior, or
per-PE column-major tile storage.
