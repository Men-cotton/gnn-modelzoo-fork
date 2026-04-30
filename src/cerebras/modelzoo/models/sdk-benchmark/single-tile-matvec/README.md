# Single Tile Matvec

Source CSL benchmark:
`csl-extras-202412111454-4-fd53c7c9/examples/benchmarks/single-tile-matvec`.

The source benchmark runs independent FP32 matvecs on every PE in a program
rectangle. The WSE-3 command uses `width = 2`, `height = 2`, `tile_size = 25`,
and `iters = 1`. Each PE owns one `25 x 25` matrix and one length-25 vector.
The CSL program reports per-PE cycle counts, memory-bandwidth estimates, and
FLOP rates.

This ModelZoo version represents the PE rectangle as leading tensor dimensions:

```text
A: [batch, height, width, tile_size, tile_size]
x: [batch, height, width, tile_size]
output = A @ x
```

The configured iteration count is a fixed Python loop bound. With the default
`iters = 1`, the output matches a single source verification run. The data
processor emits deterministic fixed-shape FP32 tensors through
`SampleGenerator`.

Semantic mismatch: this validates the per-tile matvec math through PyTorch
lowering. It does not reproduce the CSL timestamp collection, padded local
memory layout, or bandwidth accounting.
