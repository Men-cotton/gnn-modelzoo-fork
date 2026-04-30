# Wide Multiplication SDK Benchmark

This reimplements the CSL `examples/benchmarks/wide-multiplication` computation
as a ModelZoo validation benchmark. The source benchmark multiplies two 128-bit
unsigned integers represented as 16 little-endian 16-bit words and returns a
256-bit result.

The PyTorch version represents the operands as binary float tensors and uses
fixed-size tensor operations plus a fixed 256-step carry propagation loop. It
does not model the single-PE streaming H2D/D2H protocol from the CSL version.

Run locally with:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/wide-multiplication/configs/params_wide_multiplication.yaml --target_device=CPU
```
