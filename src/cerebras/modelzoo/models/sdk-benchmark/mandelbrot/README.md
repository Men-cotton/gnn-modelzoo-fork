# Mandelbrot SDK Benchmark

This reimplements the CSL `examples/benchmarks/mandelbrot` computation as a
ModelZoo validation benchmark. The source CSL benchmark computes a 16x16 image
with a 4x4 PE pipeline and up to 32 Mandelbrot iterations.

The PyTorch version keeps the same 16x16 grid and 32-iteration cap, but it does
not model the PE pipeline or eastbound packet flow. It evaluates the whole grid
as dense tensors with fixed loop count and `torch.where` masks so the traced
graph does not depend on escape decisions.

Run locally with:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/mandelbrot/configs/params_mandelbrot.yaml --target_device=CPU
```
