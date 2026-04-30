# FFT SDK Benchmark

This reimplements the CSL `examples/benchmarks/FFT` benchmark as a ModelZoo
validation benchmark. The source CSL example implements iterative radix-2
Cooley-Tukey FFT for 1D and 2D inputs; its WSE3 script runs `DIM=1,Nz=4,FP=2`
and `DIM=2,Nz=4,FP=1`.

The PyTorch version uses a dense DFT matrix multiply with static shape instead
of modeling the CSL butterfly schedule, PE-local layout, or transpose traffic.
This preserves the forward/inverse DFT semantics for small static sizes while
using ordinary tensor operations suitable for validation-path lowering.

Run locally with:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/FFT/configs/params_FFT.yaml --target_device=CPU
```
