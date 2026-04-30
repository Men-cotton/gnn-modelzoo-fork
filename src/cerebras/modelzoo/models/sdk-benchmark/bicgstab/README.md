# BiCGSTAB SDK Benchmark

This is a ModelZoo validation benchmark for the CSL `bicgstab` example. It
keeps the 7-point-stencil matrix and the fixed-count BiCGSTAB update in
PyTorch, with epsilon-stabilized denominators instead of Python-side breakdown
branches.

The forward path has static tensor shapes and a fixed iteration count. It
returns the approximate solution, precomputed reference, maximum error, and
residual norm squared.

Run locally with:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/bicgstab/configs/params_bicgstab.yaml --target_device=CPU
```
