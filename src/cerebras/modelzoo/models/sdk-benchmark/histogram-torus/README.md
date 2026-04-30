# histogram-torus

This is a ModelZoo validation benchmark inspired by the CSL
`examples/benchmarks/histogram-torus` sample.

The CSL version is primarily a routing benchmark: each PE maps local input
values to bucket-owning PEs and sends encoded wavelets over torus-style row and
column routes. PyTorch does not expose packet routes, colors, or per-PE traffic
handling. This implementation keeps only the static histogram semantics: input
values are mapped to global bucket IDs and counted with fixed-shape tensor
comparisons.

The validation graph has fixed dimensions and uses no tensor-dependent Python
control flow or eager tensor reads in `forward`.

Local smoke command after registry wiring:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/histogram-torus/configs/params_histogram-torus.yaml --target_device=CPU
```
