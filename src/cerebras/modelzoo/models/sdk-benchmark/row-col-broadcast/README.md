# row-col-broadcast

This is a ModelZoo validation benchmark inspired by the CSL
`examples/benchmarks/row-col-broadcast` sample.

The CSL version uses specialized H2D row/column broadcast memcpy APIs to reduce
host bandwidth. PyTorch cannot request those memcpy modes, so this benchmark is
only a static tensor analogue of the broadcasted layout. The model receives a
single row or column source tensor, expands it to a fixed `height x width`
region, and compares that result with a deterministic reference tensor.

The validation graph is static and uses only tensor operations in `forward`.

Local smoke command after registry wiring:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/row-col-broadcast/configs/params_row-col-broadcast.yaml --target_device=CPU
```
