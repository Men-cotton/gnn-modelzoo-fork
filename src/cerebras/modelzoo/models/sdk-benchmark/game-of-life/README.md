# game-of-life

This is a ModelZoo validation benchmark inspired by the CSL
`examples/benchmarks/game-of-life` sample.

The CSL version maps one cell to each PE and exchanges neighbor state through
fabric links, including forwarded diagonal state. PyTorch cannot expose that
PE-local communication pattern. This implementation keeps the static Conway
update rule with zero boundary conditions and a fixed generation count, using
only tensor slices and `torch.where`.

The validation graph has fixed dimensions, fixed iteration count, and no eager
tensor reads in `forward`.

Local smoke command after registry wiring:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/game-of-life/configs/params_game-of-life.yaml --target_device=CPU
```
