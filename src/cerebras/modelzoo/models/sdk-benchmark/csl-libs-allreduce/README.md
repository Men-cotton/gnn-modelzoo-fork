# csl-libs allreduce

This ModelZoo benchmark is a PyTorch analogue of the CSL `csl-libs/allreduce`
library example. The direct CSL version performs row reduction, column
reduction, and broadcast over a 2D PE rectangle for ADD or MAX reductions. This
version expresses that behavior as a static tensor reduction over the grid
dimensions followed by broadcasting the result back to every logical PE slot.

It does not import or launch the CSL library. It is intended to exercise the
normal ModelZoo validation path and Cerebras PyTorch lowering.

The default config uses a deterministic `4 x 4` logical PE grid with vector
length `16`, ADD reduction, and ten validation steps. The model returns `loss`,
`output`, `reference`, and `max_error`.

CPU smoke:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/csl-libs-allreduce/configs/params_csl-libs-allreduce.yaml --target_device=CPU
```
