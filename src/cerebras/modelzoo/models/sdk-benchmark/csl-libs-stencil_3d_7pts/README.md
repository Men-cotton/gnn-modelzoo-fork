# csl-libs stencil_3d_7pts

This ModelZoo benchmark is a PyTorch analogue of the CSL
`csl-libs/stencil_3d_7pts` library example. The direct CSL version exchanges
planes with west, east, south, and north neighboring PEs and applies a
7-point Laplacian along the local z vector. This version expresses the same
fixed-shape stencil as tensor padding, slicing, and elementwise arithmetic.

It does not import or launch the CSL library. It is intended to exercise the
normal ModelZoo validation path and Cerebras PyTorch lowering.

The default config uses a deterministic `8 x 8 x 8` volume, coefficients
`[-1, -1, -1, -1, -1, -1, 6]`, zero boundary values, and ten validation steps.
The model returns `loss`, `output`, `reference`, and `max_error`.

CPU smoke:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/csl-libs-stencil_3d_7pts/configs/params_csl-libs-stencil_3d_7pts.yaml --target_device=CPU
```
