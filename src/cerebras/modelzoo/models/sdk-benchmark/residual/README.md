# residual

ModelZoo validation benchmark for the CSL `residual` example.

The PyTorch graph computes `residual = b - A @ x` and returns the infinity norm
of that residual. The default dimensions match the CSL README's even matrix
shape, `M = 6` and `N = 4`, but the computation is expressed as a single fixed
dense `matmul` plus tensor reductions.

This does not model the CSL 2-by-2 PE partitioning, row reduction, column
reduction, or memcpy halo. It validates the same mathematical result through
the normal ModelZoo validation path.
