# 7pt-stencil-spmv

ModelZoo validation benchmark for the CSL `7pt-stencil-spmv` example.

The PyTorch graph computes one fixed-shape 3D seven-point stencil SpMV over a
`height x width x z_dim` tensor. It uses deterministic synthetic input and the
same coefficient order as the CSL example: west, east, south, north, bottom,
top, and center. Boundary values are treated as zero.

This is a tensor analogue of the CSL kernel, not a wrapper around `cslc`,
`SdkRuntime`, or the PE-local routing code. The explicit neighbor exchange,
clock synchronization, and bandwidth timing paths from CSL are not represented;
ModelZoo validation reports the normal PyTorch-lowered execution path.
