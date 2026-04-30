# spmv-hypersparse

ModelZoo validation benchmark for the CSL `spmv-hypersparse` example.

The PyTorch graph uses a deterministic masked dense matrix and computes
`y = A @ x` with fixed shapes. The mask creates a hypersparse pattern while
keeping the operation in tensor form for the ModelZoo validation path.

This cannot mirror the CSL benchmark's Matrix Market ingestion, 2D PE grid
partitioning, local nonzero buffers, sparse routing protocol, or load-balancing
preprocess step. It is intended to measure the PyTorch-lowered mathematical
analogue, not the explicit sparse communication kernel.
