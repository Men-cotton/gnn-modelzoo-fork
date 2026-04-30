# bandwidth-test

This is a ModelZoo validation benchmark inspired by the CSL
`examples/benchmarks/bandwidth-test` sample.

The CSL version measures explicit host-to-device or device-to-host transfer
bandwidth with PE-side timestamp counters, I/O channels, buffering, and
blocking/nonblocking runtime behavior. PyTorch does not expose those transfer
paths or PE timers, so this implementation is only a fixed-shape tensor
traffic analogue. It repeatedly applies a deterministic elementwise transform to
a payload tensor and compares the result with a precomputed reference.

The validation graph is static: fixed tensor length, fixed loop count, no
tensor-dependent Python control flow, and no eager tensor reads in `forward`.

Local smoke command after registry wiring:

```bash
uv run cszoo validate src/cerebras/modelzoo/models/sdk-benchmark/bandwidth-test/configs/params_bandwidth-test.yaml --target_device=CPU
```
