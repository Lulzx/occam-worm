# benches/

Performance benchmarks measured on the target Apple Silicon machine. Targets are set from measurements, not promised in advance.

- `forward_runtime/` — Forward-pass throughput on the head circuit (~188 neurons) and full graph (~302).
- `candidate_throughput/` — End-to-end candidates evaluated per hour under a real search configuration.
- `fitting/` — Parameter-fitting cost, profiled separately from rollouts.

Spec: [§6.9](../docs/runtime/SIM_SEMANTICS.md), [§18](../docs/engineering/PERFORMANCE.md)

Status: not implemented.
