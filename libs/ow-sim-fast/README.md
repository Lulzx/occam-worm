# `ow-sim-fast`

Vectorized CPU runtime with per-shape specialized kernels. Built **only** if profiling shows Python rollouts limit search.

**Invariant:** Matches `ow-sim` within declared tolerance; no virtual dispatch in the timestep loop.

Spec: [§18.1](../../docs/engineering/PERFORMANCE.md), [§18.2](../../docs/engineering/PERFORMANCE.md)

Status: not implemented.
