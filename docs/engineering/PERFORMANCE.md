# Computational design and performance

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §18

Initial performance targets (engineering goals, not measured claims) are in [§6.9](../runtime/SIM_SEMANTICS.md).

## 18. Computational design for Apple Silicon and scalable execution

### 18.1 Optimization order

1. **Correct reference interpreter** (scalar C++, transparent tracing) and a conformance-matched differentiable Python simulator.
2. **Profile real search runs** on the Python simulator. Move the forward pass to C++ only if rollouts, rather than fitting or data loading, dominate.
3. **Batched trial evaluation** on CPU with sparse graph and SoA memory.
4. **Compiled rule specialization** removing unused AST branches and registers.
5. **Batch candidate programs of same compiled shape** when possible.
6. **Parallel search** across CPU workers/compute machines.
7. **GPU/Metal** only after profiling shows a sustained bottleneck.

For a ~188/302-neuron network, one candidate's sparse forward pass is small. Individual Metal launches may cost more than the computation. The natural GPU workload is often **thousands of independent candidate/trial trajectories**, rather than one network.

### 18.2 CPU execution

- Use CSR incoming edges with contiguous neuron states.
- Precompute edge-type masks and stable node permutations.
- Compile common rule families into specialized kernels rather than interpret AST nodes per timestep.
- In C++, template-specialize or generate a kernel per canonical program shape; keep virtual dispatch out of the timestep loop.
- SIMD/vectorize across trials/candidates when memory layout allows.
- Distinguish deterministic reduction order from faster nondeterministic parallel sums.
- Chunk data to reduce file parsing overhead during search.
- Profile parameter fitting separately from rollout execution.

### 18.3 Metal backend (optional)

- Drive Metal through metal-cpp, Apple's C++ interface to Metal, so the backend lives in the same C++ codebase without an Objective-C layer.
- A single command buffer can run multiple simulation steps **only if** intra-step dependencies and required global barriers are handled correctly; otherwise dispatch one kernel per step or a proven fused strategy.
- Batch `(candidate, animal, trial)` dimensions to amortize launch cost.
- Group candidates by shared AST shape and parameter layout.
- Keep graph adjacency compact and avoid per-candidate replication if shared.
- Use float32 initially only after passing tolerance tests against CPU float64 reference.
- Ensure gradient computations used for fitting have a separate correctness oracle.
- Report bitwise/deterministic limitations where parallel reductions differ.

**Avoid:** designing the project around proprietary CUDA. The runtime should be backend-agnostic; a Metal code generator or MLX integration is an optimization option, not the scientific premise.

### 18.4 Scale estimates and resource governance

Because the biological network is small, search explosion, not a single forward simulation, is the serious scaling hazard. Put explicit bounds on:

- Total candidate ASTs evaluated.
- Numerical optimization starts per candidate.
- CPU time and GPU time per search run.
- Intermediate trace retention.
- Number of stochastic rollouts.
- Maximum graph/field dimensions.
- Storage budget for checkpoints and report figures.

Search runs should checkpoint after each completed candidate and resume without changing evaluation order or RNG mapping. Runtime profiling metrics should be included in scientific papers as supplementary evidence.

### 18.5 Batch search scheduling

A reasonable worker queue key:

```text
(grammar_tier, canonical_program_hash, outer_fold, inner_fold, fit_seed)
```

Tasks must be idempotent. A worker may retry computation but cannot alter the frozen split or model's source. Assign independent per-task RNG keys to make results insensitive to scheduling order.
