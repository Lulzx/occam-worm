# `occamworm.sim`

Two independent Python implementations of the WRL execution semantics ([WRL_SYNTAX.md](../../../docs/language/WRL_SYNTAX.md) §5-§7), next to the C++ interpreter (`libs/ow-sim`).

**Invariant:** passes the same conformance suite as the C++ reference interpreter; every implementation consumes the canonical IR JSON that `ow rule inspect` prints, so none of them parses WRL.

Tickets: [OW-009](../../../docs/planning/tickets/OW-009.md) · Spec: [§6.8](../../../docs/runtime/SIM_SEMANTICS.md), [§7.6](../../../docs/inference/SEARCH_AND_INFERENCE.md)

Status: implemented (OW-009). 52/52 conformance cases pass in both implementations; the scalar interpreter is bit-identical to C++ on the suite and JAX agrees to 4.4e-16.

| Module | Role |
|---|---|
| `ir.py` | Loads and validates IR JSON (`load_ir`) into a frozen `Program`; obtains it from the `ow` binary (`compile_source`, `compile_file`, `inspect_source`). The binary is `$OW_CLI`, else `build/clang/libs/ow-cli/ow` under the repo root; a missing binary raises `CompilerNotFoundError` naming both. `compile_source(..., cache_dir=...)` falls back to a cached `<sha256(source)>.json` only when the binary is absent. Compiler rejections raise `CompilerError` with the stable code (`E_UNIT`, ...). |
| `graph.py` | `GraphSpec` / `Graph` (CSR by postsynaptic row, orientation version 1, same validation as the C++ `Graph::build`), `StimulusEvent`, `SimInput`, edge-deletion edits, `sim_input_from_json`, `permute_neurons`. |
| `cases.py` | Conformance case loader (`load_case`, `load_suite`, `Tolerance`). |
| `reference.py` | **Scalar interpreter**: plain Python floats and `math`, explicit loops, full-history list instead of a ring. Same reference summation order as C++. Slow by design (about 75 to 300 steps/s on a 302-neuron graph); it is the second oracle, not a production backend. |
| `jaxsim.py` | **Differentiable simulator**: float64, `jax.lax.scan` over ticks, a `(R, D+1, N)` history ring, edge-list gathers and `segment_sum` per neuron, node-local semi-implicit gap step. Batches of stimuli, initial states and parameter vectors through `vmap`; `jax.grad` with respect to `theta`, the dense stimulus and the initial state. |
| `result.py`, `cpp.py` | `SimResult`, deviation helpers, and `simulate_cpp` (runs `ow sim run`). |

```python
from occamworm.sim import jaxsim
from occamworm.sim.ir import compile_file
from occamworm.sim.graph import build_graph

program = compile_file("configs/rules/leak-adapt.wrl")  # IR from the ow binary
graph = build_graph(spec)  # GraphSpec -> CSR Graph
sim = jaxsim.JaxSimulator(program, graph, dt=0.1, n_steps=200, observed=[0, 5], sample_ticks=range(0, 201, 5))
stim = jaxsim.stimulus_array(events, graph, 200)  # (T, N); stack to (B, T, N) for a batch
obs = sim.observe(program.default_theta(), stim)  # (S, K); sim.run also returns all registers
grad = jax.grad(lambda th: (sim.observe(th, stim) ** 2).sum())(program.default_theta())
```

`theta` is the full parameter vector in IR order (`program.parameters`), so fixed parameters take part in the same array and have zero influence on the transforms of `occamworm.fit`. `jaxsim.simulate(program, sim_input)` and `reference.simulate(program, sim_input)` take the §6.1 input and return the same `SimResult` as `ow sim run`.

Semantics notes. XLA CPU scatter-add accumulates sequentially, so `segment_sum` over the edge list keeps the CSR row order and the chemical and gap sums match the reference bit for bit on dyadic cases; `exp` and `tanh` come from different libm builds (JAX/XLA vs system), hence the 1 ulp class deviations. JAX cannot raise `E_RUNTIME` inside a jitted scan: non-finite values propagate as `nan`/`inf` and `jaxsim.simulate` raises `SimulationError` afterwards. Non-differentiable operators (`threshold`, `select`, `lut`, ties of `min`/`max`, `relu` at 0) have the gradient of the selected branch.

Tests: `tests/conformance/test_python_conformance.py` (the 52 cases, both implementations, agreement with `ow sim run`), `tests/unit/sim/` (loaders, random-graph agreement across all three implementations, gradients against finite differences, batching).
