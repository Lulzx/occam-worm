# Deterministic simulation runtime

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §6

## 6. Deterministic simulation runtime

### 6.1 Execution semantics

**Default:** synchronous fixed-timestep transitions, with a stable and fully specified order:

1. Apply exogenous stimulation and any declared sensory input at time `t`.
2. Read snapshot of all old neuronal states and delay buffers.
3. Compute chemical message aggregation on the old states.
4. Compute electrical coupling on a separately specified, stable discretization.
5. Update local state registers, using no partially updated neighbor state.
6. Evolve calcium/observation states.
7. Write the new state arrays and delay buffers.
8. Sample observables exactly at configured acquisition timestamps.

This avoids source-order dependence and makes the update rule reproducible. Future event-driven/asynchronous semantics must be a new execution mode, not an undocumented optimization of the synchronous one.

### 6.2 Array representation

For `N` neurons and `E` anatomical edges:

```text
neurons:
  id[N]              canonical ID index
  type[N]            compact categorical code
  state_0[N]         contiguous float32/float64
  state_1[N]         contiguous optional register
  obs_state[N]       separate calcium filter state

chemical graph:
  in_offsets[N+1]    CSR rows, postsynaptic target
  pre_id[E]          presynaptic index
  weight[E]          synaptic magnitude / latent weight
  edge_type[E]       transmitter/sign class, if supported
  delay_bucket[E]    integer delay index

gap graph:
  symmetric edge list, with separate coupling weights

modulatory graph:
  receptor/ligand-compatible candidate edges or spatial field model

simulation:
  delay_ring[D][N]   only if enabled
  stimulus[T][N]     sparse target representation preferred
  outputs[T_sampled][N_observed]
```

The canonical CSR orientation must be documented, tested and versioned; many neural simulation errors come from inadvertently reversing the source and destination indices.

### 6.3 Gap junction integration

Do **not** treat electrical connections as two arbitrary directed chemical edges. A standard electrical-coupling term is

$$
q^{gap}_i=\sum_j g_{ij}(v_j-v_i).
$$

The integration method should account for stiffness (e.g., stable semi-implicit solve for a linear coupling block), preserve constant-voltage equilibrium in the absence of other terms, and pass energy/dissipation sanity checks. The exact physical interpretation depends on the chosen state variable.

### 6.4 Delay semantics

- Quantize `d_ij` to simulation ticks in the initial runtime.
- Quantization error bounded and reported as a function of `dt`.
- No negative delays; zero-delay means old-timestep state in synchronous mode.
- Delayed signals draw from initialized history buffers; the history initialization policy is declared.
- Future continuous-delay interpolation is an explicit feature with numerical equivalence tests.

### 6.5 Stability and boundedness

For every program before fitting:

- Check divisions/parameter domains and reject NaN-producing programs.
- Run random and adversarial input tests for a configured temporal horizon.
- Reject or penalize unbounded state growth not physically justified.
- Check timestep convergence by comparing `dt` and `dt/2`; tolerance depends on observable and model family.
- Distinguish dynamical instability (potentially meaningful) from numerical instability (simulation bug).
- Keep a simulator diagnostics vector, not only an aggregate model fitness.

### 6.6 Randomness, determinism and reproducibility

By default the candidate is deterministic. For stochastic programs:

- Use counter-based or otherwise reproducible splittable RNG streams.
- Seed by `(experiment_id, program_hash, animal_id, trial_id, replicate_id)`.
- Record the PRNG algorithm and seed derivation.
- Implement sampling distributions explicitly (for example, normal variates by a documented transform). C++ standard-library distributions such as `std::normal_distribution` are implementation-defined and produce different sequences on different standard libraries, which breaks cross-platform replay.
- Use common random numbers when comparing candidate predictions where appropriate.
- Report repeated Monte Carlo predictive uncertainty rather than one lucky rollout.
- Do not confuse parameter uncertainty with intrinsic dynamical process noise.

### 6.7 Initial conditions

Initial state can materially change long-term neural response. Define three policies:

1. **Steady-state:** solve or burn in under measured baseline stimulus.
2. **Measured-calibration:** infer a constrained initial-state posterior from a fixed pre-stimulation window.
3. **Hierarchical:** draw initial states from a distribution learned on training animals.
4. **History-conditioned:** for sequential stimulations within one recording, carry the simulated state forward from the previous trial across the recorded inter-stimulus interval, or use the declared reduced history term of [§4.1](../science/BASELINES.md).

Policies 1–3 treat trials as independent exposures and are acceptable only if the M0 audit finds no material order or carryover effects. Otherwise policy 4 is the default.

The outer-test policy must not optimize arbitrary initial states against post-stimulation responder traces. Report sensitivity to initial-state assumptions.

### 6.8 Runtime correctness oracle

For a small graph (e.g., 3–8 neurons), implement a scalar reference interpreter in C++ and an independent scalar implementation in Python. The differentiable Python simulator ([§7.6](../inference/SEARCH_AND_INFERENCE.md)) and every optimized or vectorized backend must match both to tolerance on:

- Chemical excitation/inhibition.
- Gap junction diffusion.
- Delay ring wraparound.
- Adaptation/register updates.
- Calcium filtering and sampling.
- Constant stimuli, impulses, zeros and edge deletions.
- Discrete rule truth tables.
- Mixed-neuron identities and masks.

### 6.9 Performance targets (engineering goals, not measured claims)

| Workload | Initial target | Why |
|---|---|---|
| Small synthetic graphs (<16 nodes) | Exact/near-exact reference correctness | Establish trustworthy semantics |
| Head circuit (~188 neurons) | Fast enough for thousands of fit/evaluate cycles per workstation session | Most early candidate searches are small |
| Adult hermaphrodite graph (~302 neurons) | Substantially faster than real time for simple deterministic rules | Enabling search throughput; report hardware/context |
| Candidate batches | Vectorized execution where profiles justify it | Program search is many independent short simulations |
| Replay | Bitwise deterministic CPU in pinned builds where practical | Audit and reproducibility |

Benchmark on the intended Apple Silicon machine using **actual measured** throughput; do not promise fixed simulation rates, acceleration ratios or memory use before implementation. The limiting cost may be fitting parameters or reading trial data, not the tiny nervous-system forward pass.

### Implemented (OW-009)

The C++ scalar reference interpreter (`libs/ow-sim`, `ow sim run`) implements §6.1-§6.4 and §6.7-§6.8 for tiers G0 and G1 as follows. The exact formulas, JSON formats and conformance case format are in [WRL_SYNTAX.md](../language/WRL_SYNTAX.md) §5-§7; this note only points at the decisions that §6 left open.

| Topic | Decision | Where |
|---|---|---|
| Step order (§6.1) | Stimulus, snapshot, chemical sums, gap sums, local update, observation, commit, sampling, exactly as listed; every read sees only tick-`t` state and history | WRL_SYNTAX §6.3 |
| CSR orientation (§6.2) | Chemical edges are stored by postsynaptic row (row `i` lists the inputs of neuron `i`), rows sorted by (pre, delay, sign, weight); gap junctions stored in both endpoint rows; orientation version 1 | `ow-core/graph.hpp`, `tests/cpp/test_core.cpp` |
| Gap junctions (§6.3) | Node-local semi-implicit step `x_i' = (w_i + dt * sum_j c_ij x_j) / (1 + dt * sum_j c_ij)` with neighbours from the old state: unconditionally stable, preserves a constant-voltage equilibrium, first-order accurate, does not conserve the register total | WRL_SYNTAX §6.6 |
| Delays (§6.4) | Integer ticks; `delay = 0` reads the old state; history before tick 0 equals the initial state (constant extension, includes overrides); ring of `D+1` slots per register | WRL_SYNTAX §6.5 |
| Observation (§6.1 step 6) | `identity_v1` or `calcium_linear_v1`: exact first-order filter driven by the post-update register value, initial filter state equal to the register's initial value | WRL_SYNTAX §6.7 |
| Sampling (§6.1 step 8) | Sample tick `k` is the state after `k` steps (`k = 0` is the initial state); physical time `k * dt` | WRL_SYNTAX §6.3 |
| Stability (§6.5) | `leaky_integrate` is exact for constant targets; `euler_leak` needs `dt_max / tau_lower <= 1`, checked at compile time and against `dt` at run time; non-finite values abort the run with `E_RUNTIME` | WRL_SYNTAX §3, §6.9 |
| Randomness (§6.6) | G0/G1 are deterministic, so the interpreter has no RNG; the Philox plan for G5 is written down | WRL_SYNTAX §6.10 |
| Oracle (§6.8) | `tests/conformance/*.json`, run by `ow sim conformance` and CTest; expected values are analytic or hand-derived | WRL_SYNTAX §7 |

**Python implementations (OW-009, Python half).** `python/occamworm/sim/` holds the two implementations required by §6.8, kept separate and both driven by the IR JSON of `ow rule inspect` (the Python side never parses WRL): `reference.py`, a plain scalar interpreter (floats, `math`, loops, full history list instead of a ring), and `jaxsim.py`, the differentiable float64 simulator (`lax.scan`, history ring, edge-list gathers, `segment_sum`, `vmap` over trials, stimuli, initial states and parameter vectors). All 53 conformance cases pass in both; the scalar interpreter reproduces `ow sim run` bit for bit on the suite and JAX agrees with C++ and with the scalar interpreter to 4.4e-16 (tests in `tests/conformance/` and `tests/unit/sim/`, also on random graphs). Gradients with respect to parameters, stimulus and initial state match central finite differences (relative error below 1e-6 in the tests). Measured on an Apple M4 Pro CPU for a random 302-neuron graph with 7,000 chemical edges (delays 0-2) and 1,500 gap junctions, rule `leak-adapt.wrl`, 2,000 steps: JAX forward about 8,900 steps/s (observation only about 7,700), forward plus gradient about 5,900 steps/s, a batch of 8 trials about 31,000 trial-steps/s; the scalar Python interpreter about 75 steps/s; `ow sim run` about 7,800 steps/s including process start and JSON. These are single measurements, not performance claims (§6.9).

Still open from §6: steady-state and measured-calibration initial conditions (§6.7 policies 1-3; policy 4, history-conditioned, is available through `initial_state`), the dt-versus-dt/2 gate as an automatic filter on candidate programs, and continuous-delay interpolation.
