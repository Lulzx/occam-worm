# tests/conformance/

3–8-neuron graphs that define correct behavior. The C++ reference, the independent scalar Python implementation and the differentiable simulator must all agree.

Tickets: [OW-009](../../docs/planning/tickets/OW-009.md) · Spec: [§6.8](../../docs/runtime/SIM_SEMANTICS.md) · Case format and semantics: [WRL_SYNTAX.md](../../docs/language/WRL_SYNTAX.md) §6–§7

Status: 53 cases; the C++ reference interpreter, the scalar Python interpreter and the JAX simulator pass all of them (`pytest tests/conformance`; see [OW-009](../../docs/planning/tickets/OW-009.md)).

Each `NN-name.json` is self-contained (inline WRL source, graph, stimulus, parameters, `dt`, steps, sampled expected traces, tolerance, and a `derivation` field). Expected values are analytic solutions, exact rational arithmetic on the documented formulas, or hand-derivable dyadic arithmetic; none come from the interpreter. `gen_cases.py` regenerates every file deterministically (`python3 tests/conformance/gen_cases.py`). Run the suite with `ow sim conformance --suite tests/conformance` (also CTest `conformance_suite`).

`test_python_conformance.py` runs every case through `occamworm.sim.reference` and `occamworm.sim.jaxsim` against the expected values, checks both against `ow sim run` (max deviation 0 and 4.4e-16) and JAX against the scalar interpreter (limit 1e-12), and asserts that `ow` rejects the compile-error cases. Programs come from `ow rule inspect` (`$OW_CLI`, else `build/clang/libs/ow-cli/ow`) or, when the binary is absent, from `ir_cache/` (refresh with `OW_UPDATE_IR_CACHE=1 pytest tests/conformance`).

| Cases | Covers |
|---|---|
| 01–03 | exact and Euler leaky integration, decay, constant and zero stimuli |
| 04–06, 38–39 | chemical excitation, inhibition, signed sums, multigraph edges, self-loops |
| 07–13, 23, 37 | gap-junction diffusion, constant-voltage equilibrium, large-step boundedness, edge deletion, multigraph junctions, dt convergence |
| 14–16 | delay ring wraparound, history initialisation, own-register delay |
| 17–18 | adaptation register updates |
| 19–20 | calcium filter impulse and step responses, sparse sampling |
| 21–22, 33, 40 | stimulus variants, chemical edge deletion, zero steps, no edges |
| 24–27, 35 | G0 truth tables (rule 90 ring, own-state table, lut clamping, count and delay) |
| 28–32 | type masks and observed masks, operator semantics, tanh and sigmoid, parameter override, dead-code names |
| 34–35 | permutation equivariance (G1 mixed circuit, G0 ring) |
| 36–37 | dt versus dt/2 convergence against analytic solutions |
| 41–52 | compile errors: units, stability, syntax, tier, name, type and limit |
| 53 | modulatory edges: `sum_in(v, mod)` apart from chemical edges, repeated edges and self-loops |
