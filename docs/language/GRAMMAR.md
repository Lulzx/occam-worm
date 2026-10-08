# WRL: the Worm Rule Language

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §5

## 5. The Rule Language (`WRL`)

### 5.1 Purpose

`WRL` (Worm Rule Language) is a small, typed, deterministic-by-default domain-specific language for update rules on biological multigraphs. It must support exhaustive short-program enumeration, human inspection, symbolic rewrites, structured mutation, parameter inference, serialization, and compilation.

**Design preference:** restrict the grammar enough that searching it is scientifically interpretable and computationally tractable. Do not turn it into a general unrestricted neural network language.

### 5.2 Types

```text
Scalar        finite float with unit annotation
Bool          Boolean predicate
IntSmall      bounded integer state
State<K>      K-valued internal local state
NeuronId      canonical atom, only in metadata
NeuronType    categorical anatomical/functional class
EdgeType      Chemical | Gap | Modulatory | NMJ
Signal        state or message at a declared time
Duration      quantity with time unit
Count         nonnegative multiplicity
Vector<N>     small fixed-size vector, optional after v0.1
```

Every operation checks unit compatibility. Distinguish parameter values in physical units from arbitrary dimensionless activation.

### 5.3 Core expressions

| Family | Operations | Notes |
|---|---|---|
| Constants/inputs | `const`, `stimulus`, `state`, `edge_weight`, `type_mask` | Explicit permitted observables |
| Arithmetic | `add`, `mul`, `neg`, `clamp`, `abs`, `min`, `max` | No unbounded division by unknown |
| Nonlinearities | `relu`, `tanh`, `sigmoid`, `threshold`, `piecewise` | Each has a complexity cost |
| Neighborhood | `sum_in`, `mean_in`, `sum_gap`, `sum_mod` | Typed, static topological neighborhoods |
| Temporal | `delay`, `leaky_integrate`, `decay`, `hold`, `rise` | State allocation declared explicitly |
| Branching | `if`, `select` | Condition dependency must be local |
| Stochastic | `normal_noise`, `bernoulli` | Only in stochastic grammar tier |
| Observation | `calcium_filter`, `fluorescence`, `readout` | Measurement model versioned separately |

Disallow introspection of hidden test data, arbitrary graph-global broadcast except explicit declared signals, unrestricted recursion, dynamic code generation, and dependence on raw neuron identifiers as lookup-table keys in the smallest grammar tier.

### 5.4 Grammar tiers

- **G0 — Truth-table cellular automata:** bounded discrete states, neighbor state counts, deterministic transition table. Useful for synthetic tests and emergence maps; not presumed biologically realistic.
- **G1 — Stable continuous local rules:** leaky integration, thresholds, saturation, sign-constrained edges, optional delay.
- **G2 — Local memory:** one or two adaptation/synaptic state registers, bounded delays.
- **G3 — Type-conditioned rules:** rule selection by neurotransmitter/receptor/cell class, without individual ID memorization.
- **G4 — Modulatory fields:** sparse receptor-dependent spatial message fields and low-dimensional slow state.
- **G5 — Structured stochastic rules:** process noise and between-animal parameter distributions.

**Promotion rule:** later tiers are unlocked only when earlier tiers leave reproducible residual structure and the extra complexity can be evaluated with the available data.

### 5.5 Illustrative WRL program

This is **illustrative syntax**, not a working parser nor a calibrated biophysical equation:

```yaml
wrl_version: "0.1"
rule_id: leak_exc_inh_adapt_v1
neuron_state:
  v: {unit: dimensionless, init: 0.0}
  h: {unit: dimensionless, init: 0.0}
inputs:
  stimulus: {unit: dimensionless}
  chemical_exc: {op: sum_in, edge: chemical_exc, signal: v_delayed}
  chemical_inh: {op: sum_in, edge: chemical_inh, signal: v_delayed}
parameters:
  tau_v: {unit: s, value: 0.25, trainable: true, lower: 0.005, upper: 10.0}
  tau_h: {unit: s, value: 2.0, trainable: true, lower: 0.01, upper: 30.0}
  gain: {unit: dimensionless, value: 1.0, trainable: true, lower: 0.0, upper: 20.0}
  adapt: {unit: dimensionless, value: 0.2, trainable: true, lower: 0.0, upper: 10.0}
update:
  drive: "gain * (chemical_exc - chemical_inh + stimulus) - adapt * h"
  v_next: "v + dt/tau_v * (-v + tanh(drive))"
  h_next: "h + dt/tau_h * (-h + max(v, 0))"
observation:
  operator: calcium_linear_v1
```

**Compiler must reject** this explicit-Euler update when a proposed `dt/tau` exceeds its declared stability constraint, or replace it with a separately specified stable integration primitive whose semantics are documented. Inputs `chemical_exc` and `chemical_inh` require biological priors or latent edge-sign inference; they are not silently derived from anatomical synapse counts.

### 5.6 Canonical program representation

1. Parse into a typed abstract syntax tree (AST).
2. Lower to SSA-like intermediate representation with explicit state writes.
3. Normalize commutative operators and constant folding.
4. Eliminate dead registers and algebraically redundant branches.
5. Alpha-rename bound variables and perform common-subexpression elimination.
6. Assign deterministic `program_hash = SHA256(canonical_IR)`.
7. Record the grammar version and compiler build in every result.

**Distinct syntax with equivalent tested behavior** may be grouped after evaluation, but do not claim formal semantic equivalence without an actual proof.

### 5.7 Complexity accounting

Define `L_struct` from a prefix-free encoding of AST operators, argument references, register declarations, fixed typed constants and rule-dispatch tables. Define `L_params` from the trainable parameters at their declared precision; precision must be charged, not merely parameter count:

$$
L_{struct}=L_{AST}+L_{topology\ overrides}+L_{type\ dispatch},
\qquad
L_{total}=L_{struct}+L_{params}.
$$

`L_struct` depends only on the program; `L_params` depends on the fitted values and their declared precision. The objective ([§3.5](../science/PROBLEM_STATEMENT.md)), the program prior ([§7.7](../inference/SEARCH_AND_INFERENCE.md)) and the Pareto front ([§7.9](../inference/SEARCH_AND_INFERENCE.md)) all use these same quantities, and nothing else charges for complexity. Store the actual bit accounting in each run. A 3-line rule with 100,000 per-pair learned values is **not** a low-complexity model.

### 5.8 Biological constraints

Hard constraints for selected tiers:

- Nonnegative conductance magnitudes for electrical junctions.
- Physical units and positive time constants.
- Zero contribution from truly absent chemical edges unless an allowed extrasynaptic pathway is declared.
- Symmetric gap-junction coupling in the default model, with exceptions explicitly supported by evidence.
- Bounded activity or mathematically stable dynamics over specified test ranges.
- Explicit simulator timing and finite event delays.

Soft priors may incorporate transmitter phenotype, receptor expression, known functional signs and regional information, **with provenance**. Priors must not leak an outcome label from the held-out perturbation.
