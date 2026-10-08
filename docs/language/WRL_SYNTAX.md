# WRL v0.1: concrete syntax, canonical IR and execution semantics

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Companion to [GRAMMAR.md](GRAMMAR.md) (§5) and [SIM_SEMANTICS.md](../runtime/SIM_SEMANTICS.md) (§6) · Tickets [OW-008](../planning/tickets/OW-008.md), [OW-009](../planning/tickets/OW-009.md), [OW-010](../planning/tickets/OW-010.md)

This document is the **contract** between the C++ reference implementation (`libs/ow-ir`, `libs/ow-sim`, `libs/ow-search`) and any other implementation, in particular the differentiable Python simulator of OW-009. The YAML in [§5.5](GRAMMAR.md) stays illustrative; the syntax below is the one the compiler accepts. A second implementation needs only §5 (IR JSON), §6 (execution semantics) and §7 (conformance cases); it consumes the IR JSON printed by `ow rule inspect` and does not need to re-implement parsing or canonicalisation.

Everything is `float64`. Grammar version: `0.1`. IR schema: `occamworm.wrl.ir/0.1`. Simulation result schema: `occamworm.sim.result/0.1`.

## 1. Source files

Line-oriented UTF-8 (ASCII outside comments). One statement per line; a newline inside `(...)` or `[...]` does not end the statement. `#` starts a comment that runs to the end of the line. Limits (violations are `E_LIMIT`): 1 MiB per source, 200,000 tokens, 64 operands in one `+`/`*` chain, about 32 nesting levels of parentheses or calls, 50,000 expression nodes, identifiers of at most 64 characters.

```text
wrl 0.1                       # required first statement
rule <name>                   # optional, metadata only (not hashed)
tier G0 | G1                  # required
dt_max <number>[ [s] ]        # optional: largest dt (seconds) the rule is valid for
stimulus_unit <unit>          # optional, default 1: unit of the `stimulus` input

state <name> : <unit> = <number>
param <name> : <unit> [= <number>] [in [<lo>, <hi>]] ( trainable bits <n> | fixed )
input <name> = <input-expression>
let   <name> = <expression>
next  <name> = <expression>   # new value of register <name>; registers without `next` hold their value
gap   <name> [scale <param>]  # semi-implicit electrical coupling of register <name> (G1)
observe <operator>(<register> [, <tau>])
```

Statements are processed in source order and names must be defined before use (`state`, `param`, `input`, `let` define names; `next`, `gap`, `observe` refer to them). Exactly one `observe` is required. `dt_max`, `rule`, `tier`, `stimulus_unit` may appear once each, anywhere after `wrl`.

* **`state`**: a register with a unit and an initial value (applied to every neuron unless the simulation input overrides it).
* **`param`**: a trainable parameter has bounds `in [lo, hi]` (`lo < hi`) and a declared precision `bits` (1..32); its value defaults to `(lo + hi) / 2` and must lie inside the bounds. A `fixed` parameter has a value, no bounds, and is a program constant (part of the hash). The unit follows the colon.
* **`input`**: a `let` whose right-hand side must be one of `stimulus`, `sum_in(..)`, `count_in(..)`, `type_mask(..)`, `delay(..)`. It documents which values enter from outside the neuron.
* **`next r`**: the register update; its unit must equal the register unit.
* **`gap r [scale p]`**: couples register `r` over the gap-junction graph; `p` (dimensionless, lower bound >= 0) multiplies every conductance. See §7.6.
* **`observe identity_v1(r)`** reports register `r`; **`observe calcium_linear_v1(r, tau)`** reports a first-order filter of `r` (§7.7). `tau` is a parameter name (unit `s`) or a constant such as `0.5[s]`.

### 1.1 Units

`unit := term (('*' | '/') term)*` with `term := ('s' | 'V' | '1' | 'dimensionless') ['^' ['-'] int]`, left-associative, exponents bounded by 8. Units are integer powers of two base units, seconds `s` and volts `V`. Canonical spelling: `1` for dimensionless, otherwise `V` then `s`, joined by `*`, e.g. `V*s^-1`. `1/s` and `s^-1` are the same unit.

### 1.2 Expressions

```text
expr    := term (('+' | '-') term)*
term    := unary ('*' unary)*
unary   := '-' unary | primary
primary := number ['[' unit ']'] | name | name '(' [arg (',' arg)*] ')' | '(' expr ')'
arg     := expr | '[' number (',' number)* ']'          # the list form is only valid as the table of lut
```

* There is **no division**. `/` is a syntax error (`E_SYNTAX`); use `leaky_integrate`, `decay`, or multiply by a parameter. There is no exponent operator and no comparison operator.
* `a - b` is `add(a, neg(b))`; infix `+`/`*` build binary `add`/`mul` (the n-ary function forms `add(a, b, c)`, `mul(...)`, `min(...)`, `max(...)` are also accepted).
* A number is dimensionless unless written with `[unit]`. A literal that is exactly `0` without a `[unit]` adopts the unit of its context (`next x = x + 0` is legal for `x : V`); every other literal must match.
* A bare name resolves to a register (its **old** value), a parameter, or an `input`/`let` binding. `stimulus` is a keyword (also `stimulus()`).
* Reserved words (cannot be names): the statement keywords, `in scale bits trainable fixed sub decay dimensionless`, and every operator name below.

### 1.3 Operators

`unit(x)` is the unit of `x`. "Same unit" means all listed operands must have identical units (a zero literal adapts). Tiers: see §1.4.

| Surface form | IR op | Arguments | Result unit | Tier |
|---|---|---|---|---|
| number `c[u]` | `const` | attr `value` | `u` | G0 (integer, dimensionless), G1 |
| name of a parameter, `param(p)` | `param` | attr `param` | declared | G1 |
| name of a register, `state(r)` | `state` | attr `register` | register unit | both |
| `stimulus` | `stimulus` | none | `stimulus_unit` | both |
| `type_mask(T)` | `type_mask` | attr `type` (a word) | 1 | both |
| `sum_in(r, exc\|inh\|all)` | `sum_in` | attrs `register`, `select` | `unit(r)` | G1 |
| `count_in(r, k)` | `count_in` | attrs `register`, `k` (integer >= 0); `r` dimensionless | 1 | G0 |
| `delay(r, n)` | `delay` | attrs `register`, `ticks` (0..100000) | `unit(r)` | both |
| `add(a, b, ...)`, `a + b` | `add` | n >= 2, same unit | that unit | both |
| `mul(a, b, ...)`, `a * b` | `mul` | n >= 2 | product of units | both |
| `neg(x)`, `-x`, `sub(a, b)` | `neg` (+ `add`) | 1 | `unit(x)` | both |
| `abs(x)` | `abs` | 1 | `unit(x)` | both |
| `min(a, b, ...)`, `max(a, b, ...)` | `min`, `max` | n >= 2, same unit | that unit | both |
| `clamp(x, lo, hi)` | `clamp` | same unit; constant `lo <= hi` | that unit | both |
| `relu(x)` | `relu` | 1 | `unit(x)` | G1 |
| `tanh(x)`, `sigmoid(x)` | `tanh`, `sigmoid` | `x` dimensionless | 1 | G1 |
| `threshold(x, theta)` | `threshold` | same unit | 1 | both |
| `select(c, a, b)` | `select` | `c` dimensionless; `a`, `b` same unit | `unit(a)` | both |
| `lut(x, [e0, e1, ...])` | `lut` | `x` dimensionless; 1..4096 finite entries (attr `table`) | 1 | G0 |
| `leaky_integrate(x, target, tau)` | `leaky_integrate` | `x`, `target` same unit; `tau` a parameter or constant, unit `s` | `unit(x)` | G1 |
| `decay(x, tau)` | `leaky_integrate(x, 0, tau)` | as above | `unit(x)` | G1 |
| `euler_leak(x, target, tau)` | `euler_leak` | as `leaky_integrate`; stability bound of §3 | `unit(x)` | G1 |

`sum_in` selectors: `exc` sums only excitatory edges (non-negative contributions), `inh` sums only inhibitory edges (also reported as non-negative magnitudes; the program subtracts them), `all` is the signed sum. This keeps edge signs a property of the graph and the sign constraint of §5.8 a property of the program.

### 1.4 Tiers

* **G0 (truth-table cellular automata):** dimensionless integer registers, integer constants, no parameters, no `gap`, no `calcium_linear_v1`. Ops: `const state stimulus type_mask count_in delay add mul neg abs min max clamp threshold select lut`.
* **G1 (stable continuous local rules):** ops: `const param state stimulus type_mask sum_in delay add mul neg abs min max clamp relu tanh sigmoid threshold select leaky_integrate euler_leak` (+ `decay`), `gap`, both observation operators. `lut` and `count_in` are G0 only.

## 2. Diagnostics

Compilation throws one error with a stable code, a message, and a 1-based line and column when known: `E_SYNTAX` (lexical/grammar), `E_NAME` (unknown, duplicate or reserved name, missing `observe`), `E_UNIT` (unit mismatch, e.g. adding seconds to a dimensionless value), `E_TYPE` (arity, argument kind, bounds), `E_TIER` (operator not in tier), `E_STABILITY` (§3), `E_LIMIT` (resource limits). The runtime adds `E_INPUT`, `E_GRAPH`, `E_RUNTIME`, `E_JSON`, `E_IO`. Example: `E_UNIT at 6:32: unit mismatch in add: expected [1] but found [s]`.

## 3. Stability rule (§5.5)

* A time constant (`tau` of `leaky_integrate`, `euler_leak`, `decay`, and of `calcium_linear_v1`) must be a **parameter or constant with unit `s`** whose lower bound (a constant's value, a fixed parameter's value) is **strictly positive**; otherwise `E_STABILITY`/`E_TYPE`/`E_UNIT`.
* `leaky_integrate` is the exact exponential step (§7.4) and has no further bound.
* `euler_leak` is forward Euler for `dx/dt = (target - x) / tau`: `x + (dt / tau) (target - x)`. It requires a `dt_max` declaration and **`dt_max / tau_lower <= 1`** (monotone, overshoot-free regime); otherwise `E_STABILITY`. At run time the interpreter rejects `dt > dt_max` (`E_INPUT`).

`configs/rules/rejected/leak-euler-unstable.wrl` is the §5.5 failure case; `configs/rules/leak-adapt-euler.wrl` is the same rule made valid by bounding `tau` below by `dt_max`. Other unbounded dynamics (for example positive feedback through `mul`) are not detected statically; they surface as `E_RUNTIME` when a value becomes non-finite (§7.9).

## 4. Canonicalisation (§5.6)

`compile = parse -> check -> canonicalise`. The canonical program is a hash-consed expression DAG with explicit register writes.

1. **Lowering.** Locals (`input`/`let`) are substituted; sharing is preserved. `sub`, infix `-` become `neg` + `add`; `decay(x, tau)` becomes `leaky_integrate(x, 0, tau)`; `delay(r, 0)` becomes `state(r)`. Registers without `next` get an explicit hold write `state(r)`.
2. **Folding** (exact operations only): `add`, `mul`, `neg`, `abs`, `min`, `max`, `clamp`, `relu`, `threshold`, `select`, `lut` with constant operands fold. `tanh`, `sigmoid`, `leaky_integrate`, `euler_leak` never fold, because libm results are not bit-identical across platforms and the hash must be.
3. **Associative/commutative normalisation.** Nested `add`/`mul`/`min`/`max` are flattened to one n-ary node. Constants are merged into one constant (sorted ascending, then folded left to right), `0` is dropped from sums, `1` from products, `min`/`max` drop duplicate operands, `x + (-x)` cancels, `mul(-1, x)` is `neg(x)`, `max(0, x)` is `relu(x)`. Remaining operands are sorted by `(rank, digest)`: rank 0 constants, 1 parameters, 2 registers, 3 other leaves, 4 composite nodes; the digest is the SHA-256 of the node's structural encoding, so the order is independent of names and source order.
4. **Algebraic cleanup.** `neg(neg x) = x`, `abs(abs x) = abs x`, `abs(neg x) = abs x`, `relu(relu x) = relu x`, `select(c, a, a) = a`, `select(const, a, b)` picks the branch, `leaky_integrate(x, x, tau) = x`.
5. **CSE.** Nodes with identical operation, attributes, unit and operands are one node.
6. **Dead-code elimination.** Only code reachable from the observed register is kept: a register is live if it is observed or read (`state`, `sum_in`, `count_in`, `delay`) by a live register's update; unused parameters, dead registers, their writes and a `gap` on a dead register are removed. Their source names are listed under `eliminated` in the IR.
7. **Alpha-renaming.** Registers become `s0, s1, ...`, parameters `p0, p1, ...`, instructions `n0, n1, ...`. Registers and parameters are relabelled by trying every permutation and keeping the lexicographically smallest canonical text, so declaration order and names never matter; programs with more than 720 relabellings (`registers! x parameters!`) keep declaration order.
8. **Numbering.** Instructions are numbered by depth-first post-order from the register writes in canonical register order, operands in canonical order.
9. **Hash.** `program_hash = SHA-256(canonical text)` (lowercase hex).

**Canonical text** (`ow rule canon`, `ir.canonical_source`) is itself valid WRL; compiling it reproduces the same text and hash. Layout: `wrl 0.1`, `tier`, `dt_max` (if any), `stimulus_unit`, one `state` line per register, one `param` line per parameter, one `let nK = ...` line per instruction (leaves included, `const` as `<number>[<unit>]`), one `next sK = nK` per register, `gap`, `observe`. The rule name is omitted. A trainable parameter prints `in [lo, hi] trainable bits n` **without its value** (the starting value is an optimiser choice, not program identity); a fixed parameter prints `= value fixed`. Every line ends with `\n`. Numbers are shortest-round-trip decimal in the form `d[.ddd]e<exp>` with a `-` sign when negative, no `+`, no exponent padding (`0.25 -> 2.5e-1`, `1 -> 1e0`, `0 -> 0e0`); `-0` is normalised to `0`.

**Equality of behaviour is not claimed**: equal hashes mean equal canonical IR. Flattening and constant merging reassociate floating-point sums, so the canonical IR (not the surface text) defines the exact arithmetic, and the interpreter evaluates the IR.

## 5. IR JSON schema (`ow rule inspect <file>`)

```jsonc
{
  "schema": "occamworm.wrl.ir/0.1",
  "grammar_version": "0.1",
  "bit_code_version": 1,                       // version of the L_struct code (§9)
  "compiler_build": "ow 0.1.0; Clang 23.1.2; C++26; Release; fp-contract=off",
  "program_hash": "<64 hex>",
  "rule": "name",                              // metadata, not hashed
  "tier": "G0" | "G1",
  "dt_max": 0.05 | null,                       // seconds
  "stimulus_unit": "1",
  "registers": [ { "index": 0, "name": "s0", "source_name": "v", "unit": "1", "init": 0 } ],
  "parameters": [ { "index": 0, "name": "p0", "source_name": "tau", "unit": "s", "value": 0.25,
                    "lower": 0.005, "upper": 10, "trainable": true, "bits": 12 } ],
                    // fixed parameters: lower == upper == value, bits 0
  "instructions": [ { "id": 0, "op": "state", "args": [], "attrs": {"register": 0}, "unit": "1" } ],
  "writes": [ { "register": 0, "value": 7 } ], // writes[r].value = id of the instruction holding the new value of register r
  "gap": null | { "register": 0, "scale_param": 1 | null },
  "observation": { "operator": "identity_v1" | "calcium_linear_v1", "register": 0,
                   "tau": null | {"param": 2} | {"const": 0.5} },
  "eliminated": { "registers": ["d"], "parameters": ["unused"] },   // source names removed as dead
  "l_struct": { "total_bits": 123, "l_ast": 100, "l_topology_overrides": 0, "l_type_dispatch": 23,
                "components": { "header":.., "registers":.., "parameters":.., "instructions":.., "writes":.., "gap":.., "observation":.. },
                "instruction_bits_by_op": { "add": .., "state": .. } },
  "l_params": { "total_bits": 44, "per_parameter": [ { "name": "p0", "bits": 12 } ] },
  "canonical_source": "wrl 0.1\ntier G1\n..."      // the hashed bytes
}
```

Rules for consumers:

* `instructions` are in evaluation order; an instruction's `args` are ids of **earlier** instructions. Evaluate them in order, once per neuron per tick.
* `attrs` by op: `const {value}`; `param {param}` (index into `parameters`); `state {register}`; `type_mask {type}`; `sum_in {register, select}`; `count_in {register, k}`; `delay {register, ticks}`; `lut {table}`; every other op `{}`. `stimulus` has `{}`.
* Argument order is as in §1.3: `clamp(x, lo, hi)`, `threshold(x, theta)`, `select(c, a, b)`, `lut(x)`, `leaky_integrate(x, target, tau)`, `euler_leak(x, target, tau)`. `add`, `mul`, `min`, `max` are n-ary and fold **left to right in `args` order**.
* In `leaky_integrate`/`euler_leak` the `tau` argument is a `param` or `const` instruction (seconds).
* Parameter values come from the simulation input by `source_name`, else `value`. Registers are addressed in the simulation input and output by `source_name`.
* `ow sim run --program` also accepts an IR JSON file (extension `.json`).
* `ir_from_json` / `ir_to_json` in `libs/ow-ir/src/compile.cpp` are the reference reader and writer.

## 6. Execution semantics (reference interpreter, `libs/ow-sim`)

### 6.1 Simulation input JSON

```jsonc
{
  "dt": 0.1,                         // seconds, > 0, <= dt_max when declared
  "n_steps": 20,                     // 0 .. 10,000,000
  "sample_ticks": [0, 5, 10, 20],    // strictly increasing, in [0, n_steps]; default: every tick 0..n_steps
  "graph": {
    "neurons":  [ {"id": "A", "type": "sensory"} ],                   // order defines the neuron index; type defaults to "generic"
    "chemical": [ {"pre": "A", "post": "B", "weight": 0.5, "sign": 1, "delay": 2} ],   // weight >= 0, sign +1/-1, delay in ticks >= 0
    "gap":      [ {"a": "A", "b": "B", "g": 2.0} ]                    // undirected, g >= 0 (unit 1/s)
  },
  "stimulus": [ {"neuron": "A", "start": 0, "end": 5, "amplitude": 1.0} ],   // active for ticks start <= t < end
  "params":   { "tau": 0.3 },                      // by source name; must lie inside the declared bounds
  "initial_state": { "v": {"A": 1.0} },            // register -> neuron -> value (sparse), or a dense list in neuron order
  "observed": ["A", "B"],                          // neurons to report (a mask); default all, in graph order
  "edits": { "delete_chemical": [["A", "B"]], "delete_gap": [["A", "B"]] }   // edge deletions applied before building the graph
}
```

Identifiers (neuron ids, types) match `[A-Za-z0-9_.:-]{1,64}`. Parallel edges are allowed; a gap junction listed once is one undirected junction (listing `(a,b)` and `(b,a)` makes two parallel junctions). Parameter or register names that were eliminated as dead code are accepted and ignored.

### 6.2 Graph arrays and summation order

Chemical edges are stored CSR by **postsynaptic** neuron: row `i` lists the inputs of neuron `i`, entries sorted ascending by `(pre index, delay, sign, weight)` (orientation version 1, `libs/ow-core/include/occamworm/core/graph.hpp`). Gap junctions are stored in both endpoint rows, sorted by `(neighbour index, g)`. **All neighbourhood sums are accumulated sequentially in row order starting from `0.0`.** A second implementation may sum in another order within the conformance tolerance; the reference order is the tie-breaker for bit-exact replay.

### 6.3 Timestep order (§6.1)

State at tick `t` is `x_r[i](t)` for registers `r` and neurons `i`, plus the observation state `c_i(t)`. One step produces tick `t+1`:

1. **Stimulus.** `u_i(t) = sum of amplitude over events on neuron i with start <= t < end` (events accumulate in listed order).
2. **Snapshot.** Every read below uses only tick-`t` values and history; nothing written in this step is visible to any neuron in this step.
3. **Chemical aggregation.** `sum_in`/`count_in` read neighbours through the delay history (§6.5).
4. **Electrical coupling.** The gap sums `G_i`, `S_i` are computed from the old values `x_g(t)` of the coupled register (§6.6).
5. **Local update.** For each neuron, evaluate the instruction list in order; `w_r[i] = value of instruction writes[r]`. Then, for the coupled register, apply the gap formula to `w_g[i]`.
6. **Observation.** `c_i(t+1)` from the post-update value of the observed register (§6.7).
7. **Commit.** Write all new registers `x_r(t+1) = w_r` and push them into the history (§6.5).
8. **Sampling.** If `t+1` is a sample tick, record `x_r(t+1)` and `c(t+1)` for the reported neurons. Tick 0 (the initial state) is sampled before the first step when `0` is in `sample_ticks`. The tick of a sample is `k`; its physical time is `k * dt`.

### 6.4 Operator formulas

All values are `float64`. For a neuron `i` at tick `t`, with `H_r[j](s)` the history of §6.5 (`H_r[j](t) = x_r[j](t)`):

| IR op | Value |
|---|---|
| `const` | `value` |
| `param` | the parameter's value `theta_p` (constant in time) |
| `state` | `x_r[i](t)` |
| `stimulus` | `u_i(t)` |
| `type_mask` | `1.0` if `type(i) == name` (exact string match) else `0.0` |
| `sum_in(r, sel)` | `acc = 0.0`; for each row entry `e` of neuron `i` in order, with source `j = pre[e]`, `term = weight[e] * H_r[j](t - delay[e])`; `exc`: add `term` if `sign[e] > 0`; `inh`: add `term` if `sign[e] < 0`; `all`: add `term` if `sign[e] > 0`, add `-term` if `sign[e] < 0` |
| `count_in(r, k)` | number of row entries `e` with `H_r[pre[e]](t - delay[e]) == k` exactly, as a float (any sign, any weight) |
| `delay(r, n)` | `H_r[i](t - n)` |
| `add` | `a0 + a1`, then `+ a2` ... (left fold) |
| `mul` | `a0 * a1`, then `* a2` ... (left fold) |
| `neg`, `abs` | `-x`, `fabs(x)` |
| `min` | `acc = a0`; for each next `v`: `acc = (v < acc) ? v : acc` |
| `max` | `acc = a0`; for each next `v`: `acc = (v > acc) ? v : acc` |
| `clamp(x, lo, hi)` | `t = (lo > x) ? lo : x`; result `(hi < t) ? hi : t` |
| `relu(x)` | `(x > 0) ? x : 0` |
| `tanh(x)` | libm `tanh` |
| `sigmoid(x)` | `1 / (1 + exp(-x))` |
| `threshold(x, theta)` | `(x >= theta) ? 1 : 0` |
| `select(c, a, b)` | `(c != 0) ? a : b`; all three operands are always evaluated |
| `lut(x)` | `f = floor(x)`; index `0` if `f <= 0`, `n-1` if `f >= n-1`, else `f`; result `table[index]` |
| `leaky_integrate(x, target, tau)` | `a = exp(-(dt / tau))`; `target + (x - target) * a` (exact solution of `dx/dt = (target - x)/tau` with `target` held over the step) |
| `euler_leak(x, target, tau)` | `r = dt / tau`; `x + r * (target - x)` |

Transcendental functions (`tanh`, `exp`) come from the platform libm and agree between libraries only to ~1 ulp; conformance tolerances account for it. Every other operation uses IEEE-754 basic operations only (`-ffp-contract=off`, no fast-math in the reference build). `tau` is the value of its `param`/`const` argument.

### 6.5 Delay ring and history initialisation (§6.4)

Delays are whole ticks. The history of register `r` is `H_r[j](s) = x_r[j](s)` for `s >= 0` and **`H_r[j](s) = x_r[j](0)` for `s < 0`** (constant extension of the initial state; the initial state includes any `initial_state` override). `delay = 0` reads the old state at tick `t`, never a value computed in the current step. The interpreter stores the history in a ring of `D + 1` slots per register, where `D` is the largest of all edge delays and all `delay(r, n)` ticks; slot index `tick mod (D + 1)` holds tick `tick`, all slots are initialised to the initial state, and negative ticks use floor-mod, so unwritten slots return the initial state. Step 7 writes slot `(t+1) mod (D+1)`, which overwrites tick `t - D`, no longer needed at tick `t + 1`. A second implementation may use any storage that realises the `H` definition above.

### 6.6 Gap junctions: node-local semi-implicit coupling (§6.3)

For the register `g` named by `gap g [scale p]`, with `s = value of p` (1 if no scale), each junction `{i, j, conductance}` contributes `c = s * conductance` to the rows of both endpoints. For neuron `i`, from the **old** values: `G_i = sum_j c_ij` and `S_i = sum_j c_ij * x_g[j](t)`, sums in row order from `0.0`. After the local update has produced `w_g[i]` (the provisional new value from the program's `next`):

```
x_g[i](t+1) = ( w_g[i] + dt * S_i ) / ( 1 + dt * G_i )
```

This is backward Euler on the diagonal of the coupling term with neighbours taken from the old state: `x_i' = w_i + dt * sum_j c_ij (x_j - x_i')`. Properties (tested in the conformance suite): (a) **constant equilibrium is preserved**: if `w_g[j] = x_g[j](t) = c` for all neighbours, the result is exactly `c` up to rounding; (b) **unconditionally stable**: the result is a convex combination of `w_g[i]` and the old neighbour values, so it never leaves their range for any `dt * c`; (c) first-order accurate; (d) it does **not** conserve the total of the register exactly (a symmetric Jacobi-style scheme, chosen for exact replicability and order independence, not for conservation). Neurons without junctions get `w_g[i]` unchanged (`G = S = 0`). The `gap` stage acts on exactly one register.

### 6.7 Observation

`observation state c` has one value per neuron. Initial value: `c_i(0) = x_o[i](0)` for the observed register `o` (steady state of the filter).

* `identity_v1`: the reported observation is `x_o[i](t+1)`.
* `calcium_linear_v1(o, tau)`: after step 5 (including the gap stage), `x_new = x_o[i](t+1)` and `c_i(t+1) = x_new + (c_i(t) - x_new) * exp(-(dt / tau))` (the same exact integrator as `leaky_integrate`, driven by the **new** register value). The reported observation is `c_i(t+1)`. Parameters of the indicator model belong to the measurement model and are versioned separately (§4.5, [BASELINES](../science/BASELINES.md)); this operator is a linear placeholder.

### 6.8 Output JSON (`ow sim run`)

```jsonc
{ "schema": "occamworm.sim.result/0.1", "program_hash": "...", "grammar_version": "0.1", "compiler_build": "...",
  "dt": 0.5, "n_steps": 4, "sample_ticks": [0, 2, 4], "neurons": ["A", "B"],      // reported neurons, in report order
  "registers": { "v": [ [v_A, v_B] /* tick 0 */, [..] /* tick 2 */, [..] /* tick 4 */ ] },   // [sample][reported neuron], keyed by source name
  "observation": { "operator": "calcium_linear_v1", "register": "v", "values": [ [..], [..], [..] ] } }
```

### 6.9 Errors and bounds checking

Input errors (`E_INPUT`, `E_GRAPH`): non-positive or non-finite `dt`, `dt > dt_max`, `n_steps` out of range, bad `sample_ticks`, unknown neuron/register/parameter names, parameter outside bounds, invalid weights/signs/delays/conductances, duplicate ids. Runtime error (`E_RUNTIME`): any instruction value, gap result, or state that is not finite, reported with tick, neuron and instruction id. All array accesses are bounds-checked in debug and sanitizer builds (`-fsanitize=address,undefined` is part of the test matrix).

### 6.10 Stochastic operators (not implemented) and the Philox plan

G0/G1 have no stochastic operators, so the interpreter has no RNG. For G5, `normal_noise` and `bernoulli` will use **Philox4x32-10** (Salmon et al. 2011) with the key `SHA-256-derived (experiment_id, program_hash, animal_id, trial_id, replicate_id)` truncated to 64 bits and the counter `(tick, neuron index, node id, draw index)`; uniform variates are `(u32 + 0.5) * 2^-32`, normal variates use Box-Muller on two such uniforms (documented, library-independent), and `bernoulli(p)` is `u < p`. Common random numbers are obtained by keeping the key and counter equal across candidates. Nothing in v0.1 depends on this.

## 7. Conformance case format

Each file `tests/conformance/*.json` is self-contained; `ow sim conformance --suite tests/conformance` runs all of them in file-name order and exits non-zero on any failure (also run by CTest as `conformance_suite`).

```jsonc
{
  "name": "...", "description": "...", "derivation": "how the expected values were obtained, analytically",
  "program": { "source": ["wrl 0.1", "tier G1", "..."] }        // or {"path": "relative/to/this/file.wrl"}
  "program_hash": "<optional, pins the canonical hash>",
  "input": { /* simulation input, §6.1 */ },
  "check": { "type": "traces", "tolerance": {"abs": 1e-12, "rel": 0},
             "expected": { "registers": {"v": [[..]]}, "observation": [[..]] } }
}
```

`check.type`:

* `traces`: compare the sampled registers (any subset) and/or observation, `|actual - expected| <= abs + rel*|expected|`.
* `compile_error`: `{"type":"compile_error","code":"E_UNIT"}`; the program must fail with that code.
* `permutation`: `{"type":"permutation","permutation":[2,0,1],"tolerance":..}`: reorders the neuron list so that new position `j` holds old neuron `permutation[j]` and requires every output of new neuron `j` to equal that of old neuron `permutation[j]`.
* `convergence`: `{"type":"convergence","dts":[0.1,0.05,0.025],"time":1.0,"register":"v","neuron":"A","expected":<analytic value at time>,"max_ratio":0.6,"max_error_finest":0.01}`: re-runs the case at each `dt` (`n_steps = time/dt`) and requires `err_k+1 <= max_ratio * err_k` and the finest error `<= max_error_finest`.

Expected values come from analytic solutions or hand-derivable arithmetic (documented in each case's `derivation`), never from running the interpreter.

## 8. Enumeration (OW-010)

`ow search enumerate --config configs/searches/<name>.json` prints JSON lines: one `{"type":"program","index","hash","l_struct","l_params","source"}` per **distinct canonical program** (the `source` is the canonical text, so it compiles to `hash`), then one `{"type":"summary",...}` with the accounting of every attempted candidate (`generated`, `rejected.{type_or_unit,stability,budget}`, `duplicates`, `unique`).

Config keys: `name`, `tier`, `max_nodes` (1..12, total expression nodes over all register updates), `max_depth` (each update; a leaf has depth 1), `max_registers` (1..3), `max_parameters`, `ops` (allowlist: expression operators `neg abs relu tanh sigmoid add mul min max threshold clamp select lut leaky_integrate euler_leak` and leaf classes `const param stimulus sum_in count_in delay type_mask`; `state` leaves always exist), plus pools `constants`, `tau_constants` (seconds), `sum_in_selects`, `count_values`, `delays`, `type_names`, `lut {sizes, values}`, `parameters [{unit, lower, upper, bits}]`, `dt_max`, `register_init`.

Grammar: a program has `R` registers `r0..` (dimensionless, init from `register_init`), every parameter of the pool declared as trainable `p0..`, `next r_k = <tree>`, and `observe identity_v1(r0)`. A tree is a leaf (register, `stimulus`, `sum_in(r,sel)`, `count_in(r,k)`, `delay(r,n)`, `type_mask(T)`, a constant, a parameter whose unit is not `s`) or an allowed operator applied to subtrees; the time-constant operand of `leaky_integrate`/`euler_leak` ranges over time-unit parameters and `tau_constants` only; `lut` ranges over every table of an allowed size over the allowed values; commutative operators are generated with one operand order. Order of emission: `R` ascending, total node count ascending, size tuples lexicographic, then trees in generation order (leaves in the order register, stimulus, sum_in, count_in, delay, type_mask, constant, parameter; operators in the order of `kInnerOps`). Each candidate goes through the real compiler; static gates, in order: parse/type/unit/tier errors, explicit-Euler stability, `max_registers`/`max_parameters` after dead-code elimination, then the canonical-hash cache. The test `tests/cpp/test_search.cpp` proves that every canonical program in the budget is emitted exactly once by comparing with an independent brute-force enumeration.

## 9. Bit cost (§5.7): frozen prefix-free code, version 1

`L_struct = L_AST + L_topology_overrides + L_type_dispatch`; `L_total = L_struct + L_params`. Integer codes: `gamma(n)` (n >= 1) is the Elias gamma code, `2*floor(log2 n) + 1` bits; `gamma0(n) = gamma(n+1)`; signed integers use zigzag then `gamma0` (`0,-1,1,-2,... -> 0,1,2,3`); a **constant** costs `1 + gamma(d) + 4d + zigzag-gamma0(e)` where `d` is the number of significant digits and `e` the decimal exponent of its shortest round-trip form; a **unit** costs `zigzag-gamma0(s exponent) + zigzag-gamma0(V exponent)`.

| Part | Bits |
|---|---|
| header | tier (3, fixed) + `dt_max` flag (1) + constant if present + stimulus unit |
| registers | `gamma(R)` + per register: unit + constant(init) |
| parameters | `gamma0(P)` + per parameter: unit + 1 (trainable flag) + (trainable: constant(lower) + constant(upper); fixed: constant(value)) |
| instructions | `gamma0(count)` + per instruction: kind code + attributes + argument references |
| kind codes | `state` = `00`, `const` = `01`, `param` = `100`, `add` = `101`, `mul` = `110`, every other op = `111` + 5-bit index (8 bits): `stimulus 0, type_mask 1, sum_in 2, count_in 3, delay 4, neg 5, abs 6, min 7, max 8, clamp 9, relu 10, tanh 11, sigmoid 12, threshold 13, select 14, lut 15, leaky_integrate 16, euler_leak 17` |
| attributes | `const`: constant + unit; `param`/`state`: `gamma0(index)`; `sum_in`: `gamma0(reg)` + 2; `count_in`: `gamma0(reg) + gamma0(k)`; `delay`: `gamma0(reg) + gamma(ticks)`; `type_mask`: `gamma(len) + 8*len` (dispatch); `lut`: `gamma(len)` + constant per entry (dispatch); `add/mul/min/max`: `gamma(arity - 1)` |
| argument reference | `gamma(distance back to the producing instruction)` (>= 1) |
| writes | per register `gamma(count - id)` |
| gap | 1 + (present: `gamma0(reg)` + 1 + (scale: `gamma0(param)`)) |
| observation | 1 + `gamma0(reg)` + (calcium: 1 + `gamma0(param)` or constant) |

`L_params = sum of bits over trainable parameters` (declared precision: a trainable parameter is quantised to a `2^bits`-point grid over `[lower, upper]`). In v0.1 `L_params` does not depend on the fitted value; value-dependent codes are a later refinement that would bump `bit_code_version`. `L_topology_overrides` is 0 (G0/G1 have no topology overrides). `type_mask` names and `lut` tables are reported as `L_type_dispatch`.

## 10. Provenance

Every IR JSON, simulation result and enumeration summary records `grammar_version` and `compiler_build` (project version, compiler id and version, C++ standard, build type, `fp-contract=off`). `ow --version` prints the same string. The canonical program hash identifies a program; results are comparable across builds only when the conformance suite passes on both.

## 11. CLI

```text
ow --version
ow rule inspect <file.wrl>                        # IR JSON (§5) on stdout
ow rule canon <file.wrl>                          # canonical text (the hashed bytes)
ow sim run --program <file.wrl|ir.json> --case <case.json> [--out <file>]   # §6; --case may be a conformance case (its "input" is used)
ow sim conformance --suite tests/conformance
ow search enumerate --config configs/searches/g1-tiny.json [--out <file>]
```

Exit status: 0 success, 1 failure (diagnostics on stderr as `error: <code> at line:col: message`), 2 usage error.

## 12. Deviations from, and deferrals against, the illustrative §5

* No `/`: `dt/tau` stays inside the integrator primitives. `sum_gap` is realised as the `gap` declaration (a stable discretisation, §6.3), not as an expression that user code could use in an unstable explicit update.
* Not implemented in v0.1: `mean_in`, `sum_mod`, `piecewise`, `hold`, `rise`, `fluorescence`, `readout`, `edge_weight`, stochastic operators, G2+ tiers, value-dependent `L_params`, a dynamic finite-output gate in the enumerator (only the static gates of §8).
* The observation operator is a linear placeholder; the indicator model is a separate, versioned stage.
