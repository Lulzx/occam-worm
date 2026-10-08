# configs/rules/

WRL rule sources.

Tickets: [OW-008](../../docs/planning/tickets/OW-008.md) · Spec: [§5](../../docs/language/GRAMMAR.md)

Status: example rules for WRL v0.1 ([syntax](../../docs/language/WRL_SYNTAX.md)).

- `leak-adapt.yaml`: the illustrative YAML of §5.5, kept for reference (not parseable).
- `leak-adapt.wrl`: the §5.5 rule in WRL syntax with exact exponential integrators (G1).
- `leak-adapt-euler.wrl`: the same rule with explicit Euler, valid because `dt_max / tau_lower <= 1`.
- `gap-leak.wrl`: a leaky neuron with semi-implicit gap coupling (G1).
- `rule90.wrl`: a G0 truth-table cellular automaton.
- `rejected/leak-euler-unstable.wrl`: must be rejected with `E_STABILITY`.

Inspect one with `ow rule inspect configs/rules/leak-adapt.wrl`.
