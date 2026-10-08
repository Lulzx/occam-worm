# `ow-ir`

WRL front end: parser, typed AST, unit checks, canonical IR, `program_hash = SHA256(canonical_IR)` and bit costs (`L_struct`, `L_params`).

**Invariant:** Equivalent reorderings hash identically; unit errors are rejected with deterministic diagnostics.

Tickets: [OW-008](../../docs/planning/tickets/OW-008.md) · Spec: [§5](../../docs/language/GRAMMAR.md)

Status: not implemented.
