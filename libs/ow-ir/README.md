# `ow-ir`

WRL front end: parser, typed AST, unit checks, canonical IR, `program_hash = SHA256(canonical_IR)` and bit costs (`L_struct`, `L_params`).

**Invariant:** Equivalent reorderings hash identically; unit errors are rejected with deterministic diagnostics.

Tickets: [OW-008](../../docs/planning/tickets/OW-008.md) · Spec: [§5](../../docs/language/GRAMMAR.md)

Status: implemented for WRL v0.1, tiers G0 and G1 ([OW-008](../../docs/planning/tickets/OW-008.md)). Syntax, canonicalisation rules, IR JSON schema and bit code: [WRL_SYNTAX.md](../../docs/language/WRL_SYNTAX.md).

Pipeline: `parser.cpp` (lexer + parser) -> `check.cpp` (names, units, tiers, stability) -> `canonical.cpp` (flatten, fold, CSE, dead-register elimination, relabelling, hash) -> `bits.cpp` (`L_struct`, `L_params`) -> `source.cpp` (canonical source serializer) -> `compile.cpp` (driver and IR JSON). `scalar_ops.hpp` holds the exact scalar semantics shared by constant folding and the interpreter. Inspect a rule with `ow rule inspect <file.wrl>`.
