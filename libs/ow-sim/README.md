# `ow-sim`

Scalar, bounds-checked, deterministic reference interpreter. This is the correctness oracle for every other backend.

**Invariant:** Old-state synchronous semantics (read only old state, write only new state).

Tickets: [OW-009](../../docs/planning/tickets/OW-009.md) · Spec: [§6](../../docs/runtime/SIM_SEMANTICS.md)

Status: C++ reference interpreter implemented for G0/G1 ([OW-009](../../docs/planning/tickets/OW-009.md)); the differentiable Python simulator is pending. Exact semantics: [WRL_SYNTAX.md](../../docs/language/WRL_SYNTAX.md) §6 and the "Implemented (OW-009)" note in [SIM_SEMANTICS.md](../../docs/runtime/SIM_SEMANTICS.md).

`sim.cpp` is the interpreter (`ow sim run --program X --case Y`); `conformance.cpp` runs `tests/conformance/*.json` (`ow sim conformance --suite tests/conformance`).
