# `ow-core`

Typed graphs, units and neural state definitions shared by every other library.

**Invariant:** Stable neuron IDs; no invented edges; unit-checked quantities.

Tickets: [OW-008](../../docs/planning/tickets/OW-008.md) · Spec: [§3.2](../../docs/science/PROBLEM_STATEMENT.md), [§3.3](../../docs/science/PROBLEM_STATEMENT.md), [§5.2](../../docs/language/GRAMMAR.md)

Status: implemented for WRL v0.1 ([OW-008](../../docs/planning/tickets/OW-008.md)).

Contents (`include/occamworm/core/`): `error.hpp` (stable error codes), `sha256.hpp` (FIPS 180-4, tested against the FIPS 180-2 vectors), `numfmt.hpp` (canonical shortest-round-trip number text, prefix-free integer and constant codes), `json.hpp` (small strict JSON reader/writer, insertion-ordered objects), `units.hpp` (units as integer powers of `s` and `V`), `graph.hpp` (typed neuron graph in CSR form). Orientation: chemical edges are stored by postsynaptic row, rows sorted by (pre, delay, sign, weight); gap junctions are stored in both endpoint rows (`kCsrOrientationVersion = 1`, tested in `tests/cpp/test_core.cpp`).
