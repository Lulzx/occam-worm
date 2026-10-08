# `ow-search`

Program enumeration under bit budgets, structured mutation, canonical cache and Pareto ranking.

**Invariant:** Search budget and every attempted candidate are logged.

Tickets: [OW-010](../../docs/planning/tickets/OW-010.md), [OW-012](../../docs/planning/tickets/OW-012.md) · Spec: [§7.2](../../docs/inference/SEARCH_AND_INFERENCE.md), [§7.3](../../docs/inference/SEARCH_AND_INFERENCE.md), [§7.8](../../docs/inference/SEARCH_AND_INFERENCE.md)

Status: enumeration and canonical deduplication implemented ([OW-010](../../docs/planning/tickets/OW-010.md)); structured mutation, Pareto ranking and the search log of [OW-012](../../docs/planning/tickets/OW-012.md) are not.

`ow search enumerate --config configs/searches/g1-tiny.json` emits one JSON line per distinct canonical program plus a summary of every attempted candidate. Grammar, ordering and gates: [WRL_SYNTAX.md](../../docs/language/WRL_SYNTAX.md) §8. The exactly-once property is checked against an independent brute force in `tests/cpp/test_search.cpp`.
