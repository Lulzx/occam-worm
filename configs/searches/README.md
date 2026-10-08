# configs/searches/

Search configurations: grammar tier, budgets (`max_nodes`, `max_registers`, `max_depth`, `max_parameters`), seeds.

Tickets: [OW-010](../../docs/planning/tickets/OW-010.md), [OW-012](../../docs/planning/tickets/OW-012.md) · Spec: [§7.3](../../docs/inference/SEARCH_AND_INFERENCE.md), [§18.4](../../docs/engineering/PERFORMANCE.md)

Status: `g0-tiny.json` and `g1-tiny.json` (OW-010 exactly-once test budgets) are implemented; run with `ow search enumerate --config configs/searches/g1-tiny.json`. Key reference: [WRL_SYNTAX.md](../../docs/language/WRL_SYNTAX.md) §8. Seeds and fit budgets belong to OW-012.
