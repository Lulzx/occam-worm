# artifacts/

Immutable, content-addressed run outputs (audits, baselines, searches, evaluations, reports). Gitignored.

**Invariant:** No mutable `latest` pointers; run IDs hash the raw manifest, schema, split IDs, graph version, program hash, fit config, code revision and seeds.

Tickets: [OW-015](../docs/planning/tickets/OW-015.md) · Spec: [§14.7](../docs/engineering/REPRODUCIBILITY.md)

Status: not implemented.
