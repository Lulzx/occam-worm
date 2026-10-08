# configs/

Versioned configuration. Every run records the hash of the config it used.

- [`datasets/`](datasets/README.md) — Dataset manifests (for example `primary.toml`): sources, checksums, versions, licenses.
- [`splits/`](splits/README.md) — Split specifications. The scheme is chosen from M0 coverage tables before any model is scored.
- [`rules/`](rules/README.md) — WRL rule sources.
- [`experiments/`](experiments/README.md) — Experiment configurations and the feasible-intervention menu.
- [`searches/`](searches/README.md) — Search configurations: grammar tier, budgets (`max_nodes`, `max_registers`, `max_depth`, `max_parameters`), seeds.

Status: not implemented.
