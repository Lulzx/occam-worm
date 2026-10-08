# configs/datasets/

- [`sources.json`](sources.json): the source registry. Every upstream dataset and code repository, with URL or DOI, expected checksums, license evidence, milestones and metadata retrieval time. Edit source-level fields by hand; refresh asset lists with `python -m occamworm.sources refresh`.
- Dataset manifests for normalized data (for example `primary.toml`) arrive with OW-002.

Usage and findings: [docs/data/ACQUISITION.md](../../docs/data/ACQUISITION.md).

Tickets: [OW-001](../../docs/planning/tickets/OW-001.md) · Spec: [§2.2](../../docs/data/SOURCES.md)
