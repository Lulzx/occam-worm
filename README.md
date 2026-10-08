# Occam's Worm

Occam's Worm searches a small typed language of local neural update rules, runs candidates on the *C. elegans* connectome, and keeps only rules that predict causal responses in animals they never saw.

The name is the selection principle: prefer the most compact program, but only among programs that predict held-out interventions. Brevity is a prior and a tie-breaker. It never substitutes for causal accuracy.

**Status:** specification v0.4.0. No code is implemented yet.

## Documentation

- [docs/SUMMARY.md](docs/SUMMARY.md): one-page specification
- [docs/README.md](docs/README.md): full specification index, split by topic
- [docs/planning/ROADMAP.md](docs/planning/ROADMAP.md): milestones M0–M7 and their gates
- [docs/planning/tickets/](docs/planning/tickets/README.md): implementation tickets OW-001 to OW-016

## First milestone

The first scientific release (M0–M1, tickets OW-001 to OW-007) does not need the rule language. It audits the trial-level wild-type stimulation atlas, then tests whether responders to one stimulus share temporal dynamics under animal-wise holdout, with the calcium indicator modeled as a separate stage. Start with [OW-001](docs/planning/tickets/OW-001.md).

## Repository layout

```text
occam-worm/
├── docs/        specification, split by topic (start at docs/README.md)
├── libs/        C++20 libraries, namespace occamworm:: (ow-core, ow-ir, ow-sim, ...)
├── python/      the occamworm package: importers, baselines, sim, fit, analysis, plotting
├── configs/     datasets, splits, rules, experiments, searches
├── tests/       synthetic_truth, conformance, adversarial, leakage, fuzz, integration
├── benches/     forward_runtime, candidate_throughput, fitting
├── scripts/     acquisition, manifests, figure reproduction
├── notebooks/   exploratory only; never part of the canonical pipeline
├── artifacts/   immutable, content-addressed run outputs (gitignored)
└── data/        raw (immutable, gitignored), normalized, splits
```

Every directory has a README giving its purpose, its invariant, the tickets that fill it, and links into the spec. Build files (`CMakeLists.txt`, `CMakePresets.json`, `vcpkg.json`, `pyproject.toml`) arrive with the first code ticket that needs them; see [docs/engineering/REPRODUCIBILITY.md](docs/engineering/REPRODUCIBILITY.md) for the toolchain.

## License

[Apache License 2.0](LICENSE) for original code and documentation. Upstream datasets and external simulators keep their own licenses; how conflicts are handled in public releases is [open decision 12](docs/planning/OPEN_DECISIONS.md).
