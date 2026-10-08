# Occam's Worm

Searches a small typed language of local neural update rules, runs candidates on the *C. elegans* connectome, and keeps only rules that predict causal responses in animals they never saw. Brevity is a tie-breaker, never a substitute for held-out accuracy.

**Status:** spec v0.4.1. M0 in progress: data acquisition ([OW-001](docs/planning/tickets/OW-001.md)) and the trial-level atlas importer ([OW-002](docs/planning/tickets/OW-002.md)) are done. The C++26 rule toolchain, reference interpreter and program enumerator ([OW-008](docs/planning/tickets/OW-008.md), [OW-009](docs/planning/tickets/OW-009.md) C++ half, [OW-010](docs/planning/tickets/OW-010.md)) are implemented.

## Quickstart

```bash
pip install -e '.[dev]'
```

```bash
python -m occamworm.sources fetch --milestone M0
```

```bash
python -m occamworm.importers randi2023 import
```

```bash
pytest
```

## Docs

- [Spec summary](docs/SUMMARY.md) · [full spec index](docs/README.md)
- [Roadmap](docs/planning/ROADMAP.md) · [tickets](docs/planning/tickets/README.md)
- [Data acquisition](docs/data/ACQUISITION.md) · [Randi 2023 import](docs/data/RANDI2023_IMPORT.md)
- [WRL syntax and IR](docs/language/WRL_SYNTAX.md) · [C++ build and test](docs/engineering/REPRODUCIBILITY.md#local-build-and-test)

## License

[Apache 2.0](LICENSE). Upstream data and code keep their own licenses; most atlas data declares none ([details](docs/data/ACQUISITION.md#findings-2026-10-08)).
