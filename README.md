# Occam's Worm

Searches a small typed language of local neural update rules, runs candidates on the *C. elegans* connectome, and keeps only rules that predict causal responses in animals they never saw. Brevity is a tie-breaker, never a substitute for held-out accuracy.

**Status:** spec v0.4.1. M0 in progress: data acquisition ([OW-001](docs/planning/tickets/OW-001.md)) the trial-level atlas importer ([OW-002](docs/planning/tickets/OW-002.md)) and annotation mapping ([OW-013](docs/planning/tickets/OW-013.md)) are done.

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
- [Data acquisition](docs/data/ACQUISITION.md) · [Randi 2023 import](docs/data/RANDI2023_IMPORT.md) · [Annotations](docs/data/ANNOTATIONS.md)

## License

[Apache 2.0](LICENSE). Upstream data and code keep their own licenses; most atlas data declares none ([details](docs/data/ACQUISITION.md#findings-2026-10-08)).
