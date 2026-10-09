# Occam's Worm

Searches a small typed language of local neural update rules, runs candidates on the *C. elegans* connectome, and keeps only rules that predict causal responses in animals they never saw. Brevity is a tie-breaker, never a substitute for held-out accuracy.

**Status:** spec v0.4.1. Done (OW-001–005, OW-008–016):

- data and audit: acquisition, trial import, M0 audit (**go**), animal-wise splits, annotation mapping
- rule toolchain: the C++26 toolchain and interpreter, the Python scalar and JAX simulators
- modelling: scoring, fitting, nested rule search, planning, equivalence, reports, embodiment

In progress: the full baseline benchmark (OW-006/007). First search on atlas data: [SEARCH_DEMO.md](docs/evaluation/SEARCH_DEMO.md).

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
- [Progress report, 2026-10-09 (PDF)](docs/reports/occam-worm-progress-2026-10-09.pdf)
- [Data acquisition](docs/data/ACQUISITION.md) · [Randi 2023 import](docs/data/RANDI2023_IMPORT.md) · [Annotations](docs/data/ANNOTATIONS.md)
- [WRL syntax and IR](docs/language/WRL_SYNTAX.md) · [C++ build and test](docs/engineering/REPRODUCIBILITY.md#local-build-and-test)

## License

[Apache 2.0](LICENSE). Upstream data and code keep their own licenses; most atlas data declares none ([details](docs/data/ACQUISITION.md#findings-2026-10-08)).
