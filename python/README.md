# python/ — the `occamworm` Python package

Data import, baselines, the differentiable simulator, fitting, analysis and plotting. Pure Python for now (hatchling); moves to scikit-build-core with the C++ extension at OW-008.

```bash
pip install -e '.[dev]'
```

```bash
pytest
```

| Subpackage | Purpose |
| --- | --- |
| [`sources`](occamworm/sources/README.md) | Source registry, download, checksum verification and license reporting (OW-001). |
| [`importers`](occamworm/importers/README.md) | Source importers producing normalized Arrow/Parquet tables. |
| [`baselines`](occamworm/baselines/README.md) | Baselines B0–B6, each in history and no-history variants, scored on T2a and T2b. |
| [`sim`](occamworm/sim/README.md) | Differentiable simulator (JAX or PyTorch), float64 on CPU for confirmatory fits. |
| [`fit`](occamworm/fit/README.md) | Parameter fitting: bounded transforms, multi-start, gradient checks against finite differences, `L_params` accounting. |
| [`analysis`](occamworm/analysis/README.md) | Splits, uncertainty and reporting. |
| [`plotting`](occamworm/plotting/README.md) | Figures for reports and the diagnostic trace viewer. |

Spec: [§14.2](../docs/engineering/ARCHITECTURE.md)

Status: `sources`, the Randi 2023 importer and the annotation mapping (OW-013) implemented; other subpackages not started.
