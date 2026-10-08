# `occamworm.importers`

Source importers producing normalized Arrow/Parquet tables.

**Invariant:** Animal and trial IDs and provenance are preserved; unobserved pairs are never labeled non-responses; upstream code is never imported or executed.

| Module | Role |
| --- | --- |
| `randi2023.py` | Randi et al. 2023 atlas, wild type: text export + full export → `data/normalized/randi2023-wt-v1/` (OW-002). |
| `randi2023_roundtrip.py` | Independent exact round-trip check of the normalized tables against the upstream files. |
| `_safe_pickle.py` | Allowlist unpickler for upstream metadata pickles. |
| `__main__.py` | `python -m occamworm.importers randi2023 {import,roundtrip}`. |

**Planned:** `wormneuroatlas.py` and `openworm.py` (OW-013), `wormwideweb.py` (external validation).

Guide: [docs/data/RANDI2023_IMPORT.md](../../../docs/data/RANDI2023_IMPORT.md) · Tickets: [OW-002](../../../docs/planning/tickets/OW-002.md), [OW-013](../../../docs/planning/tickets/OW-013.md) · Spec: [§2.4](../../../docs/data/DATA_CONTRACT.md), [§14.4](../../../docs/data/DATA_CONTRACT.md)
