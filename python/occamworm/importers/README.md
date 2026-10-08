# `occamworm.importers`

Source importers producing normalized Arrow/Parquet tables.

**Invariant:** Animal and trial IDs and provenance are preserved; unobserved pairs are never labeled non-responses; upstream code is never imported or executed.

| Module | Role |
| --- | --- |
| `randi2023.py` | Randi et al. 2023 atlas, wild type: text export + full export → `data/normalized/randi2023-wt-v1/` (OW-002). |
| `randi2023_roundtrip.py` | Independent exact round-trip check of the normalized tables against the upstream files. |
| `_safe_pickle.py` | Allowlist unpickler for upstream metadata pickles. |
| `openworm.py` | Readers for the OpenWorm ConnectomeToolbox tarball: adjacency caches, `Cells.py` lists (parsed statically), cell and transmitter tables (OW-013). |
| `wormneuroatlas.py` | Readers for the data files of the GPL-3.0 Worm Neuro Atlas tarball; no code from it is used (OW-013). |
| `aliases.py` | Explicit versioned alias table and label resolution. |
| `annotation_neurons.py`, `annotation_edges.py` | Neuron evidence and conflicts; typed edges per reconstruction. |
| `annotations.py` | Builds `data/normalized/annotations-v1/` and the atlas label audit (OW-013). |
| `_archive.py`, `_xlsx.py` | Stream members of a tarball; minimal stdlib `.xlsx` reader. |
| `__main__.py` | `python -m occamworm.importers randi2023 {import,roundtrip}` and `annotations build`. |

**Planned:** `wormwideweb.py` (external validation).

Guides: [RANDI2023_IMPORT.md](../../../docs/data/RANDI2023_IMPORT.md), [ANNOTATIONS.md](../../../docs/data/ANNOTATIONS.md) · Tickets: [OW-002](../../../docs/planning/tickets/OW-002.md), [OW-013](../../../docs/planning/tickets/OW-013.md) · Spec: [§2.4](../../../docs/data/DATA_CONTRACT.md), [§14.4](../../../docs/data/DATA_CONTRACT.md)
