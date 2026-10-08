# `occamworm.importers`

Source importers producing normalized Arrow/Parquet tables.

**Invariant:** Animal and trial IDs and provenance are preserved; unobserved pairs are never labeled non-responses.

**Planned contents:**

- `pumpprobe.py` — trial-level atlas importer (Randi et al. 2023)
- `wormneuroatlas.py` — transmitter, receptor and peptide annotations
- `openworm.py` — typed anatomical graphs from the Connectome Toolbox
- `wormwideweb.py` — freely moving activity and behavior (external validation)

Tickets: [OW-002](../../../docs/planning/tickets/OW-002.md), [OW-013](../../../docs/planning/tickets/OW-013.md) · Spec: [§2.2](../../../docs/data/SOURCES.md), [§2.4](../../../docs/data/DATA_CONTRACT.md), [§14.4](../../../docs/data/DATA_CONTRACT.md)

Status: not implemented.
