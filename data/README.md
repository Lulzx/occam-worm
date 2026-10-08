# data/

Local data. Nothing here is committed except these READMEs; large or restricted datasets never enter the repository.

- `raw/` — original downloads, byte-for-byte; immutable, gitignored.
- `normalized/` — Arrow/Parquet tables (`animals`, `trials`, `observations`, `edges`) with versioned manifests.
- `splits/` — frozen animal-wise split IDs and checksums.

Spec: [§2.2](../docs/data/SOURCES.md), [§14.4](../docs/data/DATA_CONTRACT.md)

Status: not implemented.
