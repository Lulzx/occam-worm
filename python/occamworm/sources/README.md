# `occamworm.sources`

Source registry, acquisition and verification (OW-001). Stdlib only, so it runs in a clean environment before anything else is installed.

| Module | Role |
| --- | --- |
| `registry.py` | Load, validate and write `configs/datasets/sources.json` (stable, diff-friendly format). |
| `fetch.py` | Streamed download to `.part`, hash while streaming, check, atomic move; `_retrieval.json` records. |
| `verify.py` | Integrity and license report; JSON output for the M0 audit (OW-003). |
| `pin.py` | Record sha256 for md5-only or unchecksummed assets after first download. |
| `remote.py` | Refresh asset lists and license evidence from OSF, Zenodo and GitHub; upstream changes need explicit acceptance. |
| `cli.py` | `python -m occamworm.sources {list,fetch,verify,pin,refresh}`. |

**Invariant:** raw bytes are never modified, and a pinned checksum is never changed silently.

Guide: [docs/data/ACQUISITION.md](../../../docs/data/ACQUISITION.md) · Ticket: [OW-001](../../../docs/planning/tickets/OW-001.md)
