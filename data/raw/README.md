# data/raw/

Original downloads, byte for byte, one directory per registry source: `data/raw/<source-id>/<asset path>`. Each directory also holds `_retrieval.json`, the local record of when and from where each file was fetched.

**Invariant:** Immutable. The fetch tool never overwrites a file; a mismatch is reported for a human to resolve.

Populate and check with:

```bash
python -m occamworm.sources fetch --milestone M0
```

```bash
python -m occamworm.sources verify --milestone M0
```

Everything here except this README is gitignored. Most upstream data declares no license, so it must not be committed or redistributed ([ACQUISITION.md](../../docs/data/ACQUISITION.md)).

Tickets: [OW-001](../../docs/planning/tickets/OW-001.md) · Spec: [§2.2](../../docs/data/SOURCES.md)
