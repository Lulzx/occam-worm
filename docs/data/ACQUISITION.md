# Data acquisition and source registry

> [Spec index](../README.md) · Implements [OW-001](../planning/tickets/OW-001.md) · Policy: [§2.2](SOURCES.md)

Every upstream dataset and code repository the project uses is listed in [`configs/datasets/sources.json`](../../configs/datasets/sources.json), with its location, expected checksums, license evidence, the milestones that need it, and when the metadata was retrieved. Raw files are downloaded into `data/raw/<source-id>/`, checked against the registry, and never modified.

## Commands

The tool is stdlib-only Python (3.11+). Run it from anywhere inside the repository:

```bash
python -m occamworm.sources list --all
```

```bash
python -m occamworm.sources fetch --milestone M0 --dry-run
```

```bash
python -m occamworm.sources fetch --milestone M0
```

```bash
python -m occamworm.sources verify --milestone M0 --json artifacts/sources-verify.json
```

| Command | What it does |
| --- | --- |
| `list` | Sources, asset counts, sizes, licenses and milestones. |
| `fetch` | Downloads missing assets to a `.part` file, checks size and checksum while streaming, then moves it into place. Writes `data/raw/<source>/_retrieval.json` (URL, time, size, sha256, md5, tool version). An existing file that does not match is reported, never overwritten. |
| `verify` | Hashes every local asset and reports **missing**, **checksum mismatch**, **unpinned** (no expected checksum yet), untracked files, and license issues. Exits 1 on any integrity problem; `--require-license` also fails on unlicensed, unknown or copyleft sources. `--quick` compares sizes only. |
| `pin` | Records sha256 for downloaded assets whose upstream gives only md5 (Zenodo) or nothing (GitHub tarballs). Refuses if the file contradicts a known md5 or size. This is trust on first use, so it is a deliberate, reviewable change to the registry. |
| `refresh` | Rebuilds asset lists and license evidence from the OSF, Zenodo and GitHub APIs. New assets are added. Changed checksums, removed assets, moved git refs and license changes are reported and only applied with `--accept-upstream-changes`. |

Selection flags work for every command: `--source ID` (repeatable), `--milestone M0`, `--all` (includes superseded sources). With no flag, the default set is used; it equals the M0 set.

`scripts/acquire_sources.py [selection]` runs `fetch` and then `verify`.

## Integrity anchors by provider

| Provider | Upstream checksum | How we pin |
| --- | --- | --- |
| OSF | sha256 and md5 per file, from the API | Recorded directly at refresh |
| Zenodo | md5 per file | md5 checked on download; sha256 recorded by `pin` |
| GitHub | Commit SHA (content-addressed) | Tarball of the pinned commit; sha256 recorded by `pin` after the first download |

GitHub does not guarantee byte-stable tarballs. If a re-download of the same commit ever fails verification, re-pin only after confirming the commit is unchanged.

## Findings (2026-10-08)

**Licenses.** Most sources have no license or a copyleft one. `verify` prints this every time.

| Source | License | Consequence |
| --- | --- | --- |
| Randi et al. 2023 atlas (OSF e2syt), all files | none declared | Treat as all rights reserved. Do not redistribute raw or derived data; reference by DOI and checksum. |
| Atanas & Kim 2023 (Zenodo 19388374) | none declared | Same. |
| worm-functional-connectivity | none | Same; reference only. |
| pumpprobe, wormdatamodel, wormbrain, wormneuroatlas | GPL-3.0 | Use as external reference tools. Do not vendor them, or import them from Apache-2.0 `occamworm` code, without a decision on [open decision 12](../planning/OPEN_DECISIONS.md). OW-002 should read the atlas files directly and use pumpprobe only to cross-check. |
| OpenWorm ConnectomeToolbox | MIT (code) | Bundled datasets may carry their own terms. |
| BAAIWorm | Apache-2.0 | No restriction for reference use. |

**Atlas layout.** The OSF project holds three forms of the data (per its wiki):

- `raw_extracted_data/`: 113 recordings, matching the 113 animals in the paper, each with `<n>_gcamp.txt`, `<n>_t.txt`, `<n>_labels.txt`, `<n>_stim_neurons.txt`, `<n>_stim_volume_i.txt` and `<n>_ds_name.txt` (907 MB). Plain text, so it is the likely primary input for a GPL-free trial-level importer.
- `exported_data.tar.gz` (wild type, 523 MB) and `exported_data_unc31.tar.gz` (85 MB): plain-text basic results.
- `exported_data_full.tar.gz` (1.1 GB): pumpprobe format with fits; needs `pumpprobe/scripts/adjust_paths.py` after extraction.
- Older `*_pre_review` exports are registered as superseded and are never fetched by default.

**OSF API quirk.** Folder listings repeat entries across pages unless sorted; the refresher sorts by name and deduplicates by path.

## Download sizes

| Selection | Size |
| --- | --- |
| Default / `--milestone M0` | 2.6 GB of OSF data plus 63 MB of GitHub tarballs (685 assets) |
| `--milestone M6` | 85 MB (unc-31) + 569 MB (Atanas & Kim) |
| `--all` | 4.2 GB with known sizes, plus GitHub tarballs |
