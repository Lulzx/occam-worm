# Biological annotations: neurons, aliases, typed edges, conflicts

> [Spec index](../README.md) · Implements [OW-013](../planning/tickets/OW-013.md) · Spec: [§3.2](../science/PROBLEM_STATEMENT.md) · Contract: [§14.4](DATA_CONTRACT.md) · Sources: [ACQUISITION.md](ACQUISITION.md)

```bash
python -m occamworm.importers annotations build
```

The command re-verifies both input sources against the registry, reads the atlas ROI labels from `data/normalized/randi2023-wt-v1/neurons.parquet` (OW-002) and writes `data/normalized/annotations-v1/`. It refuses to write into a non-empty directory. Options: `--data-dir` (raw data), `--atlas-dir`, `--out`.

## Rules

1. **The neuron set is the 302 hermaphrodite neurons of the OpenWorm Toolbox** (`PREFERRED_HERM_NEURON_NAMES`, e.g. `AVAL`, `VA1`).
2. **Nothing is normalized silently.** A label is resolved only if it is canonical or listed in the alias table with a kind, a reason and a version. Labels no rule covers stay unresolved.
3. **Disagreement is kept, not resolved.** Every attribute value is stored with its source. Where sources differ, all values stay and `conflicts.parquet` records the difference.
4. **Reconstructions are never merged.** Each source reconstruction contributes its own edge rows.
5. **Sign is `unknown` unless a dataset states it for that edge** and is never derived from the presynaptic cell name (spec §3.2).

## Inputs

| Source (registry id) | License | Used for |
| --- | --- | --- |
| `openworm-connectome-toolbox` v0.3.5 | MIT (code); bundled datasets keep their own terms | JSON adjacency caches of the Toolbox readers (White, Durbin, Varshney, Cook 2019/2020, Witvliet), the 302-neuron and muscle lists from `Cells.py` (parsed with `ast`, never imported), WormAtlas cell types, the Wang 2024 transmitter atlas, Ripoll-Sanchez 2023 neuron table and neuropeptide matrices, Bentley 2016 edge lists |
| `wormneuroatlas` b2e13d88b670 | **GPL-3.0** code; data files from CeNGEN, WormBase, Beets, Bentley, Fenyves, White, Witvliet | Data files only: Fenyves 2020 workbook (transmitters, receptors, sign predictions), ganglia, neuron positions, cell descriptions, White and Witvliet CSVs, Bentley CSVs |
| `randi2023-wt-v1` (OW-002 output) | none declared upstream | Atlas ROI labels for the audit |

**License boundary.** No code from `wormneuroatlas` is imported, vendored or copied; its data files are parsed by readers written here (`openworm.py`, `wormneuroatlas.py`, `_xlsx.py`). The tarballs are streamed and nothing is extracted. Bundled datasets keep the terms of their publications (the Fenyves and Bentley papers are PLOS, CC BY; others are unchecked, see open issues). The outputs are local and gitignored.

## What each source contains

| Source file | Content | Notes |
| --- | --- | --- |
| Toolbox `cect/cache/*.json` | Square matrices, rows presynaptic, per synapse class (`Generic_CS`, `Generic_GJ`) | Written by the Toolbox readers; cell names are already the Toolbox's preferred names. Gap matrices are symmetric. |
| Toolbox `Cells.py` | Cook 2019 categories (sensory, inter, motor, pharyngeal polymodal, unknown), pharyngeal neurons, muscle names | Static evaluation of list literals |
| Toolbox `all_cell_info.csv` | WormAtlas type per cell | |
| Toolbox Wang 2024 supp. file 2 | Neurotransmitter expression and call per neuron, class | The class cell is merged upstream; values are filled down and noted |
| Toolbox Ripoll-Sanchez table S7 | Type and body segment | `Pharynx` is read as pharyngeal membership, not as a function |
| Toolbox Ripoll-Sanchez CSVs, Bentley edge lists | Extrasynaptic (putative modulatory) routes | Weights count peptide-GPCR or transmitter-receptor pathways, not synapses |
| WNA Fenyves 2020 S3 | Dominant and alternative transmitter, ionotropic receptor genes, neuron type, **predicted sign of 3,638 chemical edges** | The sign is a prediction from transmitter plus receptor expression |
| WNA ganglia JSON, positions, lineage | Region per neuron; coordinates (origin and units undocumented upstream); descriptions of neurons and non-neuron cells | `AWCON`/`AWCOFF` used as node ids |
| WNA White 1986 and Witvliet 7/8 CSVs | The same datasets as the Toolbox, bundled separately | Compared edge by edge |

## Mapping rules (`mapping_version = ow013-aliases-v1`)

| Kind | Rule | Resolves? |
| --- | --- | --- |
| exact | Label is a canonical name | yes |
| `zero_padded` | `VA01` to `VA1` (Fenyves, Ripoll-Sanchez CSVs) | yes |
| `case_variant` | Differs from exactly one canonical name only by case (`Il2DR`) | yes |
| `class_label` | A class or stem (`AVA`, `IL2V`, `DB`): one row per candidate member, from the Wang class column and from the observed labels | no, ambiguous |
| `convention` | `AWCON`/`AWCOFF` (a per-animal asymmetric state, not a side), `DB1/3` and `DB3/1` (Wang pair labels) | no, ambiguous |

Typos are never aliases. For an unresolved atlas label, `unresolved.parquet` carries `suggested_neuron_id` when exactly one canonical name is one edit away (`NMSL` suggests `NSML`); it is a hint only and does not resolve the label. Labels of known non-neuron cells (socket, sheath, glia, from the WNA descriptions) are reported as such.

A source row labelled with an ambiguous alias is attached to every candidate and flagged `ambiguous_assignment` in `neuron_attributes.parquet`.

Muscles use a separate small table (`muscle_aliases.parquet`, `ow013-muscle-aliases-v1`): `BWM-DL01` to `MDL01` (Witvliet naming in the WNA CSVs), `LegacyBodyWallMuscles` to `BWM`, `pm4` to `pm4_UNSPECIFIED`.

## Outputs (`data/normalized/annotations-v1/`)

| File | Grain | Notes |
| --- | --- | --- |
| `neurons.parquet` | neuron (302) | `cell_class`, `region`, `body_segment` (null if the sources differ); for `cell_class`, `region`, `body_segment`, `neuron_type`, `is_pharyngeal`, `transmitter`, `position_xyz` the columns `<attr>_values`, `<attr>_sources` (`value@source`) and `<attr>_conflict`; receptor genes, coordinates (only if unique), `has_transmitter_assignment`, `n_anatomical_reconstructions`, `atlas_n_recordings`, `atlas_n_rois`, `n_conflicts` |
| `neuron_attributes.parquet` | evidence row | `neuron_id`, `attribute`, `value`, `role` (reported, dominant, alternative, monoamine_source, expressed), `source`, `source_member`, `raw_value`, `raw_label`, `via_alias`, `ambiguous_assignment`, `note` |
| `aliases.parquet` | alias × candidate | `alias`, `canonical_neuron_id`, `kind`, `ambiguous`, `n_candidates`, `reason`, `observed_in`, `mapping_version` |
| `muscle_aliases.parquet` | alias | as above for muscles |
| `edges.parquet` | edge × reconstruction | Exactly the §14.4 columns |
| `edge_details.parquet` | same rows | `edge_row` (position in `edges.parquet`), `native_weight`, `weight_semantics`, `synclass`, `transmitters`, `receptors`, `n_raw_rows`, raw labels, `symmetrised`, `sign_raw`, `sign_basis`, `target_kind` |
| `conflicts.parquet` | conflict | `entity_type` (neuron, edge), `entity_id`, `attribute`, `conflict_kind`, `severity` (hard: sources disagree; soft: partial overlap or one source lists several values), `values`, `sources`, `raw_values`, `note` |
| `unresolved.parquet` | distinct atlas label | Every label with status `name`: ROI and recording counts, duplicates, resolution (exact, alias, ambiguous, unresolved), candidates, kind, reason, suggestion |
| `manifest.json` | — | Input ids and digests, code revision, sha256 and rows per table, summary, reconstructions, cross-checks, intersection |

### Edges

- `edge_kind`: `chem`, `gap`, `nmj` (neuron to muscle, chemical or gap), `putative_mod` (Bentley, Ripoll-Sanchez). `target_neuron_id` is a muscle name for `nmj` rows (`target_kind = muscle`). The muscle-to-neuron mirror of a neuromuscular gap junction is dropped.
- **Gap junctions are symmetric:** each junction has one directed row per direction with the same count. A bundle that lists one direction only gets the reverse row added and flagged `symmetrised`.
- `synapse_count` is the reconstruction's native count (float32). It is null for `putative_mod`; the native pathway count is in `edge_details.native_weight`. Units differ between reconstructions and are not comparable without care.
- `sign`: `unknown` for `chem`, `nmj`, `putative_mod`; null for `gap` (not applicable). The one exception is `fenyves2020-sign-prediction`: `+` becomes `excitatory`, `-` becomes `inhibitory`, `complex` and `no pred` stay `unknown`. `sign_basis` and `sign_raw` record that this is a published prediction.
- `confidence` is null: no source provides a per-edge confidence.
- Edges to other cells (glia, hypodermis, epithelium, gland and marginal cells) and edges whose endpoint label is ambiguous or unknown are dropped and counted per reconstruction in the manifest (`dropped`, `dropped_top_labels`).

### Reconstructions (real counts, 2026-10-08)

Rows in `edges.parquet`; every row is one directed edge.

| Reconstruction | Life stage | chem | gap | nmj | putative_mod |
| --- | --- | ---: | ---: | ---: | ---: |
| `white1986-whole` | adult (N2U) + L4 (JSH) | 2,266 | 1,132 | 120 | |
| `white1986-n2u` | adult | 1,433 | 544 | 39 | |
| `white1986-jsh` | L4 | 1,293 | 580 | 41 | |
| `varshney2011` | adult + L4 | 2,194 | 1,031 | 115 | |
| `cook2019-herm` | adult | 3,709 | 2,196 | 1,063 | |
| `cook2020-pharynx` | adult | 149 | 58 | 80 | |
| `witvliet2021-1` .. `-4` | L1 | 675 / 865 / 887 / 1,011 | 164 / 240 / 186 / 407 | 78 / 91 / 100 / 111 | |
| `witvliet2021-5` | L2 | 1,324 | 572 | 148 | |
| `witvliet2021-6` | L3 | 1,315 | 422 | 169 | |
| `witvliet2021-7` | adult | 1,944 | 528 | 226 | |
| `witvliet2021-8` | adult | 1,933 | 600 | 216 | |
| `white1986-whole:wna` | adult + L4 | 2,266 | 1,132 | 120 | |
| `witvliet2021-7:wna` | adult | 1,944 | 530 | 226 | |
| `witvliet2021-8:wna` | adult | 1,936 | 600 | 218 | |
| `fenyves2020-sign-prediction` | not stated | 3,638 | | | |
| `bentley2016-monoamine` | not stated | | | | 1,952 |
| `bentley2016-neuropeptide` | not stated | | | | 7,078 |
| `ripoll-sanchez2023-short` / `mid` / `long` | adult | | | | 31,417 / 40,425 / 53,558 |
| **Total** | | 30,782 | 10,922 | 3,161 | 134,430 |

Total 179,295 rows. Signs: 1,327 chem rows `excitatory` and 425 `inhibitory` (all from Fenyves), 29,030 chem `unknown`, 10,922 gap null, 3,161 nmj and 134,430 modulatory `unknown`. 1,121 gap rows were added to restore symmetry (the WNA White file lists one direction).

`white1986-whole`, `varshney2011` and the two Durbin series overlap by construction (Varshney builds on JSH and N2U; `white1986-whole` is effectively Varshney plus pharynx). They are listed as the Toolbox lists them and are not independent evidence.

## Neurons, conflicts and cross-checks (real counts)

- 302 neurons; 5,708 attribute rows; 400 alias rows (184 distinct aliases: 64 `zero_padded`, 1 `case_variant`, 327 `class_label` rows, 8 `convention` rows; 119 aliases are ambiguous); 34 muscle aliases.
- **Neuron conflicts: 73.** `neuron_type`: 14 hard (disjoint type sets, e.g. the IL1 neurons are sensory in Cook and WormAtlas, motor in Ripoll-Sanchez and Fenyves) and 51 soft (overlap but differ, e.g. Fenyves lists several functions); `is_pharyngeal`: 2 hard (MCL, MCR: Fenyves types them as interneurons, the other four sources place them in the pharynx); `transmitter`: 6 soft (a monoamine in the Bentley list, not in the Wang or Fenyves call: ADFL/ADFR serotonin, HSNL/HSNR serotonin, RIML/RIMR tyramine). No classical transmitter (ACh, Glu, GABA) disagrees between Wang 2024 and Fenyves 2020 in this version.
- **Edge conflicts: 81**, all between the Toolbox and WNA bundles of the same dataset: 60 weight differences and 21 edges present in one bundle only, all in Witvliet 7 and 8 (White agrees on all 3,518 edges). Missing sign information is not a conflict.
- Fenyves versus Cook 2019 (chemical): 3,516 edges in both, 3,495 with equal weight, 122 only in Fenyves, 193 only in Cook. Fenyves' edge set is an older Wormwiring version of the Cook data, and its life stage is not stated.
- Bentley 2016 monoamine (2,626 pairs) and neuropeptide (8,931 pairs) lists are identical in both bundles.
- WNA neuron ids: 300; 298 resolve, `AWCON` and `AWCOFF` are ambiguous (sides unknown), `CANL` and `CANR` are absent.
- Fenyves polarity labels over its 3,638 edges: 1,327 `+`, 425 `-`, 471 `complex`, 1,415 `no pred`.

## Atlas label audit (real counts)

`unresolved.parquet` covers all 309 distinct labels with status `name` (7,760 ROIs); 5,835 blank, 38 numeric and 788 other-status ROIs (`merge`, `target`, notes) are not audited.

| Resolution | Labels | ROIs |
| --- | ---: | ---: |
| exact | 206 | 7,111 |
| alias (`Il2DR`) | 1 | 1 |
| ambiguous | 74 | 542 |
| unresolved | 28 | 106 |

Ambiguous ROIs: 457 with left/right unspecified (`AVA`, `RIG`, ...), 38 `AWCON`/`AWCOFF`, 22 with class labels of several members (`IL1`, `SMB`, `M`), 16 with dorsal/ventral and left/right unspecified, 9 numbered-series labels (`DB`, `VD`). Unresolved ROIs: 42 known non-neuron cells (`AMsoL`, `AMsoR`, case variants), 38 class-level labels of non-neuron cells (`AMso`), 26 ROIs with 21 unknown names (typos such as `NMSL`, `URVYL`, `QLQVL`; 10 labels have a single suggestion). Counts are of ROIs, not of recordings.

## Three-way intersection (for the OW-003 audit, §2.3)

| Set | Neurons |
| --- | ---: |
| Functional atlas (an ROI label unique in its recording that resolves exactly or by a non-ambiguous alias) | 206 |
| Anatomical graph (chem or gap edge in any reconstruction) | 302 |
| Molecular annotation (a transmitter other than unknown or uptake-only, any source) | 285 |
| Functional and anatomical | 206 |
| Functional and molecular | 194 |
| **Functional, anatomical and molecular** | **194** |

Per reconstruction, functional neurons with edges: Cook 2019 206, `white1986-whole` 206, Witvliet adult (7, 8) 159 each, Varshney 187. The 12 functional neurons without a transmitter assignment are listed in the manifest (`functional_without_molecular`: ASIL, ASIR, AVHL, AVHR, AVKL, AVKR, I6, PVWL, PVWR, RID, RMGL, RMGR). The count of 206 treats class-level labels as unresolved; it grows if OW-003 decides that unspecified sides may be assigned.

## Open issues

- **Unresolved sides.** 542 ROIs carry class-level labels. Whether a side can be inferred (for example from stimulation position) is an OW-003 decision, not made here.
- **Typo suggestions** (10 labels) are unreviewed; none is applied.
- **Sign.** Only Fenyves predicts sign; 471 edges are `complex` and 1,415 `no pred`. Its edge set is not identical to any Toolbox reconstruction, so a signed overlay on Cook 2019 needs an explicit join (3,516 shared edges) and a decision on the 193 Cook edges without a prediction.
- **Dataset terms.** Terms of the Cook, Witvliet, Varshney, White, Wang, Ripoll-Sanchez and CeNGEN supplementary files have not been reviewed; outputs stay local until they are.
- **Cache fidelity.** Toolbox caches were produced by the Toolbox readers; only White, Witvliet 7/8 and Bentley were compared against a second bundle. The caches were not re-derived from the original spreadsheets.
- **Coordinates.** `anatlas_neuron_positions.txt` has 303 rows for 302 neurons (`PVCL` twice with identical values) and no documented origin or units; use with care.
- **Descriptions.** `cell_lineage.txt` calls `AMsoR` an amphid sheath; this is upstream text and is quoted as found.
- **Not imported:** Toolbox contactomes (Brittin, Yim), Yim 2024 chemical, male datasets, model readers (Gleeson, Olivares, Haspel-O'Donovan; their signs are modelling choices), Wang 2024 edge classes (assigned from the presynaptic cell, so they would give sign by cell name), the WNA `aconnectome.json` per-neuron `chemical_sign`, `funatlas.h5` and `cengen.h5` (HDF5; no h5py dependency).
- **Muscle names** differ across bundles; the alias table covers the cases seen. The WNA label `excgl` (excretory gland) is not in any name list and its 5 chemical edges are dropped as unknown.
