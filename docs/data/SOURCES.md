# Data sources and acquisition policy

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §2.1–§2.2

## 2. Prior work and source datasets


### 2.1 Data assets and what can be claimed

| Source | Primary contents | Planned use | Caveats |
|---|---|---|---|
| Randi et al. 2023 signal-propagation atlas | Optogenetic stimulation, observed neuronal calcium responses, per-recording fits; WT and available mutant conditions | Primary causal training/evaluation | Heterogeneous counts; calcium rather than membrane voltage; most head pairs not strongly and reliably connected |
| Leifer Lab `pumpprobe` | `Funatlas`, `Fconn`, per-animal data interfaces, kernels and occurrence matrices | Reference importer; trace-level audit | Depends on preprocessing and external data layout; do not mistake aggregated kernels for raw trials |
| OpenWorm Connectome Toolbox | Multiple curated anatomical networks, including chemical and electrical connections | Anatomical priors and provenance | Different connectome sources and developmental stages are not interchangeable |
| Worm Neuro Atlas | Transcriptomic, receptor/peptide, transmitter and atlas integrations | Biological type constraints and candidate extrasynaptic edges | Expression implies potential capability, **not** proof of functional communication |
| WormWideWeb / Atanas et al. | Freely moving whole-brain activity and behavioral annotation | External dynamic and behavioral evaluation | Experimental conditions/observables differ from head-fixed stimulation atlas |
| BAAIWorm | Published closed-loop worm brain–body–environment reference | Embodiment baseline and interface reference | It already demonstrates embodied simulation; cannot be presented as a novel milestone by itself |

Links and citations appear in [§22](../overview/REFERENCES.md). [R1–R5]

### 2.2 Source-of-truth acquisition policy

1. Download from the original repository or DOI; capture immutable archive checksums.
2. Record DOI, publication year, upstream commit/tag, retrieval date, license, and data-processing version.
3. Prefer official dataset/analysis libraries to reconstructing responses from paper figures.
4. Preserve all source trial and animal identifiers; never discard them when creating aggregation tables.
5. Maintain both original data and normalized datasets; do not modify original bytes.
6. Mark unknown fields `null` with a machine-readable reason, rather than silently imputing.
7. Never label a pair that was not observed as a negative response.
8. Record whether a response is directly measured, fitted, or inferred from a paper-level statistic.
