# Open decisions to freeze before confirmatory evaluation

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: Appendix B

Record each decision, its date and rationale in the experiment registry before any outer fold is scored.

## Appendix B — Open decisions to freeze before confirmatory evaluation

1. Which raw atlas releases have actual trace-level accessibility, and their licenses?
2. Which stimulus-target cohort has sufficient animal-level replication after QC?
3. What observation operator and correlated-noise model fit training controls without absorbing neural dynamics?
4. What is the exact allowed calibration information for a never-before-seen animal, and which task (T2a or T2b) is primary for each paper?
5. Which connectome version is canonical and how are divergent anatomical reconstructions compared?
6. Which G1 operators and AST budget will be frozen before the first outer fold is inspected?
7. What is the exact practical noninferiority margin for calibration/null-response metrics?
8. What statistical unit is used in primary analysis if multiple animals share acquisition batches?
9. Which second dataset or perturbation can provide truly independent validation?
10. What evidence level is required to call a mechanism 'neural', 'extrasynaptic', or 'cell-specific'?
11. Which body simulator allows a fixed neuro-muscular interface and meaningful licensed redistribution?
12. How will data availability and source-code license incompatibilities be handled in public releases?
13. Which stimulation-history term, if any, is used, chosen from training-fold audits only?
14. Which indicator model (impulse response, saturation) and resolvable-bandwidth estimate are frozen for [§4.5](../science/BASELINES.md), and what false-sharing rate counts as low?
15. Which split scheme is chosen from the coverage tables, and on what grounds?

## Evidence gathered so far

- **Decisions 1 and 12 (2026-10-08, OW-001).** Trace-level files are publicly accessible on OSF (`raw_extracted_data/`, 113 recordings), but the OSF project declares no license, nor does the Atanas & Kim Zenodo record. pumpprobe, wormdatamodel, wormbrain and wormneuroatlas are GPL-3.0; worm-functional-connectivity has no license. Details: [ACQUISITION.md](../data/ACQUISITION.md#findings-2026-10-08). Not yet decided: whether to ask the authors for explicit terms, and how GPL reference code may be used alongside Apache-2.0 code.

- **Decision 15 (frozen 2026-10-08, OW-004).** Leave-one-animal-out (113 outer × 5 inner folds) covers 144 eligible targets, against 125 for group5 and 127 for group10. Rule and evidence: [EXPERIMENT_REGISTRY.md](../evaluation/EXPERIMENT_REGISTRY.md#registered-entries), [AUDIT_REPORT.md](../data/AUDIT_REPORT.md).
- **Decision 2 (2026-10-08, OW-003).** The audit gate is *go*: 144 covered targets have at least one responder with |t| ≥ 3 in at least 2 animals. Volumes −1, 0 and +1 around stimulation carry an optical artifact and are masked (audit-v2).
- **Decision 3 (provisional, OW-005/OW-006).** Gaussian AR(1) residuals with σ² = a²s² + b², fitted on training residuals only. On development folds the 50% intervals over-cover, so residuals are heavier-tailed than Gaussian. Student-t stays open.
- **Decision 5 (provisional, OW-007/OW-012).** B4 and the WRL candidates use `cook2019-herm`, with chemical signs from `fenyves2020-sign-prediction`. WRL candidates treat edges whose sign is unknown or missing as excitatory and count them in each selection's metadata. Divergent reconstructions are not compared yet.
- **Decision 13 (frozen in the experiment config, OW-006).** Both variants are fitted, with no history term and with pre-stimulus slope plus stimulation index. Per outer fold, the variant with the better inner-validation score is selected, and both are reported.
- **Decision 14 (provisional, OW-006).** The indicator is a difference of exponentials fitted per outer fold from training autoresponses. A false-sharing rate ≤ 0.05 counts as low. The fitted indicator is slow (τ ≈ 11.5 s), so the resolvable bandwidth is an upper bound.
