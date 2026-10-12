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
- **Decision 3 (provisional, OW-005/OW-006; evidence and recommendation proposed 2026-10-12, not frozen).** Gaussian AR(1) residuals with σ² = a²s² + b², fitted on training residuals only; on the full T2a benchmark the 50% and 95% intervals cover 62% and 96%. A Student-t AR(1) option now exists (`fit_noise(..., family="student_t_ar1")`: same scale a²s² + b², i.i.d. t innovations on the standardized AR(1) residual, ν fitted with exact gradients; Gaussian stays the default). Tested on inner folds only (outer folds 0–4, T2a, no history, B0/B1/B3 at their inner-selected ridge; each noise model fitted on inner-train residuals, scored on the 5 inner-validation sets, 545 animal-fold pairs; `python -m occamworm.baselines.noise_evidence`): ΔNLL (t − Gaussian, negative = t better) is −4.02e6 (B0), −4.01e6 (B1), −4.01e6 (B3), i.e. −0.134 nats per sample, with t better in 537/545 (B0) and 535/545 (B1, B3) animal-fold pairs and the gain the same in every outer fold (−0.131 to −0.137/sample). Coverage at 50/80/95%: Gaussian 63/89/97 (B0), 62/88/96 (B1, B3); t 50/82/96 (all). ν = 5.1–5.9 (mean 5.5), AR φ 0.35–0.38 (t) against 0.40–0.45 (Gaussian). The family ranking does not change, since ΔNLL per sample is the same for B0, B1, B3 to within 0.0003. Tails: 12% of validation traces contain any |z| > 3 sample, and 97% of those samples sit in the top 10% of traces (47% in the top 1%); the stimulation target URYDL has a 4–6% rate against a 1% median across targets; animal rates have median 0.8% and maximum 4.7–6.6%. Not an onset artifact: the |z| > 3 rate is 0.3% at volumes 2–3 (Gaussian 0.27%), 0.5% at 4–5, 0.7% at 6–9, then flat at 1.2–1.3% for volumes 10–59. No sign that the noise model absorbs response structure: lag-1 autocorrelation of demeaned 9-volume residual segments is −0.32 (volumes 2–10), −0.30 (20–28), −0.33 (40–48), against −0.29 for the raw pre-stimulus baseline of the same animals. **Proposed:** use Student-t AR(1) as the primary observation model for the M4 confirmatory search and keep Gaussian as a reported sensitivity row; ν is fitted on training residuals, no test-time calibration. Caveats: the tails look like unmodelled large responses of a few targets and traces rather than measurement noise, so heavy tails are partly model misfit that richer rule families may shrink; ν should be refit per outer fold, not frozen.
- **Decision 5 (provisional, OW-007/OW-012).** B4 and the WRL candidates use `cook2019-herm`, with chemical signs from `fenyves2020-sign-prediction`. WRL candidates treat edges whose sign is unknown or missing as excitatory and count them in each selection's metadata. Divergent reconstructions are not compared yet.
- **Decision 13 (frozen in the experiment config, OW-006).** Both variants are fitted, with no history term and with pre-stimulus slope plus stimulation index. Per outer fold, the variant with the better inner-validation score is selected, and both are reported.
- **Decision 14 (provisional, OW-006; revised 2026-10-12).** The T2b indicator is fixed at published GCaMP6s kinetics: τr = 0.4 s (0.15–1.0) and τd = 2.5 s (1.0–8.4). The ranges are the §4.5 uncertainty. No direct measurement exists for nuclear GCaMP6s in *C. elegans*, so the values are extrapolated from Chen 2013, Helassa 2016 and Ding 2023 (sources in `configs/experiments/shared-timescales.toml`). The earlier per-fold fit to autoresponses came out at the alpha limit with τ ≈ 11.5 s. That is the whole stimulus-to-calcium response, not the sensor, so it is now reported only as a diagnostic. A false-sharing rate ≤ 0.05 counts as low.
