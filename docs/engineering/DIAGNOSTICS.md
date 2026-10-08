# Instrumentation, diagnostics and visualization

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §17.1–§17.3

## 17. Instrumentation, diagnostics and visualization


### 17.1 Mandatory plots

- Data coverage heatmaps: animals × stimulation targets × measured responders.
- Histograms of independent repeats, separated by genotype and batch.
- Raw and normalized response traces with QC flags and masks.
- Baseline versus discovered-model held-out response predictions.
- Paired per-animal NLL improvement, not only aggregate means.
- Predicted versus measured peak latency/sign/calibration.
- Program complexity versus performance Pareto frontier.
- Grammar tier ablation and parameter-sharing ablation.
- Counterfactual neuron perturbation comparison.
- Hypothesis-graph uncertainty / intervention-distinguishability heatmap.
- Equivalence class robustness across measurement tolerances.
- Numerical timestep and simulator backend sensitivity.
- Indicator impulse response overlaid on fitted shared kernels, with the resolvable bandwidth marked.
- Response amplitude versus stimulation index within recordings.

### 17.2 Diagnostic trace viewer

Recommended functionality:

- Search by `(animal, trial, stimulated neuron, responder neuron)`.
- Overlay fluorescence, predicted calcium, stimulus pulse and confidence bands.
- Show whether the trace was train/inner-validation/outer-test.
- Show raw-source link, preprocessing transforms and masking rationale.
- Compare any two models using **the same observation operator**.
- Export figure as SVG/PNG and provenance JSON.

No interactive editing of validation labels or folds in a report viewer.

### 17.3 Candidate model explorer

For each candidate display:

- Canonical readable WRL source and AST diagram.
- Complexity bits (`L_struct`, `L_params`, `L_total`) broken down by operator/register/parameter class.
- Training/inner-validation score distributions.
- Biological constraints and violations.
- Numerical stability diagnostics.
- Causal intervention predictions with uncertainty.
- Similar and empirically indistinguishable candidates.
- Parameters whose posterior is broad or confounded.
