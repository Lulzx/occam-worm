# Baselines, shared-timescale test and indicator-kinetics control

> Part of the **Occam's Worm specification v0.4.0** · [Spec index](../README.md) · Source: §4

## 4. Baseline first: shared-timescale experiment

Before unconstrained program search, perform a narrow and highly interpretable test of the central simplifying assumption.

### 4.1 Candidate models

For stimulated neuron `j`, and responder `i`, consider a stimulus-convolved response:

$$
\hat y_{ij}(t)=b_{ij}+a_{ij}\,(u_j*k_{ij})(t).
$$

In T2a, `u_j` is the measured autoresponse of the stimulated neuron. In T2b, it is the nominal stimulus waveform passed through a fitted stimulation-efficacy model. In both cases the indicator stage of [§3.4](PROBLEM_STATEMENT.md) is represented explicitly (§4.5).

**B0 — Null/sham:** no causal stimulus response beyond baseline, drift and noise.

**B1 — Shared timescale / shared latent kernel:**

$$
\hat y_{ij}(t)=b_{ij}+a_{ij}\,(u_j*h_j)(t-\delta_{ij}),
$$

where `h_j` is shared among responders of one stimulus target, with limited or zero per-pair delays (separate preregistered variants). More stringent version: a small *global* bank of shared kernels across stimulus targets.

**B2 — Low-rank kernel bank:**

$$
k_{ij}(t)=\sum_{m=1}^{K} a_{ijm}h_m(t), \quad K\in\{1,2,4,8\},
$$

with all kernels regularized and causal. This explicitly measures whether a small set of temporal primitives suffices.

**B3 — Independent pair kernels:** each `(i,j)` has an independently parameterized stable response kernel, with matched shrinkage and trial-level uncertainty.

**B4 — Linear network dynamics:**

$$
\dot x=(-D+W)x+Bu(t),\quad \hat y=\mathcal H(Cx),
$$

with anatomical support and regularized signs/weights. Compare both fixed anatomically-derived and learned constrained `W`.

**B5 — Conventional simple neuron network:** leaky-rate units with saturating transfer and fixed topology; optional adaptation.

**B6 — Shared latent stochastic state model:** one/few network-scale factors with state-space dynamics and observed target input, to exclude trivial low-rank-factor explanations unrelated to anatomy.

**History variants (all of B0–B6).** Each baseline is fitted twice: without history, and with a declared stimulation-history term, for example a per-animal gain that changes with stimulation index, an exponentially weighted sum of recent autoresponses, or carryover of the previous trial's responder state. The history form is chosen from training-fold audits and frozen before outer scoring. If the history variant does not improve inner-validation scores, report that and keep the simpler variant.

B1 is a restricted case of B3 under compatible parameterizations, but their predictive comparisons are only meaningful when fitted with **equally legitimate regularization/optimization**. Differences in flexibility and stimulus target coverage must be reported.

### 4.2 Data eligibility

Predefine minimum requirements from the real audit rather than inventing sample counts. A stimulus target enters primary evaluation if:

1. Multiple identified responders have valid time series in multiple animals.
2. Under the frozen split scheme ([§2.3](../data/AUDIT_GATE.md)), the target appears in at least one outer test fold and in inner validation folds, with independent animal-level observations.
3. There is sufficient temporal resolution to distinguish the candidate kernel families.
4. Controls/near-null responses are included rather than post-selected away.

Publish the full flowchart of excluded targets and reasons. If fewer targets meet criteria than expected, narrow claims rather than relaxing splits.

### 4.3 Primary statistic

For each held-out animal, score the **joint collection** of eligible observed traces using a predictive log density with masks and calibrated noise. Report:

$$
\Delta\mathrm{NLL}_{a}=\mathrm{NLL}_{a,baseline}-\mathrm{NLL}_{a,candidate},
$$

with positive improvement preferred. Also report per-target results and a hierarchical animal-level confidence interval. Report T2a and T2b separately; T2a is the primary statistic.

Do not use mere Pearson correlation as the main metric; it ignores scale, sign, baseline bias and predictive uncertainty.

### 4.4 Exit gate

Proceed from B1–B6 to program search only after:

- A fully frozen animal-wise benchmark exists.
- A competent baseline can reproduce qualitatively correct known responses and near-null cases.
- Held-out uncertainty is quantified.
- At least one hypothesis about temporal sharing remains plausible and falsifiable.
- The indicator-kinetics control (§4.5) has been run and its outcome recorded.
- History and no-history variants of the baselines have been compared.

A null result is permitted: if independent kernels dominate predictively after fair complexity control, the next program grammar must admit greater neuron/pair-specific state rather than pretending that all responses share one clock.

### 4.5 Indicator-kinetics control

A shared response kernel can be produced by the measurement rather than the neurons. How it happens depends on the task.

In T2b, and in any model driven by the nominal stimulus, every predicted trace passes through the same indicator filter, so the indicator's impulse response appears directly in every fitted kernel and makes kernels look alike.

In T2a, if the indicator were linear and time-invariant, its filter would cancel between the measured autoresponse (input) and the responder trace (output), and the fitted kernel would estimate the neural transfer. The cancellation is incomplete in practice. Indicator saturation and other nonlinearities break it. And because the indicator low-pass filters both input and output, kernel components faster than the indicator are buried in noise; regularization then shrinks those unresolvable components toward a common shape, which can look like shared timescales.

The control is preregistered before outer scoring:

1. Estimate the indicator impulse response (and saturation, if modeled) separately from neural kernels, using training-fold autoresponses and published sensor kinetics where available.
2. From the indicator model and the measured noise level, compute the resolvable kernel bandwidth and report it.
3. Fit B1–B3 with an explicit indicator stage and report the shared kernels' timescales alongside the indicator's.
4. Generate synthetic data with known, distinct neural kernels behind the estimated indicator and noise, run the full pipeline, and report how often it wrongly favors B1 or B2 (the false-sharing rate).

**Interpretation rule.** H1 is supported as a claim about neural dynamics only if (a) in T2b and other nominal-stimulus models, the favored shared kernels differ from the indicator impulse response beyond its uncertainty, and (b) in all tasks, the false-sharing rate is low at the measured noise level, using a threshold fixed in [Appendix B](../planning/OPEN_DECISIONS.md) before outer scoring. Otherwise the result is reported as: shared kernels are not distinguishable from measurement effects at the available resolution.
