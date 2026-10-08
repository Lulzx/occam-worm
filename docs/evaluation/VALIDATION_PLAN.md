# Statistical evaluation protocol

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §11

## 11. Statistical evaluation protocol

### 11.1 Split hierarchy

The minimum biological independence unit is usually **animal**, not individual neuron pairs or time frames. Where multiple sessions belong to one animal, all sessions remain in the same fold. If batch effects are strong, report separate leave-batch-out evaluation and avoid claiming independence across animals from the same acquisition batch without accounting for it.

Required split definitions:

- **Outer folds:** grouped by animal; report group IDs/hashes and target coverage per fold.
- **Inner folds:** grouped by animal within outer-training data.
- **Target holdout:** leave specific stimulated-neuron identities entirely unfit.
- **External test:** independent dataset, acquisition cohort or genotype, evaluated only after rule freeze.
- **Scheme selection:** chosen from the M0 target-by-fold coverage tables before any model is scored ([§2.3](../data/AUDIT_GATE.md)). Leave-one-animal-out is an admissible outer scheme.

Temporal splitting within a single trial is **not** a valid animal-generalization substitute.

### 11.2 Primary metric: predictive negative log likelihood

For observed data and model predictive distribution:

$$
\mathrm{NLL}=-\sum_{a,r,i,k} M_{a,r,i,k}\,
\log p_\theta(Y_{a,r,i,k}\mid G_a,U_{a,r},z_a).
$$

However, time samples within a calcium trace are correlated; naïvely treating them as independent greatly overstates precision. Implement one of:

- Explicit correlated-noise likelihood (e.g., AR residuals / state-space noise).
- Per-trace likelihood with a fitted low-dimensional covariance.
- Block-aggregated proper scoring when covariance modeling is unreliable.

A score must be comparable across candidate and baseline families; report per-observed-trace and per-animal normalizations in addition to totals.

### 11.3 Secondary metrics

- Predictive mean squared/absolute error on standardized fluorescence traces.
- Sign consistency for well-supported response classes.
- Rise time, peak latency, decay and integrated response error, **only** where estimable.
- Calibration of 50/80/95% prediction intervals.
- Null-response false positive rate.
- Posterior predictive distribution of response variability across animals.
- Event-ordering and trial-to-trial covariation metrics.
- Complexity bits, training time and inference cost.

### 11.4 Baseline fairness

- Give every baseline a defensible tuned hyperparameter budget.
- Match observable operator/measurement noise as much as feasible.
- Report both *equal compute* and *best achievable within predefined tuning budgets* comparisons.
- Do not compare a fitted rule search with a deliberately underfit zero-parameter anatomical heuristic and imply scientific superiority.
- Publish ablations showing whether improvements come from the grammar, input features, observation model, fitting procedure or model selection.

### 11.5 Uncertainty and significance

- Bootstrap at the **animal** level, optionally stratified by stimulated-neuron group and acquisition batch.
- Pair candidate/baseline differences within outer fold.
- Report 95% confidence intervals for mean or median paired score differences, and the number of independent animal clusters.
- Use hierarchical mixed-effects analyses as a sensitivity check.
- Correct or explicitly control multiplicity for secondary hypothesis families.
- Avoid treating 23,433 measured pairs as 23,433 independent replicates.

### 11.6 Preregistered success criterion

Primary claim is supported only if:

1. The frozen model has positive improvement in the predefined paired animal-level predictive score over the **strongest eligible baseline** on the primary task (T2a), with T2b reported alongside.
2. The two-sided 95% uncertainty interval for the primary aggregate improvement excludes zero in the favorable direction, under the locked analysis.
3. The effect is not driven by one unusually rich animal, a tiny responder subset or post-selection of positive responses.
4. Calibration and null-response metrics do not deteriorate beyond a predeclared practical margin.
5. A documented complexity/performance trade-off is reported.
6. Any claim about shared neural timescales passes the interpretation rule of the indicator-kinetics control ([§4.5](../science/BASELINES.md)).

**Important:** this is a proposed criterion, not a claim that available sample sizes will necessarily support it. If power is inadequate, present effect sizes and uncertainty without binary victory language.

### 11.7 Negative controls

At minimum:

- Shuffle stimulus target labels **within scientifically appropriate matched strata**.
- Permute anatomical neuron labels while preserving selected graph statistics.
- Randomize chemical edge weights preserving in/out-degree structure where feasible.
- Replace real traces with matched noise controls.
- Remove neuromodulatory channels and compare.
- Compare real anatomy with degree-preserving randomized networks.
- Fit a model to the correct targets but corrupted stimulation times.
- Check whether a metadata-only model explains the results via batch/stimulation artifacts.
- Fit shared-kernel models to synthetic data with distinct neural kernels behind the estimated indicator and noise (false-sharing control, [§4.5](../science/BASELINES.md)).
- Shuffle stimulation order within recordings to test whether history terms capture real carryover.

Each negative control must avoid invalid exchangeability assumptions; for example, arbitrary permutations across different biological cell classes may create an overly easy strawman.

### 11.8 Data leakage tests

Automated assertions:

```text
assert intersection(train.animal_ids, test.animal_ids) == empty
assert no_hidden_responder_frames_used_in_test_calibration
assert no_test_labels_in_normalization_or_feature_selection
assert no_test_stimulus_results_in_program_ranker
assert no_cross_split_duplicate_trials
assert all_reported_source_ids_map_to_immutable_raw_data
```

Pipeline tests should intentionally insert a duplicate animal or leaked statistic and verify that CI fails.
