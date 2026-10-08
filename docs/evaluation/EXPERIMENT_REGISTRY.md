# Experiment registry and configuration

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §14.6, §17.4

Concrete configuration lives in [`configs/experiments/`](../../configs/experiments/); registry entries are immutable once a run is frozen.

## 14.6 Example experiment configuration

```toml
[experiment]
name = "shared-timescales-wt-v1"
question = "Do shared response timescales generalize across animals?"
primary_condition = "wild-type"
primary_metric = "animal_paired_predictive_nll_difference"
primary_task = "T2a_autoresponse_conditioned"
secondary_tasks = ["T2b_nominal_stimulus"]

[data]
manifest = "configs/datasets/primary.toml"
use_individual_trials = true
include_valid_near_null_responses = true
calibration_policy = "pre_stimulus_only"
history_term = "frozen_from_training_audit"
initial_state_policy = "history_conditioned"   # default unless the M0 audit finds no carryover (§6.7)

[splits]
unit = "animal_id"
outer = "group_kfold"            # provisional; frozen from M0 coverage tables
outer_folds = 5
inner = "group_kfold"
inner_folds = 3
seed = 20261008
scheme_justification = "artifacts/audit-v1/fold_coverage.parquet"

[models]
names = ["null", "shared_kernel", "low_rank_kernels", "independent_kernels", "linear_network"]

[observation]
family = "student_t_correlated"
fit_noise_on = "training_only"
allow_test_specific_scale_calibration = false
indicator_stage = "separate_fitted"
indicator_control = true

[report]
bootstrap_unit = "animal"
bootstrap_replicates = 2000
confidence_level = 0.95
report_all_stim_targets = true
```

The number of folds is a **configurable default**, chosen from the M0 coverage tables: if an audit demonstrates that an outer fold has inadequate independent animal coverage, build a justified alternate split before inspecting model results.


## 17.4 Experiment registry

Each experiment has an immutable machine-readable entry:

```yaml
experiment_id: wt-shared-kernels-0001
question: Are downstream response time courses shared?
hypotheses: [H1]
data_manifest: <hash>
split_manifest: <hash>
model_families: [B0, B1, B2, B3, B4]
primary_metric: animal_paired_predictive_nll_difference
selection_protocol: nested_grouped_cv
seed_policy: fixed_and_logged
frozen_before_test: true
limitations:
  - variable pairwise replication
  - calcium measurement confounding
```

An experimental report must distinguish planned tests, completed tests, exploratory discoveries and retrospective hypotheses.
