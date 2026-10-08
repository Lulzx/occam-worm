# Roadmap and acceptance gates

> Part of the **Occam's Worm specification v0.4.0** · [Spec index](../README.md) · Source: §15

Implementation tickets for these milestones are in [tickets/](tickets/README.md).

## 15. Detailed execution roadmap and acceptance gates

All schedules are illustrative planning estimates for a focused researcher using coding assistance. Scientific discovery cannot be guaranteed by any duration. Windows were widened in 0.2.0; the near-term target remains the minimum viable scientific release (M0–M1, [§21.2](tickets/README.md)).

### Milestone M0 — Data provenance and feasibility audit

**Suggested window:** weeks 1–2 (the coverage, history and indicator audits make this longer than a few days).  
**Objective:** establish what the released data actually support.

Tasks:

- [ ] Mirror required upstream resources and write checksums/license manifest.
- [ ] Verify access to actual per-animal stimulus and response traces.
- [ ] Implement neuron-ID reconciliation and stage/genotype metadata.
- [ ] Count trials, animals, targets, observed responders and time windows.
- [ ] Detect missing/nonresponse confusion and repeated-trial structure.
- [ ] Render response variability for 10 representative stimulus targets.
- [ ] Define eligible target set and document exclusions before baseline results.
- [ ] Produce split feasibility analysis.
- [ ] Audit stimulation order, inter-stimulus intervals and within-recording drift.
- [ ] Build target-by-fold coverage tables for at least three split schemes, including leave-one-animal-out; freeze the scheme.
- [ ] Identify the indicator and assemble the data that constrain its kinetics.

**Output:** `artifacts/audit-v1/{coverage.parquet,fold_coverage.parquet,history.md,coverage.md,source_manifest.json,figures/}`.

**Acceptance gate:** source traces are recoverable, provenance is intact, and at least one animal-wise validation task is feasible. Otherwise re-scope.

### Milestone M1 — Baselines and frozen benchmark

**Suggested window:** weeks 2–5.  
**Objective:** answer the simpler shared-timescale question before building a rule-search engine.

Tasks:

- [ ] Implement dataset-to-trace iterator and train/test firewall.
- [ ] Implement null, shared-kernel, low-rank and independent-kernel models.
- [ ] Implement animal-wise cross-validation and grouped uncertainty.
- [ ] Implement calcium trace likelihood/noise modeling and alignment checks.
- [ ] Add linear anatomical dynamical baseline.
- [ ] Fit history variants of each baseline and the separate indicator stage.
- [ ] Run the indicator-kinetics control ([§4.5](../science/BASELINES.md)); report T2a and T2b separately.
- [ ] Run injection/recovery synthetic experiments.
- [ ] Freeze primary metric, reporting script and held-out split definition.
- [ ] Publish result table with every target, not only favorable examples.

**Output:** a baseline technical note plus a machine-readable benchmark dataset manifest.

**Acceptance gate:** scripts run from raw manifests to scores without manual per-figure edits, and the benchmark is nontrivial under animal holdout.

### Milestone M2 — WRL compiler and reference simulator

**Suggested window:** weeks 5–11. A typed language with units, canonical form, stable hashing, an independent reference interpreter and a conformance suite is substantial work; do not compress it.  
**Objective:** provide correct semantics before large-scale model discovery.

Tasks:

- [ ] Specify grammar G0 and G1 with grammar tests.
- [ ] Implement typed AST, units, canonicalization and stable hashing.
- [ ] Implement scalar synchronous executor and calcium readout.
- [ ] Implement sparse chemical and electrical edge semantics.
- [ ] Implement deterministic buffers, stimuli, masks and state reset.
- [ ] Validate finite-step stability and known analytical toy solutions.
- [ ] Implement the differentiable Python simulator and match it to the C++ reference interpreter on the conformance suite.
- [ ] Defer any C++ forward-pass optimization until profiling in M3 justifies it.

**Output:** `ow-ir`, `ow-sim`, `ow-observe`, `python/occamworm/sim`, conformance suite and sample models.

**Acceptance gate:** all synthetic ground-truth programs recovered/evaluated within defined tolerance, and state transitions reproducible.

### Milestone M3 — Structural search and numerical fit

**Suggested window:** weeks 11–15.  
**Objective:** show that candidate local programs can be found from observations.

Tasks:

- [ ] Enumeration under small AST budgets; canonical deduplication.
- [ ] Structured insertion/deletion/mutation search.
- [ ] Differentiable or derivative-free parameter fitting with tests.
- [ ] Model complexity and effective parameter accounting.
- [ ] Candidate filtering using stability and synthetic plausibility.
- [ ] Synthetic truth recovery across increasing noise and graph size.
- [ ] Fixed-budget comparison of enumeration, random search and guided search.

**Output:** frozen search recipe and recoverability benchmarks.

**Acceptance gate:** synthetic true rules (within the declared grammar) are recoverable under known favorable conditions, and misspecification/noise failure is documented.

### Milestone M4 — Real atlas search and causal generalization

**Suggested window:** weeks 15–20.  
**Objective:** evaluate whether small shared programs predict real held-out responses.

Tasks:

- [ ] Lock program/parameter budgets and baseline budget matching.
- [ ] Run inner-search/outer-test protocol on WT eligible cohort.
- [ ] Evaluate T2 and, where data permit, T3 target holdouts.
- [ ] Compare rule complexity/accuracy Pareto fronts.
- [ ] Run shuffled labels, shuffled connectome and no-modulation controls.
- [ ] Inspect residual structure by neuron class, response sign and latency.
- [ ] Perform replicated seed / solver / timestep sensitivity checks.

**Output:** preprint-quality tables, frozen rule programs and uncertainty analyses.

**Acceptance gate:** support H1/H2 under preregistered metrics **or** publish a calibrated negative result with evidence of why the grammar failed.

### Milestone M5 — Hypothesis-graph equivalence and experiment selection

**Suggested window:** weeks 20–24.  
**Objective:** map remaining ambiguity and identify informative perturbations.

Tasks:

- [ ] Retain top-diverse candidate models and posterior approximations.
- [ ] Compute interventional predictive distances and empirical equivalence graph.
- [ ] Define feasible intervention menu independent of outcome.
- [ ] Rank interventions by expected information gain and cost.
- [ ] Run retrospective blinded selection test where suitable.
- [ ] Produce diagnostic visualizations for mutually inconsistent model predictions.

**Output:** intervention-ranked table, uncertainty estimates and equivalence-level report.

**Acceptance gate:** generated proposals demonstrably distinguish candidate predictions better than predeclared random or heuristic choices in held-out simulations; real-world discrimination is a separate later claim.

### Milestone M6 — Perturbation/genotype transfer

**Suggested window:** after successful M4/M5, data-dependent.  
**Objective:** falsify or strengthen mechanistic interpretations.

Tasks:

- [ ] Audit mutant/intervention data availability separately.
- [ ] Pre-register WT-frozen prediction and nuisance allowance.
- [ ] Compare mechanism-only, mechanism+nuisance and flexible models.
- [ ] Quantify confounds due to stimulus efficacy, excitability and sampling.
- [ ] Report truly out-of-condition prediction and uncertainty.

**Acceptance gate:** credible external perturbation generalization; if infeasible, state exact underpowering/identifiability limitations.

### Milestone M7 — Embodiment and cross-scale fidelity

**Suggested window:** after a validated neural result; duration open.  
**Objective:** test whether neural causal competence survives full sensorimotor coupling.

Tasks:

- [ ] Select validated body physics backend; audit licensing/interoperability.
- [ ] Implement NMJ/muscle mapping interface and sensory feedback.
- [ ] Calibrate non-neural body parameters on independent data.
- [ ] Lock or strongly constrain neural-to-muscle decoder.
- [ ] Evaluate muscle, kinematic, behavioral and perturbation tests.
- [ ] Compare to randomized-neural and decoder-only controllers.

**Acceptance gate:** statistically supported sensorimotor improvements attributable to correct neural dynamics, not a powerful decoder alone.

### Decision tree

```text
Does trial-level data support genuine animal-wise held-out testing?
    NO  -> release identifiability/coverage audit; seek alternative data.
    YES -> are baseline predictions calibrated and reproducible?
               NO  -> fix measurement/split model; do not search rules.
               YES -> can the grammar recover known synthetic rules?
                          NO  -> fix DSL/inference implementation.
                          YES -> does the discovered program outperform strong baselines?
                                     YES -> test new targets and interventions.
                                     NO  -> inspect misspecification and residuals;
                                            expand grammar only with evidence.
```
