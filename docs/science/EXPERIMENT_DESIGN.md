# Hypothesis graph and causal experiment design

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §8

## 8. Hypothesis graph and causal experiment design

### 8.1 Interpretation

A **hypothesis graph** is a directed graph over candidate programs, fitted parameters, experimental beliefs and observed data. Edges indicate structural edits, alternative hypotheses, or posterior updates. Branches record where hypotheses diverge in their predictions; the graph describes the state of the inquiry, not the dynamics of the worm.

Two programs may produce identical unperturbed locomotion and diverge strongly when neuron `AVA` is stimulated. That divergence is exactly what the system should search for.

### 8.2 Objects

```text
Hypothesis:
  program_hash
  grammar_version
  fitted_parameter_posterior
  training_log_likelihood
  heldout_score_summary
  biological_constraint_violations
  posterior_weight_estimate
  predictive_feature_vector

HypothesisEdge:
  source_program_hash
  target_program_hash
  mutation_operator
  edit_cost_bits
  evaluation_round

ExperimentProposal:
  intervention_type
  target_neurons
  waveform
  duration_and_timing
  proposed_readout
  feasible_controls
  safety_or_protocol_constraints
  expected_information_gain
  expected_cost
  predictions_by_hypothesis
```

### 8.3 Ranking discriminative experiments

Let `R` denote the uncertain program model and `Y_a` the future measured outcome under intervention `a`. Select

$$
a^*=\arg\max_{a\in\mathcal A_{feasible}}
\bigl[I(R;Y_a\mid D)-\lambda_c\,Cost(a)\bigr].
$$

Compute expected information gain with predictive Monte Carlo:

$$
I(R;Y_a\mid D)
=\mathbb E_{R,Y_a}\bigl[\log p(Y_a\mid R,a,D)-\log p(Y_a\mid a,D)\bigr].
$$

In practice estimate expected posterior entropy reduction using particles, with uncertainty bars from Monte Carlo replication. Experiments with visually divergent predictions but **large predicted noise** may have low information gain.

### 8.4 Allowed experiment menu

Start with retrospective, genuinely held-out stimulation protocols (which are not presented as prospectively optimized experiments). Candidate prospective choices include:

- Stimulated-neuron identity.
- Stimulus duration, amplitude and pulse pattern within the physical apparatus capability.
- Multiple-pulse temporal separation.
- One or multiple cell-specific intervention(s), only if technically realizable.
- Control condition, sham illumination and recording windows.
- Specific neuron type or peptide/receptor perturbation, subject to biological validation.

Explicitly constrain allowable dose and stimulation combinations by actual experimental safety/feasibility. The project generates **research suggestions**, not an automated wet-lab actuator.

### 8.5 Retrospective leakage control

If using an already available external held-out trial to demonstrate a chosen experiment:

- The model may see the intervention design, but **not its outcome** before selection.
- Record its prior ranking against all eligible interventions.
- Evaluate calibration and rank-based enrichment after unblinding.
- Do not call retrospective selection a prospective discovery.

### 8.6 Scientific output

A useful paper figure shows: (i) 5–20 top candidate programs with nearly identical WT predictions; (ii) their sharply divergent predictions under a new stimulation; (iii) actual measurement; (iv) posterior weight update; and (v) new equivalence-class structure. This is stronger than simply displaying one worm simulation.
