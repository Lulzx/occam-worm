# Risks, failure modes and anti-patterns

> Part of the **Occam's Worm specification v0.4.0** · [Spec index](../README.md) · Source: §19

## 19. Risks, failure modes and mitigation

| Risk | Why it matters | Mitigation / decision |
|---|---|---|
| Insufficient repeats | One-shot pair observations cannot support independent held-out estimation | Animal/trial coverage audit, restricted eligible cohort, uncertainty not overstated |
| Calcium observation ambiguity | Many latent voltage processes give similar fluorescence | Explicit measurement operator; alternate kernels; external electrophysiology where possible |
| Model non-identifiability | Different rules agree on available interventions | Preserve candidate population and select discriminating experiments |
| Grammar bias | A true mechanism may not be expressible | Structured misspecification suite; tier expansion based on residuals |
| Search overfitting | Many candidate programs exploit validation noise | Nested grouped CV; frozen external holdout; search history |
| Anatomy heterogeneity | Different connectome sources/stages alter topology | Source-specific graph manifests and stage-aware sensitivity analyses |
| Transmitter sign uncertainty | Anatomical chemical connections are not universally labeled excitatory/inhibitory | Explicit latent signs with priors and posterior uncertainty |
| Extrasynaptic confounds | Dense-core-vesicle signaling may be fast and context-dependent | Graduated modulation tiers and independent mutant controls |
| Animal state variation | Trial/state differences mimic new mechanisms | Hierarchical nuisance model, pre-stimulus controls, paired trials |
| Artificially learned motor decoder | Movement can come from decoder rather than correct neural rules | Decoder restriction and shuffled-neural/decoder-only controls |
| Unstable integration | Numerical blowup mistaken for complex emergence | Stability tests, timestep convergence, independent reference engine |
| Overclaiming unpredictability | Complexity interpreted as proof of no short prediction | Empirical shortcut benchmark and limited claims |
| Ambiguous equivalence | Similar observables over few tests interpreted as same mind | Observer-indexed empirical similarity only; no consciousness claim |
| Licensing/provenance failure | Data or reference code not legally redistributable | Preserve source licenses; metadata-only references when needed |
| Indicator-kinetics confound | A shared sensor filter or limited resolvable bandwidth can masquerade as shared neural timescales | Separate indicator stage, resolvable-bandwidth report, synthetic false-sharing control ([§4.5](../science/BASELINES.md)) |
| Within-recording carryover | Sequential stimulations change animal state; order effects mimic mechanism or noise | Stimulation-history audit, history terms, order-shuffle control |
| Gradients in C++ | Hand-built or tool-generated gradients add build complexity and are easy to get wrong | Differentiable Python simulator for fitting; C++ for toolchain and reference interpreter |
| Memory and undefined-behavior bugs in C++ | Silent corruption or nondeterminism can masquerade as scientific results | Sanitizers and fuzzing in CI, warnings as errors, bounds-checked reference interpreter, two-compiler builds |
| Large research scope | Full worm physics delays a decisive result | Data-first milestones with hard go/no-go gates |

### 19.1 Key methodological anti-patterns

#### Anti-pattern A: optimize against paper figures

A model tuned to a handful of published mean traces may look impressive while failing all independent animals. Use trial-level records with frozen splits.

#### Anti-pattern B: predict only significant responses

Selecting downstream neurons on the outcome removes hard near-null cases and inflates predictive success. Include the observation mask, nulls and failed-response uncertainty.

#### Anti-pattern C: infer peptide identity from fast signal loss in a mutant

The mutant may affect broad physiology or stimulus efficacy. Test multiple genotype-nuisance alternatives and state the limits of causal attribution.

#### Anti-pattern D: confuse compressed code with biological mechanism

Short descriptions are useful priors. They are not experimental evidence by themselves; the evidence is out-of-sample and interventional prediction.

#### Anti-pattern E: use neural-to-muscle training to prove connectome-driven behavior

If the decoder can be trained to mimic motion using random features, successful movement is not evidence of correct neural computation.

#### Anti-pattern F: pick a winner after repeatedly observing the final test set

The outer test is not a public tuning leaderboard. Any development after unblinding changes which data remain confirmatory.
