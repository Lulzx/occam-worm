# Publication strategy and claims discipline

> Part of the **Occam's Worm specification v0.4.0** · [Spec index](../README.md) · Source: §20, §23, Appendix C

## 20. Publication strategy, scientific claims and negative results

### 20.0 Framing for publication

Each paper is framed in standard methodological terms (nested cross-validation, minimum description length, program synthesis, Bayesian experimental design, causal intervention testing) and stands on its held-out results. Project-specific terms such as hypothesis graph, observer-relative equivalence and predictive shortcut appear in a paper only together with the precise operational definition given in this specification. Paper A needs none of them.

### 20.1 Paper A: shared temporal primitives

**Proposed title:** *How Many Temporal Laws Govern the Functional Connectome of C. elegans?*

Core result: carefully controlled shared-timescale / low-rank / independent-kernel comparison over stimulation targets with animal-wise holdout. Include uncertainty by neuron class and recording density. The paper reports the indicator-kinetics control ([§4.5](../science/BASELINES.md)), both T2a and T2b, and the history-variant comparison, and states plainly whether any shared kernels reflect neural dynamics or measurement.

This paper can stand independently of the WRL engine and is the most achievable first scientific milestone.

### 20.2 Paper B: program synthesis on anatomical neural graphs

**Proposed title:** *Causal Rule Discovery from the C. elegans Connectome and Perturbation Atlas*

Core result: typed rule language, reliable computational benchmark, synthesized neural update laws, proper held-out generalization and a complexity–prediction Pareto frontier.

A publishable positive result must show that compact program synthesis contributes beyond its baseline fitting procedure and observation model.

### 20.3 Paper C: experimental distinguishability of neural programs

**Proposed title:** *Interventional Equivalence Classes of Neural Dynamics*

Core result: posterior over mechanistic programs, explicit observer-based equivalence, optimal perturbation choice and (ideally) external discrimination.

Even if multiple models fit equally well, a carefully quantified non-identifiability result and a precise informative-experiment proposal are worthwhile scientific outcomes.

### 20.4 Possible later paper: minimum functional scanning requirements

Take a validated program family and corrupt or remove structural/molecular observations, then estimate degradation of held-out causal fidelity. This can motivate scanner requirements, but in-silico perturbations of measurements alone do not establish practical scanner performance or physical tissue-preparation feasibility.

### 20.5 Claims discipline

The following wording is appropriate only when supported by the relevant test:

- “A short program predicts held-out stimulation responses.”
- “A type-shared rule matches or exceeds a per-pair model under a specified complexity budget.”
- “Two models are empirically indistinguishable under these interventions and measurement tolerances.”
- “The proposed intervention is predicted to disambiguate model hypotheses.”

Avoid unsupported claims such as:

- “We discovered the worm's true code.”
- “The simulation has the same experience as the worm.”
- “The system was proven unpredictable from a chaotic trace.”
- “A mutant proves a specific peptide caused the response.”
- “Moving like a worm proves connectome-level emulation.”


## 23. Final research commitment

> **Do not start by trying to make the whole worm move. Start by discovering whether neural temporal dynamics contain compact, shared, causal computational structure that survives independent-animal evaluation.**

The first decisive project is a test of shared temporal dynamics in the wild-type optogenetic response atlas. If that works, expand the hypothesis language to local recurrent rules executed on the connectome. If those rules predict new perturbations, explicitly study which internal mechanisms remain indistinguishable. If a mechanistic model survives those tests, attach the body and close the loop.

The longer-term ambition is a **functional fidelity compiler**: given a biological network, candidate physiological rules, experimental observations, and an explicit fidelity target, produce the smallest model with quantified predictive coverage—and determine which additional biological measurements would most improve confidence.

That is a scientifically defensible bridge from *simple programs* to *neural emulation*. A successful worm model would be one milestone in the bridge, not its endpoint.


### Appendix C — Suggested first research report outline

1. Biological coverage and trial counts.
2. Eligibility decisions with inclusion flowchart.
3. Primary fixed comparison: shared kernel versus independent kernel.
4. Low-rank timescale bank comparison.
5. Regularized anatomical linear model comparison.
6. Animal-level held-out effect sizes and confidence intervals.
7. Per-target heterogeneity and residual analysis.
8. Confounds: stimulation efficacy (T2a versus T2b), indicator-kinetics control, stimulation-history effects, fluorescence noise and others.
9. Predeclared conclusions and falsified simplifications.
10. Decision: remain at simple kinetics, proceed to network rule grammar, or collect more data.
