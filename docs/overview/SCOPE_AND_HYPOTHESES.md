# Scientific motivation, scope and hypotheses

> Part of the **Occam's Worm specification v0.4.0** · [Spec index](../README.md) · Source: §1

## 1. Scientific motivation and scope

### 1.1 The actual scientific problem

A structural connectome gives a graph of anatomical connections. It does **not** uniquely determine functional dynamics because synaptic signs, intrinsic neuronal response properties, neuromodulatory state, gap junction biophysics, receptor expression, physiology, measurement operators, body feedback, and developmental differences can alter the response to the same connectivity.

The neural signal-propagation atlas of Randi et al. measured responses for **23,433 head-neuron pairs** in **113 animals**, with highly unequal replication (some pairs observed only once, others many times). A pairwise aggregate is **not** a set of 23,433 independent training examples; animals and stimulus trials create a strongly clustered observational structure. The paper also reports extrasynaptic signaling, including dense-core-vesicle-dependent effects. These observations motivate a model that is more expressive than adjacency-matrix multiplication, yet need not begin with hundreds of custom neuronal differential equations. [R1]

The target is therefore a sequence of increasingly demanding questions:

- **Prediction:** Which compact local dynamical laws predict measured neural responses?
- **Generalization:** Do those laws transfer to new animals, stimulation targets and perturbations?
- **Identification:** Which mechanisms can current data distinguish, and which are empirically equivalent?
- **Minimal sufficiency:** What is the least biological detail needed to reproduce a specified set of causal observables?
- **Embodiment:** Can the same inferred nervous system generate appropriately constrained sensorimotor behavior in a closed-loop body?

These questions are separable. A good model for an isolated optogenetic response is not automatically a model of locomotion, and a locomoting synthetic controller is not automatically a faithful neural emulation.

### 1.2 Five computational design principles

| Principle | Operational interpretation in Occam's Worm | Guardrail |
|---|---|---|
| Simple programs can generate complexity | Enumerate small local dynamical rules rather than starting with maximal equation complexity | Complexity of output is **not** evidence of biological correctness |
| The space of short programs can be explored systematically | Treat short programs as a finite search space, with syntax, semantics and canonicalization | Search bias must be measured; grammar choice is a strong prior |
| Some dynamics may have no predictive shortcut | Empirically ask which observables need full simulation and which are predictable from reduced models | Do not infer the absence of a shortcut from a visually complex trajectory |
| Competing hypotheses form a branching structure | Retain a population of plausible hypotheses and their divergent experimental predictions | Branching is **bookkeeping over hypotheses**, not a claim about the system's dynamics |
| Equivalence is relative to an observer | Define explicit measurement operators and intervention-specific equivalence classes | Observational equivalence is **not** identity of mechanism, consciousness or subjective experience |

### 1.3 In scope

**Phase 1–3:** adult hermaphrodite, head-neuron stimulation/response data; model comparison; rule language; parameter inference; generalization; controlled synthetic tests; uncertainty and equivalence analysis.

**Phase 4:** genotype-specific or molecular interventions, where dataset and replication actually support it.

**Phase 5:** freely moving neural recordings; motor readouts; neural-body-environment co-simulation; realistic perturbation tests.

**Research extensions:** developmental variability, sensory adaptation, neuromodulatory fields, individual-specific model inference, minimum scanning requirements.

### 1.4 Out of scope for v0.1

- Claiming to produce a complete conscious or mind-uploaded worm.
- A new automated electron microscopy reconstruction system.
- Replacing all biological chemistry with a Boolean network by assumption.
- Predicting mutant phenotypes before checking mutant sample sizes and confounds.
- Producing realistic movement solely by training an unrestricted neural-to-muscle decoder.
- Inferring a unique molecular mechanism from calcium traces alone.
- Treating all 302 neurons as simultaneously observed in the head-only stimulation atlas.
- Building a full 3D body solver before passing neural held-out tests.

### 1.5 Falsifiable hypotheses

**H1 — Shared computational motifs:** Across many stimulated neurons, a small collection of shared dynamical response motifs explains held-out calcium responses nearly as well as or better than independently fitted kernels under matched regularization. H1 is a claim about **neural** dynamics: it is supported only if the shared motifs survive the indicator-kinetics control of [§4.5](../science/BASELINES.md). Shared kernels that the control cannot separate from measurement effects support only the weaker statement that the recordings are dominated by a common sensor filter at the available resolution.

**H2 — Compositional local rules:** A low-description-length, network-executed rule grammar predicts held-out neural perturbations better than an equally budgeted linear anatomical model and independent pairwise-response model.

**H3 — Identifiable causal distinctions:** A set of apparently equivalent rule programs can be separated by a small number of prospective or naturally held-out perturbations chosen using expected information gain.

**H4 — Fidelity hierarchy:** More stringent interventions split coarse functional-equivalence classes in measurable and interpretable ways.

**H5 — Embodied transfer (stretch):** A model selected on neural causal data can be coupled to a body without an unconstrained learned controller and retain meaningful locomotor behavior and perturbation sensitivity.

**Nulls:** no transferable shared motifs; apparent shared motifs explained by indicator kinetics or resolvable bandwidth alone; improvements vanish under animal-wise splits; rules perform no better than linear baselines; data are too sparse to distinguish mechanism families; closed-loop behavior fails without arbitrary readout training. Any of these is a scientifically acceptable outcome if established rigorously.
