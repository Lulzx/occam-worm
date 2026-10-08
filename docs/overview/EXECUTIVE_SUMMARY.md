# Executive summary

> Part of the **Occam's Worm specification v0.4.0** · [Spec index](../README.md) · Source: front matter and executive summary

> **One-sentence goal:** Discover compact, interpretable, biologically constrained computational programs that reproduce and **predict** causal neural responses in *C. elegans*—and quantify exactly which microscopic details must be preserved for progressively stronger forms of neural emulation.


Occam's Worm is a computational-science research platform for **program synthesis over biological nervous systems**. It starts with actual neural anatomy, perturbation experiments, molecular annotations and neural activity recordings; defines a typed language of candidate local update laws; executes those laws on anatomical networks; and selects models using held-out **causal predictions**, not attractive emergent patterns or training-set fit alone.

The name refers to the project's selection principle: prefer the most compact program, but only among programs that predict held-out interventions. Brevity is a prior and a tie-breaker; it never substitutes for causal accuracy ([§7.9](../inference/SEARCH_AND_INFERENCE.md), Anti-pattern D).

The design rests on a few general ideas about computation. Short programs can generate rich behavior. The space of short programs can be enumerated and searched systematically. Whether an observable can be predicted faster than by simulating every step is an empirical question, answered separately for each observable. Competing hypotheses are worth keeping as a branching set rather than collapsing early to one winner. And equivalence between models is always relative to a declared observer. [§1.2](SCOPE_AND_HYPOTHESES.md) turns each idea into an operational rule with a guardrail. None of them is assumed to be a fact about biology; each earns its place only through held-out predictive tests.

The shortest credible path to a publishable result is deliberately smaller than a whole moving worm:

1. Audit the **trial-level wild-type** neural signal-propagation atlas, keeping animal and stimulation identity intact.
2. Establish strong, leakage-free baselines for **shared-timescale versus independent-response kernels**, with the calcium indicator modeled as a separate stage so that measurement effects are not mistaken for shared neural timescales.
3. Construct a minimal typed rule grammar that can express those baselines and selected mechanistic alternatives.
4. Discover compact rules that predict **held-out animals, stimuli, and stimulation targets**.
5. Test predictions on an external perturbation, genotype, or independent dataset, with genotype-specific nuisance effects treated explicitly.
6. Only then connect a validated neural model to an embodied worm and test closed-loop behavior.

The project's claimed novelty is **not** that small programs yield complex patterns, that *C. elegans* has a connectome, or that a virtual worm can crawl. It is the intersection of **automated rule search + anatomical constraints + trial-level causal validation + explicit equivalence classes + experiment selection**.

### Primary scientific deliverable

A frozen, reproducible benchmark and at least one compact shared-rule model that improves **held-out predictive log likelihood** over an appropriately matched baseline, without a large loss of calibration or mechanistic plausibility. If no discovered model improves the baselines, the negative result should quantify where simple rules fail and what measurements could resolve the ambiguity.

### Primary engineering deliverable

A local-first CLI/library capable of ingesting experimental datasets, validating provenance, compiling a small rule DSL to deterministic execution, fitting model parameters, evaluating held-out data, and producing audit-ready comparison reports.
