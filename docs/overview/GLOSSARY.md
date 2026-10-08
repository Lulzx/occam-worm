# Glossary

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: Appendix A

## Appendix A — Terminology

- **Anatomical connectome:** observed synaptic/electrical structural graph.
- **Functional atlas:** measured effects of experimentally stimulating neurons and observing downstream activity.
- **Autoresponse:** activity of the directly stimulated neuron in response to its own stimulation. An input channel in T2a; never a scored responder output.
- **Calcium observation operator:** model mapping neural latent state to observable fluorescence.
- **WRL:** project-specific typed Worm Rule Language.
- **Program structure:** operators, topology-conditioned expressions and state registers, distinct from numeric parameters.
- **MDL:** minimum description length, a complexity-adjusted modeling principle.
- **`L_struct`, `L_params`, `L_total`:** structural bits, parameter bits at declared precision, and their sum; the only complexity measures used ([§5.7](../language/GRAMMAR.md)).
- **Indicator-kinetics control:** the preregistered test in [§4.5](../science/BASELINES.md) of whether shared kernels reflect neural dynamics or measurement.
- **Stimulation history:** the order, timing and autoresponses of earlier stimulations in the same recording.
- **Hypothesis graph:** alternative candidate programs, the edits and updates relating them, and their predicted observations ([§8](../science/EXPERIMENT_DESIGN.md)).
- **Empirical interventional equivalence:** failure to distinguish models over a declared finite set of experiments and resolution.
- **Scientific negative result:** a properly powered or appropriately uncertainty-qualified failure to support a hypothesis.
- **T2a/T2b/T3/T4:** new-animal prediction conditioned on the measured autoresponse, new-animal prediction from the nominal stimulus, stimulation-target generalization, and cross-condition generalization, respectively.
