# Perturbations, genetics and neuromodulation

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §12

## 12. Perturbations, genetics and neuromodulation

### 12.1 Why mutation prediction is not a trivial test

A genotype such as `unc-31` may alter transmitter/neuropeptide release, receiver physiology, development, excitability and experimental stimulus responses. It is not scientifically safe to represent genotype merely as `gamma=0` on a global peptide edge and then interpret a flat response as verification of a particular mechanism.

A real comparison requires stimulus and autoresponse controls, receiver baseline differences, sample-size audit and explicit alternative explanations.

### 12.2 Genotype-conditioned forward models

Predefine at least three candidate classes:

- **Mechanism-only intervention:** genotype changes a restricted release/receptor component; all other parameters fixed.
- **Mechanism + nuisance:** restricted mechanism plus measured changes to observation/baseline/excitability.
- **Flexible genotype model:** genotype-specific neural parameters with shrinkage, treated as an upper-flexibility comparator.

A strict transfer claim fits the WT model and freezes it before mutant data are examined. Genotype nuisance estimation, if allowed, must be specified and isolated from the targeted response prediction.

### 12.3 Modulatory model tiers

**M0:** no modulatory edges.  
**M1:** global or region-specific slow latent field.  
**M2:** one/few ligand/receptor-compatible channels.  
**M3:** spatial diffusion/volume transmission dynamics.  
**M4:** richer cell-specific ligand–receptor and state-dependent release, only if independently constrained.

Example field dynamics:

$$
\partial_t m_k(x,t)=D_k\nabla^2 m_k-\kappa_k m_k
+\sum_j q_{jk}(t)\,K(x-x_j).
$$

This PDE is an **optional approximation**, not a literal assertion that the experimental system supports identifiable diffusion constants. A lower-dimensional graph diffusion model may be more appropriate with current recordings.

### 12.4 Causal isolation tests

Compare:

- Condition-aware response predictions at fixed stimulus autoresponse.
- Sign/latency/shape changes, not only binary presence of response.
- Peptide/receptor expression-compatible pathways vs matched incompatible controls.
- With and without spatial diffusion.
- Mutants and sham controls when statistically powered.

Do not label a candidate ligand–receptor path as a validated neurotransmission mechanism on transcriptomic matching alone.
