# Observer-relative functional equivalence

> Part of the **Occam's Worm specification v0.4.0** · [Spec index](../README.md) · Source: §9

## 9. Observer-relative functional equivalence

### 9.1 Define the observer, do not mystify it

An observer for this project is a **predeclared measurement/intervention protocol**:

$$
\mathcal O=(\mathcal A,\;H,\;\mathcal T,\;d,\;\varepsilon),
$$

where:

- `A`: allowable interventions / input histories.
- `H`: measurement operator (calcium, electrical, motor, behavior).
- `T`: observation horizon and sampling grid.
- `d`: distance or statistical divergence between predicted measurement distributions.
- `epsilon`: tolerance, fixed from noise and scientific requirements.

Two candidate models are indistinguishable relative to this observer if

$$
\sup_{a\in\mathcal A}\ d(P^{M_1}_{Y\mid a},P^{M_2}_{Y\mid a})\le\varepsilon.
$$

Because finite samples do not establish a true supremum, the implementation stores **empirical indistinguishability under tested interventions**, plus uncertainty intervals; it must not silently claim global equivalence.

### 9.2 Equivalence levels

| Level | Observer | Permissible claim |
|---|---|---|
| F0 | Fixed input, aggregate endpoint score | Similar summary behavior |
| F1 | Calcium traces for known stimulation | Similar measured dynamics on tested trials |
| F2 | Novel stimulation targets/pulse schedules | Similar held-out intervention response |
| F3 | Molecular/cell ablations/mutants | Similar perturbation-conditioned behavior |
| F4 | Electrophysiology and spatially resolved dynamics | Finer physiological similarity |
| F5 | Closed-loop sensorimotor history + neural responses | Broader embodied functional similarity |

There is no level at which these metrics prove subjective awareness, personhood or persistence of identity.

### 9.3 Equivalence versus proximity

Do not confuse:

- Small Euclidean distance between latent states (coordinate dependent).
- Matching a behavior classifier's category labels (lossy summary).
- Similar full predictive response distributions (stronger operational test).
- Mechanistic/structural isomorphism (different mathematical requirement).

For causal emulation, interventional predictive agreement is primary; observational correlation alone is weak evidence.

### 9.4 Constructing empirical equivalence classes

Use a set of experiments `A_test` to create pairwise distances between candidates. Build an **indistinguishability graph** where a statistically supported equivalence edge joins candidate pairs. Do not automatically take transitive connected components as mathematically valid equivalence classes: thresholded noisy distances can fail transitivity. Instead report maximal cliques, cluster assignments with robustness intervals, or another explicitly defined approximation.

### 9.5 Minimal necessary biological detail

Starting with a model family `M_full`, ablate features one class at a time:

- Cell-specific parameters → type-shared parameters.
- Detailed kinetics → one/two shared response timescales.
- Per-edge synaptic weights → anatomical count-based weights.
- Molecularly specific modulation → generic slow field.
- Gap junctions → no gap junctions.
- Explicit body proprioception → open-loop input.

Quantify the change in held-out predictive distribution. A feature is *operationally necessary* at observer level `F_k` if removing it causes a replicated, scientifically consequential and confidence-bounded loss under a predeclared metric. This is more informative than counting model parameters in isolation.
