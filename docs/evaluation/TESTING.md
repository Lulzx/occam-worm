# Testing and scientific quality assurance

> Part of the **Occam's Worm specification v0.4.0** · [Spec index](../README.md) · Source: §16

## 16. Testing and scientific quality assurance

### 16.1 Test matrix

| Level | Test | Expected result |
|---|---|---|
| Static | Type/units checker rejects invalid rules | Deterministic diagnostic errors |
| Static | Canonicalization idempotence | Same IR after repeat normalization |
| Static | Commutative AST normalization | Same hash for equivalent reordered add/sum |
| Dynamics | Zero-edge graph | No unexplained propagation |
| Dynamics | Two-neuron excitatory toy | Expected sign/timing within tolerance |
| Dynamics | Two-neuron inhibitory toy | Expected response suppression |
| Dynamics | Symmetric gap coupling | No drift from uniform initial voltage |
| Dynamics | Delay operator | Impulse arrives at correct timestep |
| Dynamics | Calcium filter | Matches analytic impulse response where defined |
| Dynamics | Random seed | Same outputs for identical run, differing draws under distinct seeds |
| Dynamics | Halved timestep | Converges within declared tolerance |
| Inference | Parameter recovery | Recovers synthetic identifiable parameters |
| Inference | Non-identifiable synthetic family | Reports posterior ambiguity rather than false certainty |
| Search | Hidden toy program | Recovers true/equivalent program in predefined small grammar |
| Search | Canonical duplicates | Evaluated only once unless stochastic replicates required |
| Statistics | Animal-group split | No animal or session leakage |
| Statistics | All-null stimulus dataset | Correctly favors null model under calibrated scoring |
| Statistics | Synthetic animal random effects | Hierarchical uncertainty captures clustering |
| Reporting | Frozen artifact replay | Reproduces tables/figures within tolerance |
| Reporting | Intentionally corrupted manifests | Fails loudly with provenance error |

### 16.2 Property-based tests

Randomly generate small valid graphs and programs and check:

- Permuting neuron storage order leaves canonically remapped outputs unchanged.
- Removing a chemical edge affects only permissible downstream propagation paths within the finite horizon, assuming no independent global coupling.
- Disconnected subnetworks are independent under appropriately local rules.
- Equal input and equal initial conditions in symmetrical subgraphs produce equal trajectories unless explicitly stochastic.
- Nonnegative magnitude parameters remain in allowed domains after fit.
- Serialization/deserialization preserves behavior and program hash.

### 16.3 Synthetic truth recovery design

Create a family of known generators covering:

1. Single leaky neuron.
2. Two-neuron excitatory/inhibitory coupling.
3. Three-node chain with delay.
4. Competing parallel pathways with different timescales.
5. Adaptation-driven transient response.
6. Recurrent coupled circuit with oscillation.
7. Slow modulation and context-dependent response.
8. Misspecified case: underlying conductance model outside grammar.
9. Identifiability failure: distinct latent models with same calcium readout.
10. Trial effects: animal-specific gain, onset jitter and correlated observation noise.
11. Sequential-stimulation carryover: adaptation or gain drift across stimulations within one recording.
12. Distinct neural kernels behind a shared slow indicator at the measured noise level (false-sharing check, [§4.5](../science/BASELINES.md)).

Sweep sample size, noise level, stimulation density, number of observed cells and grammar complexity. Recovery metrics distinguish (a) AST identity, (b) parameter recovery, (c) predictive equivalence and (d) true interventional equivalence. These are different outcomes.

### 16.4 Adversarial scientific tests

- Can a model simply memorize target-neuron identity with a hidden lookup table?
- Can a shared latent component explain apparent network dynamics solely from broad optical artifact?
- Does an arbitrary calcium filter absorb wrong neuronal timescales?
- Does training on reliable-response-only subsets inflate apparent rule generalization?
- Does a model selected on many weak trials fail on a few strong, repeatedly observed interventions?
- Are purported shared update laws just flexible per-edge weights in disguise?
- Do variable recording lengths bias the likelihood toward long traces?
- Can global normalization accidentally expose held-out animal statistics?

Document at least one deliberate attempted falsification of the preferred model family before publication.

### 16.5 Continuous integration

Run on every pull request:

- C++ build with warnings as errors on Clang and GCC; clang-format and clang-tidy checks.
- C++ unit tests (GoogleTest) and property tests (RapidCheck), run under AddressSanitizer and UndefinedBehaviorSanitizer.
- A short fuzzing run on the WRL parser and canonicalizer; longer runs nightly.
- Python typing, formatting and import smoke tests.
- Small synthetic simulation plus parser conformance.
- Leakage assertions on synthetic fixtures.
- Determinism and report checksumming tests.
- License/data redistribution scan for committed assets.

Nightly or manual expensive CI runs may include full synthetic recovery and public non-sensitive benchmark subsets. Do not embed large restricted datasets in the source repository.
