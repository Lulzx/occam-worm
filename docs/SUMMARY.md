# Occam's Worm — Concise Specification

> One-page summary of the v0.4.0 specification. Section numbers (§) in this file refer to its own sections, not the full specification. The full specification is split by topic; see the [spec index](README.md).

Oct 8, 2026 · @Lulzx

## 1. Overview

Occam's Worm searches a small typed language of local neural update rules, runs candidates on the *C. elegans* connectome, and keeps only rules that predict causal responses in animals they never saw.

**Goal.** Discover compact, interpretable, biologically constrained programs that predict causal neural responses in *C. elegans*, and quantify which biological details each level of fidelity requires.

**Name.** Prefer the most compact program, but only among programs that predict held-out interventions. Brevity is a prior and a tie-breaker; it never substitutes for causal accuracy.

**Status.** Version 0.4 research design; not implemented. Python (JAX or PyTorch) handles data, baselines, differentiable simulation and fitting. C++20 handles the rule toolchain, enumeration, reference interpreter and CLI.

**Claimed novelty.** Not that simple rules make complex patterns, or that a virtual worm can crawl. It is the combination of automated rule search, anatomical constraints, trial-level causal validation, explicit equivalence classes and experiment selection.

**Deliverables.**

- Scientific: a frozen, reproducible benchmark and at least one compact shared-rule model that beats the strongest matched baseline on held-out predictive log likelihood without losing calibration. Failing that, a quantified negative result showing where simple rules fail.
- Engineering: a local-first CLI and library that ingests data with provenance, compiles rules, fits parameters, evaluates on held-out data and writes audit-ready reports.

**Order of work.**

1. Audit the trial-level wild-type stimulation atlas, keeping animal and trial identity.
2. Compare shared versus independent response kernels, with the calcium indicator modeled separately.
3. Build a minimal typed rule grammar that can express those baselines.
4. Find compact rules that predict held-out animals, stimuli and stimulation targets.
5. Test on an external perturbation, genotype or dataset.
6. Only then couple the neural model to a body.

**Out of scope for v0.x:** consciousness or mind-upload claims; new electron-microscopy reconstruction; inferring a unique molecular mechanism from calcium alone; a full 3D body before neural held-out tests pass; treating all 302 neurons as observed in the head-only atlas.

## 2. Principles and hypotheses

Five design principles shape the engine; none is assumed to be a fact about biology.

| Principle | How it is used | Guardrail |
| --- | --- | --- |
| Simple programs can generate complexity | Enumerate small local rules before complex equations | Complex output is not evidence of correctness |
| Short programs can be explored systematically | Treat them as a finite space with syntax, semantics and canonical forms | Grammar choice is a strong prior; measure its bias |
| Some dynamics may have no predictive shortcut | Test per observable whether reduced models suffice | Never infer "no shortcut" from a complex-looking trace |
| Competing hypotheses form a branching structure | Keep a population of candidates and their diverging predictions | Branching is bookkeeping over hypotheses, not system dynamics |
| Equivalence is relative to an observer | Declare measurement operators and intervention sets | Observational equivalence is not identity of mechanism or experience |

**Hypotheses.**

- **H1 Shared motifs.** A few shared temporal response motifs predict held-out calcium responses about as well as independent kernels under matched regularization. Counts as a neural claim only if it passes the indicator-kinetics control (§5).
- **H2 Compositional rules.** A low-description-length rule run on the connectome beats an equally budgeted linear anatomical model and an independent pairwise model on held-out perturbations.
- **H3 Identifiable distinctions.** Rules that look equivalent can be separated by a few interventions chosen for expected information gain.
- **H4 Fidelity hierarchy.** Stricter interventions split coarse equivalence classes in measurable, interpretable ways.
- **H5 Embodiment (stretch).** A neurally validated model drives a body without an unconstrained learned controller and keeps meaningful behavior.

**Acceptable nulls:** no transferable motifs; motifs explained by indicator kinetics; gains vanish under animal-wise splits; rules no better than linear baselines; data too sparse to separate mechanisms; behavior fails without arbitrary readout training.

## 3. Data

The primary data are trial-level optogenetic stimulation and calcium responses from the Randi et al. 2023 signal-propagation atlas: about 23,433 head-neuron pairs across 113 animals, with very uneven replication. Pairs are not independent examples; animals and trials are.

| Source | Use | Caveat |
| --- | --- | --- |
| Randi et al. 2023 atlas (Leifer Lab) | Primary causal training and evaluation | Calcium, not voltage; most pairs weak or unreplicated |
| `pumpprobe` library | Reference importer and trace-level audit | Fitted kernels are not raw trials |
| OpenWorm Connectome Toolbox | Typed anatomical graphs | Sources and life stages are not interchangeable |
| Worm Neuro Atlas | Transmitter, receptor and peptide annotations | Expression implies capability, not communication |
| WormWideWeb (Atanas et al. 2023) | External freely moving activity and behavior | Different preparation and observables |
| BAAIWorm (Zhao et al. 2024) | Embodiment reference | Already demonstrates embodied simulation |

**Acquisition rules.** Download from the original DOI or repository and store checksums, versions, retrieval dates and licenses. Never modify raw bytes. Keep every animal and trial ID. Mark unknowns `null` with a reason. Never label an unobserved pair as a non-response. Record whether each value is measured, fitted or inferred.

**Audit gate (M0).** Before any rule work, a generated report must give:

- counts of animals, recordings, trials and targets by genotype, and repeats per stimulated→responder pair;
- stimulus parameters, sampling rates, QC failures and uncertain neuron identities;
- trial-to-trial versus animal-to-animal variance;
- stimulation order, inter-stimulus intervals and drift within each recording;
- target-by-fold coverage tables for at least three split schemes, including leave-one-animal-out;
- the calcium indicator's identity and the data that constrain its kinetics;
- neuron-ID overlap across atlas, connectome and annotations.

**Go/no-go:** at least one defined set of targets must support genuine animal-held-out evaluation with nontrivial variability. Otherwise the first paper is an identifiability audit. The split scheme is chosen from coverage tables alone, before any model is scored.

**Trials.** A trial is one stimulation producing a vector of simultaneous responses; correlations within it are kept in modeling and bootstrapping. Each trial records its position in the recording (index, elapsed time, time since and target of the previous stimulation). The stimulated neuron's own trace (autoresponse) is a separate input channel, never a scored responder output. Unobserved, uninterpretable and near-noise responses are distinguished, and near-null responses are never filtered out of primary metrics.

## 4. Model and prediction tasks

A candidate is a program R with parameters θ, run on a typed connectome and read out through an explicit calcium observation model.

```latex
X_{t+\Delta t}=F_{R,\theta}(X_t,\,G_a,\,u_{a,r}(t),\,z_a,\,\xi_t),
\qquad
Y_{i,k}\sim p_\phi\big(\cdot \mid H_\phi[X_i]_{t_k},\,\eta_a\big)
```

X is latent neural state, G the animal's typed graph, u the stimulus, z animal metadata, ξ optional process noise, H the observation operator, η per-animal nuisance and φ observation parameters. Trials in one recording are sequential, so the state at trial onset depends on the previous trial and on stimulation history s.

**Typed connectome.** Chemical, gap-junction, candidate modulatory and neuromuscular edges are kept as separate edge types with counts, confidence and provenance. Unknown synaptic signs are parameters with priors, never inferred from a cell name.

**Neural state.** Up to five registers per neuron: activation v, calcium c, adaptation h, local memory ℓ and modulatory state ρ. Each rule declares the registers it uses; absent ones cost nothing.

**Measurement.** Calcium follows a leaky filter of activation and maps to fluorescence through a per-animal baseline and gain, with Student-t or temporally correlated noise. The indicator's impulse response is a separate, fitted stage shared by all neurons. A fitted v is never reported as measured voltage.

| Task | Inputs at prediction time | Predicted | Role |
| --- | --- | --- | --- |
| T0 | Everything, same trial | Same trace | Debugging only |
| T1 | Known animal and target | Held-out trials of training animals | Weak generalization |
| **T2a (primary)** | Stimulus design, covariates, history, measured autoresponse | All responders in unseen animals | Propagation given achieved stimulation |
| T2b | Stimulus design, covariates, history only | Responders (and autoresponse) in unseen animals | Propagation plus stimulation efficacy |
| T3 | Anatomy and priors; target never fit | Responses to held-out targets | Compositionality |
| T4 | Frozen wild-type model plus a declared intervention transform | Mutant or perturbed responses | Mechanistic stress test |
| T5 | Sensory input and internal state | Neural and body responses, closed loop | Long-horizon emulation |

T2a and T2b are always reported separately. No task may calibrate on a test animal's post-stimulus responder traces; the autoresponse is an input, not a responder.

## 5. Baselines and the indicator control

Before any rule search, a narrow test asks whether responders to one stimulus share temporal dynamics. Each responder i to target j is modeled as a stimulus convolved with a kernel:

```latex
\hat y_{ij}(t)=b_{ij}+a_{ij}\,(u_j * k_{ij})(t)
```

In T2a, u is the measured autoresponse; in T2b, the nominal stimulus passed through a fitted efficacy model. The indicator stage is always explicit.

| Model | Description |
| --- | --- |
| B0 Null | Baseline, drift and noise only |
| B1 Shared kernel | One kernel per stimulated target; optional small per-pair delays; strict variant uses a global kernel bank |
| B2 Low-rank bank | Each kernel is a mix of K shared causal kernels, K ∈ {1, 2, 4, 8} |
| B3 Independent | One regularized kernel per pair, matched shrinkage |
| B4 Linear network | dx/dt = (−D + W)x + Bu on anatomical support, fixed or learned constrained W |
| B5 Rate network | Leaky saturating units on fixed topology, optional adaptation |
| B6 Latent factors | A few network-wide state-space factors, to rule out trivial low-rank explanations |

**History variants.** Every baseline is fitted with and without a declared stimulation-history term (for example gain drift with stimulation index, or carryover of recent autoresponses). The form is chosen on training folds; the simpler variant is kept if history does not help.

**Primary statistic.** Paired per-animal difference in masked predictive negative log likelihood, candidate versus baseline, with per-target results and an animal-level confidence interval. Correlation is never the main metric.

**Indicator-kinetics control.** Shared kernels can come from the sensor rather than the neurons. In T2b the shared indicator filter appears directly in every kernel. In T2a a linear indicator would cancel between input and output, but saturation breaks that, and the indicator's low-pass filtering buries fast kernel components in noise, which regularization then pulls toward a common shape. Before outer scoring:

1. Estimate the indicator impulse response (and saturation) separately, from training autoresponses and published kinetics.
2. Compute and report the resolvable kernel bandwidth at the measured noise level.
3. Fit B1–B3 with the explicit indicator stage; report shared-kernel timescales beside the indicator's.
4. Run the pipeline on synthetic data with distinct neural kernels behind the same indicator and noise; report the false-sharing rate.

H1 holds as a neural claim only if T2b's shared kernels differ from the indicator beyond its uncertainty and the false-sharing rate is below a preregistered threshold.

**Exit gate to rule search:** frozen animal-wise benchmark; competent, calibrated baselines including near-null cases; indicator control and history comparison done; at least one falsifiable sharing hypothesis still standing. If independent kernels win fairly, the grammar must allow more neuron-specific state.

## 6. Rule language (WRL)

WRL (Worm Rule Language) is a small, typed, deterministic-by-default language for update rules on biological multigraphs, restricted enough that searching it stays interpretable and tractable.

**Types.** Unit-annotated scalars, booleans, small bounded integers, K-valued states, neuron and edge types (chemical, gap, modulatory, neuromuscular), signals at declared times, durations and counts. Every operation checks units.

**Operators.** Constants and inputs; bounded arithmetic (no unchecked division); nonlinearities (relu, tanh, sigmoid, threshold, piecewise); typed neighborhood sums (`sum_in`, `sum_gap`, `sum_mod`); temporal operators (delay, leaky integration, decay, hold); local branching; noise (stochastic tier only); and observation operators. Forbidden: access to test data, undeclared global broadcast, unrestricted recursion, code generation, and neuron-ID lookup tables in the lowest tier.

**Grammar tiers.** Each tier unlocks only when the one below leaves reproducible residual structure the data can evaluate.

1. G0: truth-table cellular automata (synthetic tests only).
2. G1: stable continuous local rules with thresholds, saturation, signed edges, optional delay.
3. G2: one or two memory registers (adaptation, synaptic state).
4. G3: rules selected by cell or transmitter type, never by neuron ID.
5. G4: sparse receptor-dependent modulatory fields.
6. G5: process noise and between-animal parameter distributions.

```yaml
# Illustrative only
rule_id: leak_exc_inh_adapt_v1
neuron_state: {v: {init: 0.0}, h: {init: 0.0}}
inputs:
  exc: {op: sum_in, edge: chemical_exc, signal: v_delayed}
  inh: {op: sum_in, edge: chemical_inh, signal: v_delayed}
parameters:
  tau_v: {unit: s, lower: 0.005, upper: 10.0}
  tau_h: {unit: s, lower: 0.01, upper: 30.0}
  gain: {lower: 0.0, upper: 20.0}
  adapt: {lower: 0.0, upper: 10.0}
update:
  drive: "gain * (exc - inh + stimulus) - adapt * h"
  v_next: "v + dt/tau_v * (-v + tanh(drive))"
  h_next: "h + dt/tau_h * (-h + max(v, 0))"
observation: {operator: calcium_linear_v1}
```

The compiler rejects explicit-Euler updates that violate a declared dt/τ stability bound. Excitatory and inhibitory edge sets need priors or inferred signs.

**Canonical form.** Parse to a typed AST, lower to SSA-like IR, normalize commutative operators, fold constants, remove dead registers and common subexpressions, then hash: `program_hash = SHA256(canonical_IR)`. Programs that behave the same in tests may be grouped, but never called formally equivalent without proof.

**Complexity.** Defined once and used everywhere (objective, prior, Pareto front):

```latex
L_{total}=L_{struct}+L_{params},\qquad L_{struct}=L_{AST}+L_{topology}+L_{dispatch}
```

L\_params charges each trainable parameter at its declared precision. A three-line rule with 100,000 per-pair values is not low-complexity.

**Hard constraints.** Positive time constants and units; non-negative, symmetric gap-junction conductance by default; zero flow on absent chemical edges unless a modulatory route is declared; bounded or stable dynamics; finite, explicit delays. Soft priors (transmitter, receptor, known signs) carry provenance and never leak held-out outcomes.

## 7. Simulation runtime

The runtime is synchronous, fixed-step and fully specified, so a program's output never depends on evaluation order.

**Step order.**

1. Apply stimulation and sensory input at time t.
2. Snapshot all old states and delay buffers.
3. Aggregate chemical messages from old states.
4. Apply gap-junction coupling with a stable (semi-implicit) discretization.
5. Update local registers without reading partially updated neighbors.
6. Advance calcium and observation states.
7. Write new states and buffers.
8. Sample observables at the configured acquisition times.

**Data layout.** Contiguous per-register arrays; incoming-edge CSR for chemical synapses with documented, tested orientation; a separate symmetric list for gap junctions; delay ring buffers only when delays are enabled; sparse stimulus input.

**Gap junctions.** Modeled as q\_i = Σ\_j g\_ij (v\_j − v\_i), never as two directed chemical edges. Uniform states must stay uniform.

**Delays.** Quantized to ticks, quantization error reported, no negative delays, declared history initialization.

**Stability.** Every program is checked for domain errors, NaNs, unbounded growth and dt versus dt/2 convergence before fitting. Numerical instability (a bug) is kept distinct from dynamical instability (possibly real).

**Determinism.** Deterministic by default. Stochastic rules use a counter-based RNG seeded by (experiment, program hash, animal, trial, replicate), with distributions implemented explicitly because C++ standard-library distributions differ across platforms. Comparisons use common random numbers; predictive uncertainty comes from repeated rollouts.

**Initial conditions.** Four declared policies: steady state under baseline; inferred from a fixed pre-stimulus window; drawn from a distribution learned on training animals; or history-conditioned, carrying state from the previous stimulation in the recording. History-conditioned is the default unless the audit finds no carryover. Test-time initial states are never fitted to post-stimulus responder traces.

**Conformance.** A scalar C++ reference interpreter and an independent scalar Python implementation define correct behavior on 3–8-neuron graphs: excitation and inhibition, gap diffusion, delay wraparound, adaptation, calcium filtering and sampling, impulses, zero inputs, edge deletions, truth tables and masks. The differentiable Python simulator and every optimized backend must match both within tolerance. Performance targets are set from measurements on the target Apple Silicon machine, not promised in advance.

## 8. Search and inference

Structure (which operators and registers exist) and parameters (their values) are searched separately, inside a nested animal-wise protocol whose outer test data the search never sees.

**Objective (training folds only).**

```latex
\mathcal J(R,\theta,\phi)=-\log p(Y_{train}\mid R,\theta,\phi,G,U)+\lambda\,L_{total}(R,\theta)+\gamma\,P_{bio}(R,\theta)
```

There is no other complexity penalty, so parameters are charged once. Hyperparameters and rule choice use inner folds; the outer fold is scored once, after selection is frozen.

**Search phases.**

1. S0: exhaustive enumeration of small G0/G1 programs under a bit budget, deduplicated by canonical hash.
2. S1: local mutations (add or remove an operator or register, swap threshold for saturation, change parameter sharing).
3. S2: guided proposals (surrogates, grammar-based Monte Carlo, evolutionary search) that keep diversity.
4. S3: parameter fitting by gradients where smooth, otherwise derivative-free or likelihood-free methods.
5. S4: keep several candidates when data cannot separate them.
6. S5: choose experiments that separate the survivors (§9).

**Cheap gates, in order:** type and unit checks → canonical deduplication → stability on a synthetic graph → budget limits → pilot fit on a training subset → full inner-fold fit and validation → diversity-aware Pareto filter → outer test for frozen finalists only. Limits on nodes, registers, depth and parameters are configuration, not advice.

**Parameter sharing**, strongest to weakest: global; per rule family or cell type; per transmitter class; hierarchical with shrinkage; per neuron (charged); per edge (mainly for baselines). Parameters are optimized in bounded transformed domains (for example τ = softplus(raw) + τ\_min) with multiple starts.

**Gradients.** Fitting runs on a differentiable JAX or PyTorch simulator in float64 on CPU for confirmatory fits; it must pass the same conformance suite as the C++ interpreter. Gradients are checked against finite differences; truncation and clipping are recorded. Discrete tiers use exact enumeration or evolutionary search, and evaluation always uses the true forward semantics.

**Program posterior.** An approximate posterior p(R, θ | D) ∝ p(D | R, θ) p(θ | R) 2^−L\_struct(R), a chosen prior and not a claim about nature. Large spaces keep weighted top candidates, clearly labeled as approximations.

**Search-volume overfitting.** Testing a million programs can overfit validation data. Controls: nested CV, target holdouts, a frozen benchmark with few final submissions, a full log of attempted candidates, search-budget curves, a random-grammar control and repeated seeds.

**Reporting.** A Pareto front of L\_total against held-out NLL, not one weighted score. A slightly worse but much shorter rule can matter scientifically; a small saving that badly hurts causal prediction does not.

## 9. Hypotheses, equivalence and shortcuts

When several programs fit equally well, the project keeps them all, finds the experiments that would separate them, and states equivalence only relative to a declared observer.

**Hypothesis graph.** A directed graph over candidate programs, fitted parameters, beliefs and data; edges are structural edits, alternatives or posterior updates. Each hypothesis records its program hash, parameter posterior, scores, constraint violations, posterior weight and predicted responses. Each proposed experiment records intervention, targets, waveform, timing, readout, controls, feasibility limits, cost, expected information gain and predictions per hypothesis. The graph describes the state of the inquiry, not the worm.

**Experiment selection.**

```latex
a^*=\arg\max_{a\in\mathcal A_{feasible}}\big[I(R;Y_a\mid D)-\lambda_c\,\mathrm{Cost}(a)\big]
```

Expected information gain is estimated by Monte Carlo over candidates, with error bars; visibly different predictions drowned in noise score low. Allowed choices: stimulated neuron, pulse amplitude, duration and pattern within apparatus limits, multi-target stimulation if feasible, controls, and validated genetic or receptor perturbations. Retrospective selection may see an experiment's design but not its outcome, records its rank among all eligible experiments, and is never called prospective. The tool proposes experiments; it does not run a lab.

**Observer.** A declared protocol O = (allowed interventions, measurement operator, horizon and sampling, divergence measure, tolerance ε). Two models are indistinguishable under O if their predicted measurement distributions stay within ε for every tested intervention. Only empirical indistinguishability over tested interventions is ever claimed.

| Level | Observer | Permitted claim |
| --- | --- | --- |
| F0 | Fixed input, summary score | Similar summary behavior |
| F1 | Calcium traces, known stimulation | Similar dynamics on tested trials |
| F2 | New targets or pulse schedules | Similar held-out intervention response |
| F3 | Ablations and mutants | Similar perturbation-conditioned behavior |
| F4 | Electrophysiology, spatial dynamics | Finer physiological similarity |
| F5 | Closed-loop sensorimotor history | Broader embodied similarity |

No level supports claims about awareness or identity. Equivalence classes are reported as cliques or robust clusters, not transitive closures of noisy thresholds.

**Minimal necessary detail.** Ablate one feature class at a time (cell-specific to type-shared parameters, detailed to shared kinetics, per-edge to count-based weights, specific to generic modulation, gap junctions removed, proprioception removed). A feature is necessary at level F\_k if removing it causes a replicated, confidence-bounded loss on a preregistered metric.

**Predictive shortcuts.** Whether an observable can be predicted faster than by full simulation is tested, not assumed. For a frozen rule, compare exact simulation against learned direct predictors, reduced-order and linearized models, and event summaries on runtime, calibrated error and memory, training cost included. A coarse variable Z counts only if it predicts its own future under the declared interventions about as well as the full state does. A shortcut for one observable says nothing about the whole trajectory.

## 10. Evaluation protocol

The animal is the unit of independence, every split is grouped by animal, and the outer test is scored once after the model is frozen.

**Splits.** Outer and inner folds grouped by animal, with all sessions and trials of an animal in one fold; scheme chosen from M0 coverage tables (leave-one-animal-out allowed). Target holdouts leave whole stimulated neurons unfit. External tests use an independent dataset, cohort or genotype after the rule is frozen. Strong batch effects get a separate leave-batch-out analysis. Splitting time within a trace never substitutes for animal generalization.

**Primary metric.** Masked predictive negative log likelihood with a correlated-noise likelihood (AR residuals or state space), per-trace covariance, or block scoring, so correlated time samples do not overstate precision. Reported in total, per trace and per animal; T2a and T2b separately.

**Secondary metrics.** Error on standardized traces; sign agreement on well-supported responses; rise time, latency, decay and integrated response where estimable; calibration of 50/80/95% intervals; false positives on null responses; variability across animals; complexity bits and compute cost.

**Fairness.** Every baseline gets a defensible tuning budget and the same observation model where feasible; results are shown at equal compute and at best-within-budget; ablations show whether gains come from grammar, inputs, observation model, fitting or selection.

**Uncertainty.** Animal-level bootstrap (optionally stratified by target group and batch), candidate–baseline differences paired within outer folds, 95% intervals with the number of animal clusters, mixed-effects sensitivity checks, and multiplicity control for secondary families.

**Success criterion (preregistered).** The primary claim holds only if:

1. the frozen model improves the paired animal-level score over the strongest eligible baseline on T2a, with T2b reported alongside;
2. the two-sided 95% interval for that improvement excludes zero;
3. no single animal, small responder subset or post-selection drives the effect;
4. calibration and null-response metrics stay within a preregistered margin;
5. the complexity–performance trade-off is reported;
6. any shared-timescale claim passes the indicator control (§5).

If power is inadequate, report effect sizes and intervals without victory language.

**Negative controls.** Shuffle target labels within matched strata; permute neuron labels preserving graph statistics; degree-preserving edge-weight and network randomization; matched-noise traces; no-modulation models; corrupted stimulus times; metadata-only models; synthetic false-sharing data; stimulation-order shuffles within recordings. Each avoids exchangeability assumptions that make a strawman.

**Leakage checks (automated, CI-enforced).** No shared animals across train and test; no hidden responder frames in test calibration; no test labels in normalization or feature selection; no test results in the program ranker; no duplicate trials across splits; every reported ID maps to immutable raw data. CI deliberately injects a leaked animal and must fail.

## 11. Perturbations and embodiment

Genotype and body tests come only after the neural model passes held-out tests, and each is designed so that a pass cannot come from a flexible nuisance model or decoder.

**Genotypes.** A mutant such as `unc-31` can change release, receiver physiology, development, excitability and stimulus efficacy, so setting one peptide edge to zero is not a valid test. Three model classes are compared: mechanism-only (one restricted component changes); mechanism plus measured nuisance (baseline, observation, excitability); and a flexible genotype model with shrinkage as the upper comparator. A strict transfer claim fits and freezes the wild-type model before mutant data are examined, with any nuisance estimation specified in advance and isolated from the target prediction.

**Modulation tiers.** M0 none; M1 global or regional slow field; M2 a few ligand–receptor channels; M3 spatial diffusion (an optional approximation, likely unidentifiable from current data); M4 cell-specific, state-dependent release only if independently constrained. Comparisons cover sign, latency and shape at fixed autoresponse, expression-compatible versus incompatible pathways, and diffusion versus none. Transcriptomic matching alone never validates a pathway.

**Embodiment loop.** Environment → sensory transduction → neural rule simulator → validated neuromuscular map → body dynamics → proprioception and stimuli back to the neurons. Interfaces: `SensoryInput` (time, environment, pose, sensor map → per-neuron stimulus with uncertainty); `MotorOutput` (motor-neuron states → muscle activation with provenance and saturation flags); `BodyStep` (pose, velocity, activation, environment, dt → next pose, forces, sensory fields). Both open-loop replay and true closed loop are supported.

**Escalating tests.**

1. Muscle pattern: plausible activation timing and dorsoventral relationships under replayed stimuli.
2. Kinematics: drives a simplified body without a trajectory-fitted readout.
3. Full loop: forward–reverse switching, sensory orientation and preserved perturbation effects in a validated body engine.

**Decoder limits.** Any learned neural-to-muscle map reports exactly what is learned and from which data, and is run against decoder-only and randomized-neural controls. If random features can be decoded into realistic motion, motion is not evidence of correct neural rules.

**Behavior endpoints.** Speed and persistence; forward and reverse bout durations; curvature wave frequency, wavelength and phase velocity; turn and omega-turn rates; gradient navigation error; recovery after perturbation; joint neural–kinematic predictiveness. Never a single movement score.

## 12. Software architecture

C++20 owns the rule toolchain, enumeration, reference interpreter and CLI; Python owns data, baselines, differentiable simulation, fitting and reporting; the forward pass moves to C++ only if profiling shows Python rollouts limit search.

**Principles.** Local-first and offline; separate APIs for ingestion, compilation, simulation, inference, selection, evaluation and visualization; raw data and frozen test targets are immutable; every output traces to source version, split, model hash, code version and config; no silent filtering or fallback; each added modality must improve a declared endpoint; no GPU until profiling demands it.

| Component | Language | Inputs → outputs | Invariant |
| --- | --- | --- | --- |
| Importers | Python | Source files → normalized tables | Animal and trial IDs and provenance preserved |
| Graph builder | C++ | Connectome + ID map → typed CSR graph | No invented edges; stable IDs |
| WRL compiler (`ow-ir`) | C++ | WRL source → canonical IR, hash, bits | Type, unit and stability checks |
| Reference interpreter (`ow-sim`) | C++ | IR + graph + inputs → latent traces | Old-state synchronous semantics |
| Differentiable simulator | Python | Same → traces + gradients | Matches the C++ reference |
| Observer and likelihood | C++ / Python | Latent traces → predicted measurements, scores | No access to true test outputs |
| Fitter | Python | Training trials + program → parameters | Training-only calibration |
| Searcher (`ow-search`) | C++ | Grammar + inner folds → candidates | Search budget logged |
| Evaluator | Python | Frozen model + outer test → scores | Scored once after freeze |
| Experiment planner | C++ | Candidates + menu → ranked experiments | Uncertainty included |
| Reports | Python | Frozen artifacts → Markdown, JSON, figures | Every number traceable |

**Repository.**

```text
occamworm/
├── CMakeLists.txt, CMakePresets.json, vcpkg.json, pyproject.toml
├── docs/            SPEC, DATA_CONTRACT, GRAMMAR, SIM_SEMANTICS, BENCHMARK, REPRODUCIBILITY
├── libs/            C++20, namespace occamworm::
│   ├── ow-core  ow-ir  ow-sim  ow-sim-fast  ow-observe  ow-likelihood
│   └── ow-search  ow-infer  ow-experiments  ow-equivalence  ow-data  ow-bindings  ow-cli
├── python/occamworm/
│   ├── importers/   pumpprobe, wormneuroatlas, openworm, wormwideweb
│   ├── baselines/   null, shared/low-rank/independent kernels, linear network, history, indicator
│   ├── sim/  fit/  analysis/  plotting/
├── configs/         datasets, splits, rules, experiments, searches
├── tests/           synthetic_truth, conformance, adversarial, leakage, fuzz, integration
├── benches/  scripts/  notebooks/ (exploratory only)
├── artifacts/       immutable run outputs
└── data/            raw (immutable, untracked), normalized, splits
```

**Core tables (Arrow/Parquet).** `animals` (animal, session, batch, strain, genotype, stage, preparation, source, QC flags); `trials` (trial, animal, target, stimulus timing and amplitude, stimulation index in recording, time since previous stimulation and its target, autoresponse reference, QC); `observations` (trial, neuron, time, ΔF/F, observed, valid, QC, processing version); `edges` (source, target, kind, count, sign or unknown, confidence, reconstruction, stage, provenance). Each run writes a manifest of dataset, split, program, config and code hashes, seed, animal counts, wall time and memory.

**C++ interfaces.** `NeuralProgram` (hash, state layout, `step(context, const old, new)`), `ObservationModel` (predict, log\_prob), `CandidateSearch` (propose, update) and `InterventionPlanner` (rank). They serve the reference path; fast kernels are template-specialized or generated per program shape, with no virtual dispatch in the time loop. Test labels are never reachable from any of them.

**CLI.**

```bash
occamworm data audit --manifest configs/datasets/primary.toml --out artifacts/audit-v1/
occamworm splits build --dataset data/normalized/atlas-v1/ --strategy <frozen-from-audit> --out data/splits/atlas-v1/
occamworm baseline benchmark --config configs/experiments/shared-timescales.toml
occamworm rule inspect --input configs/rules/leak-adapt.yaml
occamworm sim conformance --suite tests/synthetic_truth/
occamworm search run --config configs/searches/g1-small.toml
occamworm evaluate locked --selection artifacts/search-g1-v1/frozen.json
occamworm experiments rank --models artifacts/eval-g1-v1/models/ --menu configs/experiments/feasible.toml
occamworm report build --run artifacts/eval-g1-v1/
```

**C++ toolchain.** C++20 with CMake presets; dependencies pinned in a vcpkg manifest; Python extension built by scikit-build-core with nanobind bindings and explicit array shape and dtype checks. The reference build uses float64 with `-ffast-math` and floating-point contraction off and fixed reduction order. No output may depend on unordered-container order, pointers or thread timing. RAII and value types, no owning raw pointers, bounds-checked reference interpreter. Libraries: GoogleTest, RapidCheck, yaml-cpp, nlohmann/json and a vetted SHA-256. Pinned Clang for reference builds, plus GCC in CI.

**CI on every pull request.** Warnings-as-errors builds on Clang and GCC; clang-format and clang-tidy; C++ unit and property tests under AddressSanitizer and UndefinedBehaviorSanitizer; a short WRL parser fuzz run; Python typing and tests; small synthetic simulation and conformance; leakage assertions; determinism and report checksums; license scan of committed assets. Full synthetic recovery runs nightly.

**Artifacts.** Run IDs are content hashes of raw manifest, schema, split IDs, graph version, program hash, fit config, code revision and seeds. No mutable `latest` pointers in published results.

**Compute.** The network is small, so search volume, not one simulation, is the scaling risk. Order of optimization: correct reference → profile real searches → batched CPU trials → per-shape specialized kernels → batched candidates → parallel workers → Metal (through metal-cpp) only for a sustained bottleneck, batching thousands of candidate–trial runs and passing float32 only after matching the float64 reference. Budgets cap candidates, fit restarts, CPU and GPU time, stored traces and rollouts; runs checkpoint and resume without changing order or RNG mapping.

## 13. Roadmap and tickets

The first scientific release is M0–M1 (about five weeks); everything after it is gated, and a failed gate re-scopes the project rather than relaxing the gate.

Milestones M0–M7 and their gates are in [planning/ROADMAP.md](planning/ROADMAP.md).

Weeks are planning estimates for one focused researcher with coding assistance, not promises of discovery.

**Decision rule.** No animal-wise held-out test possible → publish the coverage audit and seek other data. Baselines uncalibrated → fix the measurement or split model; do not search rules. Grammar cannot recover synthetic rules → fix the language or inference. Rules do not beat strong baselines → study residuals and expand the grammar only on evidence. Rules win → test new targets and interventions.

| Ticket | Deliverable | Done when |
| --- | --- | --- |
| OW-001 | Source downloader and `sources.json` (DOI, hash, date, license) | A clean environment verifies every checksum and reports missing or unlicensed assets |
| OW-002 | `pumpprobe` trial importer with stable IDs, stimulation order and autoresponse channel | Sampled observations round-trip to the official traces |
| OW-003 | Audit report: counts, inclusion flowchart, replication, history and drift, fold coverage, indicator data | Regenerates from the manifest with no hand-typed numbers |
| OW-004 | Immutable animal-wise outer and inner splits | A leaked-animal fixture fails; seed-stable; scheme justified by OW-003 before any scoring |
| OW-005 | Masked correlated-noise scorer with separate indicator stage | Calibrated on well-specified synthetic data; consistent under missing data |
| OW-006 | B0–B3 with history variants, on T2a and T2b | Synthetic data pick the right family; false-sharing rate reported |
| OW-007 | B4 linear network with stable integration | Matches analytic toy solutions; scored on the same folds |
| OW-008 | C++ WRL parser, typed AST, units, canonical IR, bit costs | Unit errors rejected; equivalent reorderings hash identically |
| OW-009 | C++ reference interpreter and differentiable Python simulator (G0/G1) | Both pass conformance and agree within tolerance |
| OW-010 | Length-limited enumerator with canonical cache | Every program in a small budget is visited once up to canonical form |
| OW-011 | Python fitting: bounded transforms, gradient checks, multi-start, L\_params | Synthetic parameters recovered within uncertainty; budget logged |
| OW-012 | Nested search, freeze and locked outer evaluation | No test labels reachable; leak-injection tests fail as expected |
| OW-013 | Neuron and edge annotation adapter with unresolved-ID audit | Conflicts carry provenance instead of forced labels |
| OW-014 | Program posterior and experiment planner | Picks known discriminating stimuli in synthetic pairs, with uncertainty |
| OW-015 | Reports from frozen artifacts only | Any table cell traces to model, fold, sample IDs and code hash |
| OW-016 | Sensory, neural, muscle and body interfaces | A fixed program connects unchanged; decoder-only controls run |

**Dependencies.** Data chain OW-001 → 002 → 003 → 004 → 005 → 006 and 007. Language chain OW-008 → 009 → 010 and 011. OW-012 needs splits, scoring, the network baseline, enumeration, fitting and annotations (OW-013). OW-014 and OW-015 follow OW-012; OW-016 needs OW-009 and OW-013.

**Release boundaries.** Minimum viable scientific release: OW-001 to OW-007 plus audit and reporting, enough for Paper A. Minimum viable engine: OW-001 to OW-013 and OW-015, with G0/G1, synthetic recovery and real-data baseline comparisons. OW-014 adds experiment design; OW-016 adds embodiment.

## 14. Risks, claims and publication

The biggest risks are sparse replication, measurement confounds and overfitting through search; each has a named mitigation, and negative results are publishable outcomes.

| Risk | Mitigation |
| --- | --- |
| Too few animal-level repeats | Coverage audit, restricted eligible cohort, honest intervals |
| Indicator kinetics mimic shared timescales | Separate indicator stage, bandwidth report, false-sharing control |
| Carryover within recordings | History audit, history terms, order-shuffle control |
| Calcium cannot pin down voltage or kinetics | Explicit observation model; electrophysiology where available |
| Non-identifiable rules | Keep candidate populations; select discriminating experiments |
| Grammar cannot express the truth | Misspecification suite; expand tiers only on residual evidence |
| Search overfits validation | Nested grouped CV, frozen external holdout, full search log |
| Connectome sources and signs differ | Source-specific graph manifests; latent signs with priors |
| Extrasynaptic or state confounds | Graduated modulation tiers; hierarchical nuisance; mutant controls |
| Learned decoder fakes behavior | Restricted decoder; decoder-only and shuffled-neural controls |
| Numerical blowup read as emergence | Stability checks, dt convergence, independent reference |
| Memory or undefined-behavior bugs in C++ | Sanitizers, fuzzing, warnings as errors, two compilers |
| Overclaiming unpredictability or equivalence | Shortcut benchmarks; observer-indexed claims only |
| Licensing and provenance | Preserve source licenses; reference by metadata when needed |
| Scope creep toward a full moving worm | Data-first milestones with hard gates |

**Anti-patterns.** Tuning to published mean traces; predicting only significant responses; attributing a mutant's signal loss to one peptide; treating short code as mechanism; using a trained decoder as proof of connectome-driven behavior; re-tuning after viewing the final test set.

**Allowed claims, when tested:** "A short program predicts held-out stimulation responses." "A type-shared rule matches or beats a per-pair model under a stated complexity budget." "Two models are empirically indistinguishable under these interventions and tolerances." "This intervention is predicted to separate these hypotheses."

**Never claimed:** the worm's true code; shared experience; proven unpredictability from a chaotic trace; a single peptide proven by a mutant; connectome emulation proven by movement.

**Papers.** Each is framed in standard methods terms and stands on its held-out results; project terms appear only with their operational definitions.

- **A. Shared temporal primitives** (the first, standalone result): shared versus low-rank versus independent kernels with animal-wise holdout, T2a and T2b, history comparison and the indicator control, stating whether sharing is neural or measurement.
- **B. Program synthesis on anatomical graphs:** the rule language, benchmark, discovered rules and complexity–prediction Pareto front, showing synthesis adds value beyond fitting and observation modeling.
- **C. Interventional equivalence classes:** program posterior, observer-based equivalence and chosen discriminating experiments; a quantified non-identifiability result also counts.
- **Later: minimum measurement requirements**, from corrupting structural and molecular inputs and measuring lost causal fidelity (in silico only).

**Open decisions to freeze before confirmatory runs.**

1. Which atlas releases have trace-level access, and under what licenses.
2. Which target cohort has enough animal-level replication after QC.
3. Which observation and correlated-noise model fits controls without absorbing neural dynamics.
4. Exact calibration information allowed for an unseen animal; which of T2a or T2b leads each paper.
5. Which connectome version is canonical and how alternatives are compared.
6. Which G1 operators and AST budget are frozen before the first outer fold.
7. Practical margins for calibration and null-response metrics.
8. The statistical unit when animals share acquisition batches.
9. Which independent dataset or perturbation provides external validation.
10. Evidence required to call a mechanism neural, extrasynaptic or cell-specific.
11. Which body simulator allows a fixed neuromuscular interface and redistribution.
12. How data and code license conflicts are handled in public releases.
13. Which stimulation-history term, chosen from training audits only.
14. Which indicator model, bandwidth estimate and false-sharing threshold.
15. Which split scheme, chosen from coverage tables, and why.

**References** (not re-verified in this revision; check before release): Randi et al. 2023, *Nature* 623:406–414 (doi:10.1038/s41586-023-06683-4; data doi:10.17605/OSF.IO/E2SYT); Leifer Lab `pumpprobe` and `worm-functional-connectivity` on GitHub; OpenWorm Connectome Toolbox; Worm Neuro Atlas (F. Randi); Atanas et al. 2023, *Cell* (WormWideWeb); Zhao et al. 2024, *Nature Computational Science* 4:978–990 (BAAIWorm).
