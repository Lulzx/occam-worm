# Formal problem statement

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §3

## 3. Formal problem statement

### 3.1 Experiment and observation notation

An experiment is indexed by animal/session `a`, trial `r`, and stimulation target `j`. Let:

- `G_a`: observed or canonical typed anatomical graph for animal/condition `a`.
- `z_a`: known animal metadata (genotype, acquisition setup, strain, experimental batch).
- `u_{a,r}(t)`: optogenetic input waveform, target location and stimulation timing.
- `y^{auto}_{a,r}(t)`: measured calcium trace of the stimulated neuron itself (autoresponse); an input in T2a.
- `s_{a,r}`: stimulation history for trial `r` within its recording: prior targets, their timing and their measured autoresponses.
- `X_{a,r}(t)`: unknown internal state of the neural system.
- `Y_{a,r,i}(t_k)`: measured calcium response for neuron `i`, if visible/identified at frame `k`.
- `M_{a,r,i,k}`: validity mask for each observation.
- `R`: candidate update program; `theta`: shared physical/dynamical parameters.
- `eta_a`: animal/session-specific nuisance effects; `phi`: observation-system parameters.

The forward model is

$$
X_{t+\Delta t}=F_{R,\theta}(X_t,G_a,u_{a,r}(t),z_a,\xi_t),
\qquad
Y_{i,k}\sim p_\phi(\cdot\mid H_\phi[X_i]_{t_k},\eta_a).
$$

`xi_t` denotes optional explicitly specified process noise. The full model is jointly conditional on stimulation and anatomical context, and must include an **observation operator** mapping latent membrane/synaptic dynamics into measured calcium fluorescence.

Because stimulations within a recording are sequential, the state at trial onset is not an independent draw: `X_{a,r}(t_0)` depends on the end state of trial `r-1` and on `s_{a,r}`. [§6.7](../runtime/SIM_SEMANTICS.md) defines the allowed policies for representing this dependence.

### 3.2 Typed connectome

Represent anatomy as a typed multigraph, not one monolithic signed matrix:

$$
G=(V,E_{chem},E_{gap},E_{mod},E_{nmj},A_V,A_E).
$$

- `V`: canonical neuron IDs; possible neurons beyond the stimulation-atlas coverage retained if included in a network simulation.
- `E_chem`: directed chemical synaptic counts and source provenance.
- `E_gap`: electrical junctions with symmetrical default conductance and provenance.
- `E_mod`: candidate extrasynaptic peptide/receptor routes, preferably sparse/prior-weighted rather than declared proven.
- `E_nmj`: validated neuron-to-muscle mapping for later embodiment.
- `A_V`: cell type, transmitter/receptor evidence, anatomy region, coordinates if available.
- `A_E`: counts, confidence, developmental stage, reconstruction method, evidence source.

Missing synaptic sign is an *unknown parameter or constrained categorical uncertainty*, never silently determined by presynaptic cell name alone. Resolve differing canonical IDs and bilateral aliases with an explicit mapping table and version.

### 3.3 Neural state schema

Start with the minimum state sufficient for a family of models:

$$
X_i(t)=(v_i,c_i,h_i,\ell_i,\rho_i),
$$

where `v` is activation/membrane proxy, `c` calcium proxy, `h` adaptation or refractory state, `ell` a compact local memory register, and `rho` optional neuromodulator/receptor state. Each candidate rule declares exactly which registers exist. Absent registers consume **zero** code-length budget and should not be allocated in the compiled simulation.

A large class of candidates has the form

$$
q_i(t)=\sum_{j\to i} w_{ji}\,T_{ji}\bigl(v_j(t-d_{ji}),s_{ji}(t)\bigr)
+q^{gap}_i(t)+q^{mod}_i(t)+u_i(t),
$$

$$
(v_i,h_i,\ell_i)_{t+\Delta t}
=R_\theta\bigl(v_i,h_i,\ell_i,q_i,\operatorname{type}(i)\bigr).
$$

This *general form* does not require every rule to have delays, synaptic state, or modulation.

### 3.4 Measurement model

A minimal calcium observation model is

$$
\dot c_i=-c_i/\tau_{Ca,i}+\alpha_i f_{Ca}(v_i),
\qquad
\hat y_i(t)=b_{a,i}+g_{a,i}\,\mathcal{H}(c_i(t);\phi)+\epsilon_i(t),
$$

where `H` may be linear or saturating, and the noise model can be Student-t or temporally correlated. Filter bandwidth, sampling times, baseline drift and stimulus-to-imaging alignment must be represented.

**Identifiability warning:** calcium traces generally cannot uniquely recover voltage, exact spike timing, membrane conductances or release kinetics. Never interpret a fitted latent `v_i` directly as measured biological membrane voltage unless separately calibrated.

**The indicator is shared by construction.** Every observed neuron is read through the same indicator, so any part of the response time course set by the indicator is common to all responders regardless of neural dynamics. The indicator's impulse response (and saturation, if modeled) must be parameterized separately from neural kernels, estimated from data that constrain it, and reported. Claims about neural timescales are made only about what remains after this stage ([§4.5](BASELINES.md)).

### 3.5 Primary optimization objective

$$
\mathcal J(R,\theta,\phi)=
\underbrace{-\log p(Y_{train}\mid R,\theta,\phi,G,U)}_{\text{predictive fit}}
+\lambda L_{total}(R,\theta)
+\gamma P_{bio}(R,\theta),
\qquad
L_{total}(R,\theta)=L_{struct}(R)+L_{params}(\theta\mid R).
$$

- `L_struct`: bit length of the program structure under a frozen prefix-free encoding (AST, topology overrides, type-dispatch tables; [§5.7](../language/GRAMMAR.md)).
- `L_params`: bit length of the trainable parameters at their declared precision, given the structure ([§5.7](../language/GRAMMAR.md)). This is the only parameter-complexity term; there is no separate parameter penalty, so parameters are not charged twice.
- `P_bio`: penalties or hard constraints from known biophysics and anatomy.

Parameter fitting happens **only on training folds**; hyperparameters and rule search use inner validation folds; the untouched outer fold is inspected only after model selection.

### 3.6 A hierarchy of prediction tasks

| Task | Inputs at inference | Predicted quantities | Interpretation |
|---|---|---|---|
| T0 trace reconstruction | Complete observed input and fit animal | Same animal/trial trace | Debugging only |
| T1 new-trial prediction | Known stimulus target, animal context; no responder future trace | Held-out trials from training animals | Weak generalization |
| T2a new-animal prediction, autoresponse-conditioned **primary** | Stimulus design, public covariates, stimulation history, and the stimulated neuron's measured autoresponse in the test trial | All observed responder traces in unseen animals | Signal propagation given the stimulation actually achieved |
| T2b new-animal prediction, nominal stimulus | Stimulus design, public covariates and stimulation history only | Responder traces (and, separately, the autoresponse) in unseen animals | Propagation plus stimulation efficacy; harder |
| T3 new-stimulus-target prediction | Anatomy and source-independent priors; target never fit | Responses to held-out targets | Stronger rule compositionality |
| T4 cross-condition prediction | WT fit, predefined intervention transform | Mutant/perturbed responses | Mechanistic stress test |
| T5 closed-loop embodiment | Sensory input and internal state | Neural and motor/body responses | Long-horizon emulation test |

T3 inherits the T2a/T2b distinction; every T3 result states which conditioning it uses.

**Why both T2 variants.** Achieved stimulation varies substantially across trials. T2a separates propagation from stimulation efficacy and is the natural primary task for questions about network dynamics. T2b tests whether a model also predicts how strongly a nominal stimulus drives its target, which matters for prospective experiment design ([§8](EXPERIMENT_DESIGN.md)). Report T2a and T2b separately; never average them into one score.

**Invalid shortcut:** neither T2a nor T2b may use per-animal nuisance estimates derived from that animal's hidden post-stimulation responder traces. In T2a the autoresponse is an observed input, not a responder output, so using it is legitimate; using any responder's post-stimulus trace for calibration is not. If an allowed calibration period exists, specify its duration, information and cost before splitting.
