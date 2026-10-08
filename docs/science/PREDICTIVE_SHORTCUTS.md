# Predictive shortcuts

> Part of the **Occam's Worm specification v0.4.0** · [Spec index](../README.md) · Source: §10

## 10. Predictive shortcuts: test rather than assume

### 10.1 Precise, limited question

A recurring idea in the study of complex systems is that some processes cannot be predicted faster than by running them step by step. That idea is a motivation, not a usable result: for finite biological data, we cannot generally prove a dynamical system lacks a faster predictive method. We can empirically test **shortcuts for defined observables and horizons**.

Research questions:

1. Can the state at time `T` be predicted substantially faster than simulating all timesteps?
2. Does a coarse state variable predict selected interventions despite a high-dimensional microstate?
3. Which observables become unreliable as forecasting horizon increases?
4. Are apparently chaotic trajectories actually driven by measurement noise, unmodeled input, or numerical instability?

### 10.2 Shortcut benchmark

For a frozen simulation rule, compare:

- Exact simulator for `T` steps.
- Learned direct `state_t → observable_(t+T)` predictor.
- Reduced-order state-space models.
- Koopman/spectral or linearized approximations.
- Event-level predictive summaries.

Measure elapsed runtime, calibrated predictive error and memory, *including* training cost when evaluating scientific efficiency. A shortcut that works for one narrow observable does not imply reducibility of the full underlying trajectory.

### 10.3 Causal coarse-graining

Given microstate `X` and coarse state `Z=f(X)`, seek

$$
p(Z_{t+\Delta}\mid X_t, do(a))\approx
p(Z_{t+\Delta}\mid Z_t,do(a)),
$$

across a declared intervention set. This is an approximate causal-Markov/sufficiency criterion, not just preservation of variance under PCA.

Explore:

- Graph-informed neuron communities.
- Behavioral latent states (forward, reverse, turn, quiescent).
- Slow neuromodulatory dimensions.
- Neural activity manifolds with explicit intervention tests.
- Symbolically discovered coarse rewrite rules.

Never optimize the coarse representation against the final held-out evaluations.
