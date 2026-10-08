# Embodied worm extension

> Part of the **Occam's Worm specification v0.4.1** · [Spec index](../README.md) · Source: §13

## 13. Embodied worm extension

### 13.1 Why it is phase-gated

BAAIWorm already integrates neural simulation with a physical worm body and environment; its published model includes learned motor-neuron-to-muscle readout machinery. Reproducing a crawling trajectory with a freely fitted decoder is therefore not a convincing new causal emulation result by itself. [R5]

Occam's Worm should first demonstrate predictive competence on neural perturbation data, then close the sensorimotor loop and test whether it still behaves coherently.

### 13.2 Minimal closed-loop architecture

```text
Environment fields and contacts
       | sensory transduction
       v
Neural rule simulator (typed connectome)
       | motor-neuron activity
       v
Validated NMJ/muscle activation map
       | contractile forces
       v
Body dynamics / external environment
       | proprioception and external stimuli
       +-------------------------------------> neural sensory inputs
```

### 13.3 Interfaces

`SensoryInput`:

```text
(time, environment_state, body_pose, known_sensor_map)
    -> stimulus_by_neuron + observation_uncertainty
```

`MotorOutput`:

```text
(motor_neuron_states, time)
    -> muscle_activation + provenance + saturation_flags
```

`BodyStep`:

```text
(body_pose, body_velocity, muscle_activation, environment, dt)
    -> next_pose, next_velocity, forces, sensory_fields
```

The simulator must allow both open-loop replay of recorded sensory histories and genuine closed-loop interaction with a virtual environment.

### 13.4 Three escalating embodiment tests

1. **Muscle-pattern test:** Does the frozen neural model produce plausible activation timing and dorsoventral relationships under replayed stimuli?
2. **Kinematic test:** Can its predicted motor activity control a simplified low-dimensional worm body without an unrestricted target-trajectory-fitted readout?
3. **Full loop:** Can it produce forward/reverse switching and sensory orientation in a validated body/environment engine, and preserve specified perturbation effects?

### 13.5 Motor decoder restrictions

If a learned mapping is needed, report:

- Exactly which parameters are learned.
- Training targets and data source.
- Whether decoder training alone can generate behavior without informative neural dynamics.
- Decoder-only and randomized-neural-input controls.
- Generalization to body parameters, sensory contexts and perturbations absent from training.

A very flexible decoder can mask an inaccurate nervous system; restrict it or explicitly separate controller quality from emulation fidelity.

### 13.6 Quantitative behavior endpoints

- Speed distribution and persistence.
- Forward/reverse bout durations and transitions.
- Body curvature wave frequency, wavelength and phase velocity.
- Turn/omega-turn distributions where available.
- Sensory gradient orientation and navigation error.
- Recovery dynamics under perturbations.
- Joint neural/kinematic trial predictiveness.

Avoid compressing all behavior into a single arbitrary movement score.
