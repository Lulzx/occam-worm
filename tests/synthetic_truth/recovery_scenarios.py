"""Known-truth generators for the OW-011 recovery tests: small G1 circuits with a known parameter vector.

Each scenario takes a compiled WRL rule, builds a small graph and several stimulus trials, simulates the noiseless
trace at the true parameters, adds noise from a chosen loss's own noise model and hides a random fraction of the
samples (the mask). The generator is deterministic in its seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from occamworm.fit.losses import ObservationLoss
from occamworm.fit.problem import FitProblem
from occamworm.sim.graph import ChemicalEdge, GapJunction, GraphSpec, StimulusEvent, build_graph
from occamworm.sim.ir import Program
from occamworm.sim.jaxsim import JaxSimulator, initial_array, stimulus_array

FloatArray = npt.NDArray[np.float64]

ADAPT_RULE = """wrl 0.1
rule adapting_leaky_ei
tier G1
state v : 1 = 0
state h : 1 = 0
param tau_v : s = 0.3 in [0.05, 2] trainable bits 12
param gain : 1 = 1.0 in [0, 4] trainable bits 10
param adapt : 1 = 0.5 in [0, 3] trainable bits 10
param tau_h : s = 1.5 fixed
param tau_ca : s = 0.4 fixed
input u = stimulus
input e = sum_in(v, exc)
input i = sum_in(v, inh)
next v = leaky_integrate(v, tanh(gain * (u + e - i) - adapt * h), tau_v)
next h = leaky_integrate(h, relu(v), tau_h)
observe calcium_linear_v1(v, tau_ca)
"""

GAP_RULE = """wrl 0.1
rule gap_leak_chem
tier G1
state v : 1 = 0
param tau : s = 0.5 in [0.05, 3] trainable bits 12
param g_scale : 1 = 1.0 in [0, 6] trainable bits 10
param w : 1 = 0.5 in [0, 2] trainable bits 10
input u = stimulus
input chem = sum_in(v, all)
next v = leaky_integrate(v, u + w * chem, tau)
gap v scale g_scale
observe identity_v1(v)
"""

ADAPT_TRUTH = {"tau_v": 0.35, "gain": 1.6, "adapt": 0.9}
GAP_TRUTH = {"tau": 0.4, "g_scale": 2.5, "w": 0.9}


@dataclass
class Scenario:
    program: Program
    simulator: JaxSimulator
    stimulus: FloatArray
    mask: npt.NDArray[np.bool_]
    truth: dict[str, float]
    clean: FloatArray  # noiseless observation (B, S, K)

    def problem(self, loss: ObservationLoss, noise_seed: int, **kwargs: object) -> FitProblem:
        rng = np.random.default_rng(noise_seed)
        target = self.clean + loss.sample_noise(rng, self.mask)
        return FitProblem(self.program, self.simulator, self.stimulus, target, loss, mask=self.mask, **kwargs)  # type: ignore[arg-type]


def _circuit(n: int, seed: int) -> GraphSpec:
    rng = np.random.default_rng(seed)
    neurons = tuple((f"N{k}", "sensory" if k < 3 else "inter") for k in range(n))
    chemical = []
    for post in range(n):
        for pre in rng.choice([k for k in range(n) if k != post], size=3, replace=False):
            chemical.append(
                ChemicalEdge(
                    f"N{pre}",
                    f"N{post}",
                    float(rng.uniform(0.3, 1.2)),
                    int(rng.choice([1, 1, -1])),
                    int(rng.integers(0, 3)),
                )
            )
    gap = tuple(GapJunction(f"N{k}", f"N{(k + 1) % n}", float(rng.uniform(0.5, 1.5))) for k in range(0, n, 2))
    return GraphSpec(neurons, tuple(chemical), gap)


def build_scenario(
    program: Program,
    truth: dict[str, float],
    n_neurons: int = 8,
    n_trials: int = 6,
    n_steps: int = 100,
    dt: float = 0.1,
    missing: float = 0.1,
    seed: int = 0,
) -> Scenario:
    graph = build_graph(_circuit(n_neurons, seed))
    sim = JaxSimulator(program, graph, dt, n_steps)
    rng = np.random.default_rng(seed + 1)
    stimuli = []
    for b in range(n_trials):
        events = [
            StimulusEvent(
                f"N{b % 3}", int(rng.integers(0, 10)), int(rng.integers(25, 60)), float(rng.uniform(0.8, 2.0))
            ),
            StimulusEvent(
                f"N{(b + 1) % 3}", int(rng.integers(30, 50)), int(rng.integers(60, 90)), float(rng.uniform(-1.0, 1.5))
            ),
        ]
        stimuli.append(stimulus_array(events, graph, n_steps))
    stimulus = np.stack(stimuli)
    theta = np.asarray(program.theta_from_dict(truth))
    clean = np.asarray(sim.observe(theta, stimulus, initial_array(program, graph)), dtype=np.float64)
    mask = rng.random(clean.shape) > missing
    return Scenario(program, sim, stimulus, mask, dict(truth), clean)
