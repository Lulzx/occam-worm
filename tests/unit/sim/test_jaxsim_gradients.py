"""OW-009: JAX gradients against central finite differences on tiny programs, plus batching."""

from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np
import pytest

from occamworm.sim.graph import ChemicalEdge, GapJunction, GraphSpec, build_graph
from occamworm.sim.ir import Program
from occamworm.sim.jaxsim import JaxSimulator, initial_array

Compile = Callable[[str], Program]

LEAK_CHEM_CA = """wrl 0.1
tier G1
state v : 1 = 0
state h : 1 = 0
param tau : s = 0.4 in [0.05, 2] trainable bits 10
param gain : 1 = 1.3 in [0, 4] trainable bits 10
param adapt : 1 = 0.5 in [0, 3] trainable bits 10
param tau_h : s = 1.2 in [0.1, 5] trainable bits 10
param tau_ca : s = 0.5 in [0.05, 3] trainable bits 10
input u = stimulus
input e = sum_in(v, exc)
input i = sum_in(v, inh)
next v = leaky_integrate(v, tanh(gain * (u + e - i) - adapt * h), tau)
next h = leaky_integrate(h, v * v, tau_h)
observe calcium_linear_v1(v, tau_ca)
"""

GAP_DELAY = """wrl 0.1
tier G1
state v : 1 = 0
param tau : s = 0.3 in [0.05, 2] trainable bits 10
param g : 1 = 1.5 in [0, 6] trainable bits 10
param k : 1 = 0.7 in [0, 3] trainable bits 10
input u = stimulus
input e = sum_in(v, all)
next v = leaky_integrate(v, sigmoid(u + e + k * delay(v, 2)), tau)
gap v scale g
observe identity_v1(v)
"""

EULER_SMOOTH = """wrl 0.1
tier G1
dt_max 0.1
state v : 1 = 0
state w : 1 = 0
param tau : s = 0.5 in [0.1, 3] trainable bits 10
param a : 1 = 0.8 in [0, 3] trainable bits 10
param b : 1 = 0.3 in [-2, 2] trainable bits 10
input u = stimulus
input e = sum_in(v, exc)
next v = euler_leak(v, clamp(a * (u + e) - b * abs(w), -2.0, 2.0), tau)
next w = euler_leak(w, max(v, b * v), tau)
observe identity_v1(v)
"""

PROGRAMS = {"leak_chem_calcium": LEAK_CHEM_CA, "gap_delay_sigmoid": GAP_DELAY, "euler_clamp_abs": EULER_SMOOTH}


def _setup(program: Program, steps: int = 14, dt: float = 0.1) -> tuple[JaxSimulator, np.ndarray, np.ndarray]:
    spec = GraphSpec(
        neurons=tuple((f"N{k}", "generic") for k in range(5)),
        chemical=(
            ChemicalEdge("N0", "N1", 0.9, 1, 0),
            ChemicalEdge("N0", "N2", 0.5, 1, 2),
            ChemicalEdge("N1", "N3", 0.7, -1, 1),
            ChemicalEdge("N2", "N3", 0.8, 1, 0),
            ChemicalEdge("N3", "N4", 0.6, 1, 3),
            ChemicalEdge("N4", "N0", 0.4, -1, 1),
            ChemicalEdge("N3", "N3", 0.2, 1, 1),
        ),
        gap=(GapJunction("N1", "N2", 1.2), GapJunction("N2", "N4", 0.6), GapJunction("N1", "N2", 0.3)),
    )
    graph = build_graph(spec)
    rng = np.random.default_rng(3)
    stimulus = rng.uniform(-0.8, 1.2, size=(steps, graph.n))
    init = initial_array(program, graph) + rng.uniform(-0.2, 0.2, size=(len(program.registers), graph.n))
    sim = JaxSimulator(program, graph, dt, steps, observed=[0, 2, 3, 4], sample_ticks=[0, 3, 7, 14])
    return sim, stimulus, init


def _loss(
    sim: JaxSimulator, weights: tuple[np.ndarray, np.ndarray]
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    wo, wr = weights

    def loss(theta: jax.Array, stimulus: jax.Array, init: jax.Array) -> jax.Array:
        out = sim.run(theta, stimulus, init)
        return (wo * out.observation**2).sum() + (wr * out.registers).sum()

    return loss


def _weights(sim: JaxSimulator) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    k = len(sim.reported)
    s = len(sim.sample_ticks)
    return rng.uniform(0.5, 1.5, size=(s, k)), rng.uniform(0.5, 1.5, size=(sim.n_registers, s, k))


def _central(f: Callable[[np.ndarray], float], x: np.ndarray, h: float) -> np.ndarray:
    g = np.zeros_like(x)
    for k in range(x.size):
        d = np.zeros_like(x)
        d.flat[k] = h * max(1.0, abs(x.flat[k]))
        g.flat[k] = (f(x + d) - f(x - d)) / (2.0 * d.flat[k])
    return g


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_theta_gradient_matches_central_differences(compile_wrl: Compile, name: str) -> None:
    program = compile_wrl(PROGRAMS[name])
    sim, stimulus, init = _setup(program)
    loss = _loss(sim, _weights(sim))
    theta = np.asarray(program.default_theta())
    grad = np.asarray(jax.grad(loss, argnums=0)(theta, stimulus, init))
    fd = _central(lambda th: float(loss(th, stimulus, init)), theta, 1e-6)
    assert np.all(np.isfinite(grad))
    assert np.abs(grad).max() > 1e-3  # a non-trivial test
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_stimulus_and_initial_state_gradients_match_central_differences(compile_wrl: Compile, name: str) -> None:
    program = compile_wrl(PROGRAMS[name])
    sim, stimulus, init = _setup(program)
    loss = _loss(sim, _weights(sim))
    theta = np.asarray(program.default_theta())
    g_stim, g_init = jax.grad(loss, argnums=(1, 2))(theta, stimulus, init)
    rng = np.random.default_rng(5)
    for _ in range(3):  # random directional derivatives
        v_s, v_i = rng.standard_normal(stimulus.shape), rng.standard_normal(init.shape)
        h = 1e-6

        def along(eps: float, v_s: np.ndarray = v_s, v_i: np.ndarray = v_i) -> float:
            return float(loss(theta, stimulus + eps * v_s, init + eps * v_i))

        fd = (along(h) - along(-h)) / (2 * h)
        analytic = float(np.sum(np.asarray(g_stim) * v_s) + np.sum(np.asarray(g_init) * v_i))
        assert analytic == pytest.approx(fd, rel=1e-6, abs=1e-8)


def test_batched_runs_match_single_runs_and_gradients(compile_wrl: Compile) -> None:
    program = compile_wrl(LEAK_CHEM_CA)
    sim, stimulus, init = _setup(program)
    rng = np.random.default_rng(1)
    stimuli = rng.uniform(-1, 1, size=(3, *stimulus.shape))
    theta = np.asarray(program.default_theta())
    batched = sim.run(theta, stimuli, init)
    assert batched.observation.shape == (3, len(sim.sample_ticks), len(sim.reported))
    assert batched.registers.shape == (3, sim.n_registers, len(sim.sample_ticks), len(sim.reported))
    for b in range(3):
        single = sim.run(theta, stimuli[b], init)
        np.testing.assert_allclose(batched.observation[b], single.observation, atol=1e-14)
        np.testing.assert_allclose(batched.registers[b], single.registers, atol=1e-14)
    # batch of parameter vectors against a batch of initial states
    thetas = np.stack([theta, theta * 0.9, theta * 1.1])
    inits = np.stack([init, init * 0.5, init + 0.1])
    both = sim.observe(thetas, stimulus, inits)
    for b in range(3):
        np.testing.assert_allclose(both[b], sim.observe(thetas[b], stimulus, inits[b]), atol=1e-14)

    def total(th: jax.Array) -> jax.Array:
        return (sim.observe(th, stimuli, init) ** 2).sum()

    grad = np.asarray(jax.grad(total)(theta))
    separate = sum(
        np.asarray(jax.grad(lambda th, b=b: (sim.observe(th, stimuli[b], init) ** 2).sum())(theta)) for b in range(3)
    )
    np.testing.assert_allclose(grad, separate, rtol=1e-12, atol=1e-12)


def test_observe_equals_run_observation(compile_wrl: Compile) -> None:
    program = compile_wrl(GAP_DELAY)
    sim, stimulus, init = _setup(program)
    theta = np.asarray(program.default_theta())
    np.testing.assert_array_equal(sim.observe(theta, stimulus, init), sim.run(theta, stimulus, init).observation)
