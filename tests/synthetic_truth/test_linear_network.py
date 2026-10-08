"""OW-007: B4 reproduces toy analytical solutions and is stable by construction."""

import jax.numpy as jnp
import numpy as np
import pytest

from occamworm.baselines.data import KERNEL_KNOTS, aggregate, compute_stats, trial_designs
from occamworm.baselines.linear_network import (
    STABILITY_MARGIN,
    B4Model,
    kernels_at_knots,
    network_from_edges,
    system,
)
from occamworm.baselines.synthetic import SyntheticSpec, generate

DT = 0.5
T = KERNEL_KNOTS * DT


def edge(s: str, t: str, kind: str, n: float = 1.0) -> dict[str, object]:
    return {
        "source_neuron_id": s,
        "target_neuron_id": t,
        "edge_kind": kind,
        "synapse_count": n,
        "source_reconstruction": "toy",
    }


def theta_for(tau: float, g_gap: float, g: float) -> jnp.ndarray:
    return jnp.asarray([np.log(np.expm1(tau - 0.1)), np.log(np.expm1(g_gap)) if g_gap > 0 else -50.0, np.arctanh(g)])


def kernel(net_edges: list[dict[str, object]], neurons: list[str], theta: jnp.ndarray, j: str) -> np.ndarray:
    net = network_from_edges(neurons, net_edges, "toy")
    a, b = system(theta, jnp.asarray(net.chem), jnp.asarray(net.gap), learned=False)
    col = neurons.index(j)
    return np.asarray(kernels_at_knots(a, b[:, [col]], DT))[:, :, 0]  # (M, N)


def test_single_synapse_is_an_exponential() -> None:
    tau, g = 2.0, 0.5
    k = kernel([edge("J", "I", "chem")], ["I", "J"], theta_for(tau, 0.0, g), "J")
    w = STABILITY_MARGIN / tau * g
    np.testing.assert_allclose(k[:, 0], w * np.exp(-T / tau), rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(k[:, 1], 0.0, atol=1e-14)  # the stimulated neuron's own state is not driven


def test_two_synapse_chain_is_an_alpha_function() -> None:
    tau, g = 1.5, -0.8
    k = kernel([edge("J", "K", "chem"), edge("K", "I", "chem")], ["I", "J", "K"], theta_for(tau, 0.0, g), "J")
    w = STABILITY_MARGIN / tau * g
    np.testing.assert_allclose(k[:, 2], w * np.exp(-T / tau), rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(k[:, 0], w * w * T * np.exp(-T / tau), rtol=1e-8, atol=1e-14)


def test_gap_pair_has_two_modes() -> None:
    tau, gg = 3.0, 0.4
    k = kernel([edge("J", "I", "gap")], ["I", "J"], theta_for(tau, gg, 0.0), "J")
    fast = 1 / tau + 2 * gg
    np.testing.assert_allclose(k[:, 0], gg / 2 * (np.exp(-T / tau) + np.exp(-fast * T)), rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(k[:, 1], gg / 2 * (np.exp(-T / tau) - np.exp(-fast * T)), rtol=1e-9, atol=1e-14)


def test_stable_for_extreme_parameters() -> None:
    rng = np.random.default_rng(0)
    neurons = [f"N{i}" for i in range(12)]
    edges = [
        edge(neurons[i], neurons[j], kind, float(rng.integers(1, 20)))
        for i in range(12)
        for j in range(12)
        for kind in ("chem", "gap")
        if i != j and rng.random() < 0.3
    ]
    net = network_from_edges(neurons, edges, "toy")
    for learned in (False, True):
        for _ in range(20):
            theta = jnp.asarray(rng.normal(0, 5, 2 + (12 if learned else 1)))
            a, _ = system(theta, jnp.asarray(net.chem), jnp.asarray(net.gap), learned)
            assert np.max(np.linalg.eigvals(np.asarray(a)).real) < 0


def test_fixed_network_recovers_time_constant_from_synthetic_traces() -> None:
    neurons = ["T0", "T1", "R0", "R1", "R2"]
    edges = [
        edge("T0", "R0", "chem", 2),
        edge("T0", "R1", "chem", 1),
        edge("T1", "R2", "chem", 3),
        edge("R0", "R2", "chem", 1),
        edge("R1", "R2", "gap", 2),
    ]
    net = network_from_edges(neurons, edges, "toy")
    true = jnp.asarray([np.log(np.expm1(3.0 - 0.1)), np.log(np.expm1(0.3)), np.arctanh(0.9)])
    a, b = system(true, jnp.asarray(net.chem), jnp.asarray(net.gap), learned=False)
    import jax

    full = np.asarray(jax.vmap(lambda t: jax.scipy.linalg.expm(a * t) @ b)(jnp.arange(60) * DT))  # (60, N, N)
    spec = SyntheticSpec("shared", n_animals=8, n_targets=2, n_responders=3, s_range=(0.002, 0.004), seed=5)
    kern = np.stack([[DT * full[:, 2 + i, j] for i in range(3)] for j in range(2)])  # (targets, responders, 60)
    data, _ = generate(spec, kernels=kern)
    data.targets = ["T0", "T1"]
    data.pairs = [(f"T{j}", f"R{i}") for j in range(2) for i in range(3)]
    labels = {n: n for n in neurons}
    designs = trial_designs(data)
    ps = aggregate(compute_stats(data, designs), np.ones(len(data.animals), bool), len(data.pairs))
    fit = B4Model(data, net, labels, learned=False).fit(ps, 0.0, 0.0, None)
    assert fit.extra["tau_s"] == pytest.approx(3.0, rel=0.05)
    assert fit.extra["g_gap"] == pytest.approx(0.3, rel=0.2)
