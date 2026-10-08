"""OW-014: the planner picks the known discriminative stimulus and reports information-gain uncertainty."""

import math

import numpy as np
import pytest

from occamworm.equivalence.empirical import (
    Observer,
    distance_matrix,
    divergence,
    indistinguishability,
    maximal_cliques,
    operational_necessity,
)
from occamworm.experiments.planner import Intervention, expected_information_gain, rank_interventions
from occamworm.infer.posterior import Hypothesis, effective_number, posterior_weights, top_diverse

T = 40
SIGMA = np.full(3, 0.05)
PHI = 0.5
t = np.arange(T) * 0.5


def response(amp: float, tau: float) -> np.ndarray:
    return amp * (np.exp(-t / tau) - np.exp(-t / 0.5))


def model(taus: dict[str, float]):  # type: ignore[no-untyped-def]
    def predict(a: Intervention) -> np.ndarray:
        target = a.target_neurons[0]
        tau = taus.get(target, 2.0)
        amp = {"SAME": 0.3, "SPLIT": 0.3, "WEAK": 0.004}[target]
        return np.stack([response(amp, tau)] * 3)[None]  # (particles=1, traces=3, T)

    return predict


MENU = [
    Intervention("optogenetic", (name,), "pulse", "t=0", "calcium", cost=1.0, design=name)
    for name in ("SAME", "SPLIT", "WEAK")
]
M1 = model({"SPLIT": 1.0, "WEAK": 1.0})
M2 = model({"SPLIT": 6.0, "WEAK": 6.0})


def test_planner_selects_the_discriminative_stimulus() -> None:
    ranked = rank_interventions(MENU, [M1, M2], ["m1", "m2"], np.array([0.5, 0.5]), SIGMA, PHI, samples=200)
    assert ranked[0].intervention.target_neurons == ("SPLIT",)
    eig = {p.intervention.target_neurons[0]: p for p in ranked}
    assert eig["SPLIT"].expected_information_gain == pytest.approx(math.log(2), abs=0.02)
    assert abs(eig["SAME"].expected_information_gain) < 1e-9
    # divergent in shape but buried in noise: small gain, with a reported Monte Carlo error
    assert eig["WEAK"].expected_information_gain < 0.3
    assert eig["WEAK"].eig_standard_error > 0


def test_eig_is_bounded_by_prior_entropy() -> None:
    w = np.array([0.9, 0.1])
    means = [M1(MENU[1]), M2(MENU[1])]
    eig, se = expected_information_gain(means, w, SIGMA, PHI, samples=200)
    entropy = -float(np.sum(w * np.log(w)))
    assert eig <= entropy + 3 * se


def test_posterior_prefers_short_programs_at_equal_fit() -> None:
    a = Hypothesis("a", "0.1", l_struct_bits=10, l_params_bits=8, validation_nll=100.0)
    b = Hypothesis("b", "0.1", l_struct_bits=11, l_params_bits=8, validation_nll=100.0)
    w = posterior_weights([a, b])
    assert w[0] == pytest.approx(2 / 3)
    assert effective_number(w) == pytest.approx(1 / (4 / 9 + 1 / 9))


def test_top_diverse_skips_near_duplicates() -> None:
    hs = [Hypothesis(str(i), "0.1", 0, 0, float(i)) for i in range(4)]
    posterior_weights(hs)
    dist = {frozenset(("0", "1")): 0.0}
    picked = top_diverse(hs, lambda x, y: dist.get(frozenset((x.program_hash, y.program_hash)), 1.0), 2, 0.5)
    assert [h.program_hash for h in picked] == ["0", "2"]


def test_divergence_is_gaussian_kl_under_ar1_covariance() -> None:
    from scipy import linalg

    rng = np.random.default_rng(0)
    mu1, mu2 = rng.normal(size=(2, 1, 10))
    sigma = np.array([0.3])
    cov = sigma[0] ** 2 * linalg.toeplitz(PHI ** np.arange(10))
    diff = (mu1 - mu2)[0]
    assert divergence(mu1, mu2, sigma, PHI) == pytest.approx(0.5 * diff @ np.linalg.solve(cov, diff))


def test_equivalence_classes_are_cliques_not_closures() -> None:
    adj = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)  # 0~1, 1~2, but not 0~2
    assert maximal_cliques(adj) == [[0, 1], [1, 2]]
    obs = Observer("calcium-known-targets", "F1", tuple(MENU), "calcium", T, epsilon=1.0)
    d = distance_matrix([lambda a: M1(a)[0], lambda a: M2(a)[0], lambda a: M1(a)[0]], obs, SIGMA, PHI)
    res = indistinguishability(d, obs.epsilon)
    assert res["maximal_cliques"] == [[0, 2], [1]]


def test_operational_necessity() -> None:
    full = {f"a{i}": 0.0 for i in range(20)}
    worse = {f"a{i}": 5.0 + 0.1 * i for i in range(20)}
    assert operational_necessity(full, worse, margin=1.0)["necessary"]
    assert not operational_necessity(full, full, margin=1.0)["necessary"]
