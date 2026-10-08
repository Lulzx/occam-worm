"""OW-005: masked scoring is exact under missing data and a well-specified model is calibrated."""

import numpy as np
import pytest
from scipy import stats

from occamworm.analysis.scoring import NoiseModel, ar1_nll, block_nll, fit_noise, interval_coverage
from occamworm.analysis.uncertainty import paired_difference
from occamworm.baselines.indicator import (
    Indicator,
    difference_of_exponentials,
    fit_to_mean_response,
    resolvable_bandwidth,
)


def ar1(rng: np.random.Generator, n: int, t: int, sigma: np.ndarray, phi: float) -> np.ndarray:
    e = np.empty((n, t))
    e[:, 0] = rng.standard_normal(n)
    for k in range(1, t):
        e[:, k] = phi * e[:, k - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(n)
    return e * sigma[:, None]


def test_masked_nll_equals_marginal_gaussian_of_observed_samples() -> None:
    rng = np.random.default_rng(0)
    t, phi = 12, 0.7
    for _ in range(20):
        sigma = rng.uniform(0.5, 2.0, 1)
        e = ar1(rng, 1, t, sigma, phi)
        valid = rng.random((1, t)) < 0.6
        valid[0, rng.integers(t)] = True
        idx = np.nonzero(valid[0])[0]
        cov = sigma[0] ** 2 * phi ** np.abs(idx[:, None] - idx[None, :])
        expected = -stats.multivariate_normal(np.zeros(idx.size), cov).logpdf(e[0, idx])
        assert ar1_nll(e, valid, sigma, phi)[0] == pytest.approx(expected, rel=1e-10)


def test_fully_masked_trace_scores_zero() -> None:
    e = np.ones((1, 5))
    assert ar1_nll(e, np.zeros((1, 5), bool), np.ones(1), 0.5)[0] == 0.0


def test_block_score_of_single_samples_is_independent_gaussian() -> None:
    rng = np.random.default_rng(1)
    e = rng.standard_normal((3, 8))
    valid = rng.random((3, 8)) < 0.7
    sigma = np.array([1.0, 2.0, 0.5])
    expected = -np.where(valid, stats.norm(0, sigma[:, None]).logpdf(e), 0.0).sum(axis=1)
    np.testing.assert_allclose(block_nll(e, valid, sigma, 0.9, block=1), expected, rtol=1e-12)


@pytest.mark.parametrize("missing", [0.0, 0.4])
def test_noise_fit_recovers_parameters_and_is_calibrated(missing: float) -> None:
    rng = np.random.default_rng(2)
    n, t = 4000, 60
    true = NoiseModel(a=1.5, b=0.02, phi=0.6)
    s = rng.uniform(0.01, 0.05, n)
    e = ar1(rng, n, t, true.sigma(s), true.phi)
    valid = rng.random((n, t)) >= missing
    fit = fit_noise(e[:2000], valid[:2000], s[:2000])
    assert fit.a == pytest.approx(true.a, rel=0.05)
    assert fit.b == pytest.approx(true.b, rel=0.25)
    assert fit.phi == pytest.approx(true.phi, abs=0.03)
    cov = interval_coverage(e[2000:], valid[2000:], fit.sigma(s[2000:]))
    for level, frac in cov.items():
        assert frac == pytest.approx(float(level), abs=0.02)


def test_true_mean_beats_shifted_mean_under_missing_data() -> None:
    rng = np.random.default_rng(3)
    n, t = 500, 40
    sigma = np.full(n, 0.1)
    mu = np.sin(np.linspace(0, 3, t))[None, :] * 0.2
    y = mu + ar1(rng, n, t, sigma, 0.5)
    valid = rng.random((n, t)) < 0.6
    good = ar1_nll(y - mu, valid, sigma, 0.5).sum()
    bad = ar1_nll(y - 0.8 * mu, valid, sigma, 0.5).sum()
    assert good < bad


def test_indicator_fit_recovers_time_constants() -> None:
    dt = 0.5
    t = np.arange(60) * dt
    y = 0.7 * difference_of_exponentials(t, 0.6, 4.0)
    fit, amp = fit_to_mean_response(y, dt)
    assert fit.tau_r == pytest.approx(0.6, rel=1e-4)
    assert fit.tau_d == pytest.approx(4.0, rel=1e-4)
    assert amp == pytest.approx(0.7, rel=1e-6)


def test_indicator_apply_is_causal_convolution() -> None:
    ind = Indicator(0.5, 3.0)
    x = np.zeros((1, 20))
    x[0, 4] = 1.0
    y = ind.apply(x, 0.5)
    np.testing.assert_allclose(y[0, :4], 0.0)
    np.testing.assert_allclose(y[0, 4:], ind.kernel(16, 0.5))


def test_resolvable_bandwidth_shrinks_with_noise() -> None:
    ind = Indicator(0.5, 3.0)
    quiet = resolvable_bandwidth(ind, 0.5, 0.01, 0.5, 0.5)["f_resolvable_hz"]
    noisy = resolvable_bandwidth(ind, 0.5, 0.2, 0.5, 0.5)["f_resolvable_hz"]
    assert quiet > noisy


def test_paired_difference_resamples_animals() -> None:
    base = {f"a{i}": 10.0 + i for i in range(30)}
    cand = {f"a{i}": 9.0 + i for i in range(30)}
    r = paired_difference(base, cand, reps=500)
    assert r["mean"] == pytest.approx(1.0)
    assert r["mean_ci"] == pytest.approx([1.0, 1.0])
    assert r["n_animals"] == 30
    with pytest.raises(ValueError):
        paired_difference(base, {"a0": 1.0})
