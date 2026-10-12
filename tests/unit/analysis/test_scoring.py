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


def test_resolvable_bandwidth_integrates_window_and_trials() -> None:
    """White noise: SNR(0) = n N A^2 / sigma^2; more trials, a longer window or a faster indicator widen the band."""
    ind, dt = Indicator(0.4, 2.5), 0.5
    r = resolvable_bandwidth(ind, 0.07, 0.2, 0.0, dt, n_trials=2, n_window=60)
    assert r["snr_at_zero_hz"] == pytest.approx(2 * 60 * 0.07**2 / 0.2**2)
    # at f_res the SNR crosses 1: 2 * 60 * 0.5 * (A |H|)^2 / sigma^2 = 1
    w = 2 * np.pi * r["f_resolvable_hz"]
    gain2 = 1 / ((1 + (w * 0.4) ** 2) * (1 + (w * 2.5) ** 2))
    assert 60 * 0.07**2 * gain2 / 0.2**2 == pytest.approx(1.0, rel=0.02)
    assert resolvable_bandwidth(ind, 0.07, 0.2, 0.0, dt)["f_resolvable_hz"] == 0.0
    more = resolvable_bandwidth(ind, 0.07, 0.2, 0.0, dt, n_trials=10, n_window=60)["f_resolvable_hz"]
    slow = resolvable_bandwidth(Indicator(1.0, 8.0), 0.07, 0.2, 0.0, dt, n_trials=2, n_window=60)["f_resolvable_hz"]
    assert slow < r["f_resolvable_hz"] < more


def test_paired_difference_resamples_animals() -> None:
    base = {f"a{i}": 10.0 + i for i in range(30)}
    cand = {f"a{i}": 9.0 + i for i in range(30)}
    r = paired_difference(base, cand, reps=500)
    assert r["mean"] == pytest.approx(1.0)
    assert r["mean_ci"] == pytest.approx([1.0, 1.0])
    assert r["n_animals"] == 30
    with pytest.raises(ValueError):
        paired_difference(base, {"a0": 1.0})


def test_noise_objective_matches_ar1_nll_and_its_gradient() -> None:
    from occamworm.analysis.scoring import _flatten, noise_objective

    rng = np.random.default_rng(4)
    e = rng.standard_normal((30, 20)) * 0.1
    valid = rng.random((30, 20)) < 0.7
    s = rng.uniform(0.01, 0.1, 30)
    x = np.array([0.3, -1.0, 0.4])
    flat = _flatten(e, valid)
    f, g = noise_objective(x, flat, s, 0.1)
    m = NoiseModel(float(np.exp(x[0])), float(np.exp(x[1]) * 0.1), float(np.tanh(x[2])))
    assert f == pytest.approx(ar1_nll(e, valid, m.sigma(s), m.phi).sum() / valid.sum(), rel=1e-12)
    for k in range(3):
        d = np.zeros(3)
        d[k] = 1e-6
        num = (noise_objective(x + d, flat, s, 0.1)[0] - noise_objective(x - d, flat, s, 0.1)[0]) / 2e-6
        assert g[k] == pytest.approx(num, rel=1e-5, abs=1e-9)


def test_indicator_fit_survives_near_alpha_responses() -> None:
    """Slow, noisy alpha-like means drove the old (tau_r, tau_d - tau_r) fit to tau_d == tau_r and a crash."""
    from occamworm.baselines.indicator import fit_to_mean_response

    dt = 0.5
    t = np.arange(60) * dt
    rng = np.random.default_rng(0)
    for _ in range(40):
        tau = rng.uniform(3, 40)
        fit, amp = fit_to_mean_response((t / tau) * np.exp(1 - t / tau) + rng.normal(0, 0.02, t.size), dt)
        assert 0 < fit.tau_r < fit.tau_d and np.isfinite(amp)


def test_configured_indicator_follows_the_experiment_stage() -> None:
    from occamworm.baselines.indicator import Indicator, configured_indicator

    pub = {"tau_r_s": 0.3, "tau_d_s": 2.0}
    assert configured_indicator({"observation": {"indicator_stage": "published"}, "indicator": pub}) == Indicator(
        0.3, 2.0
    )
    assert configured_indicator({"observation": {"indicator_stage": "separate_fitted"}}) is None
    with pytest.raises(ValueError):
        configured_indicator({"observation": {"indicator_stage": "guess"}})


def test_published_half_decay_range_brackets_the_central_kernel() -> None:
    from occamworm.baselines.control import _published_half_decay_range, _timescales
    from occamworm.baselines.indicator import Indicator

    dt, n = 0.1, 400
    lo, hi = _published_half_decay_range({"tau_r_range_s": [0.1, 0.5], "tau_d_range_s": [1.0, 4.0]}, n, dt)
    mid = _timescales(Indicator(0.3, 2.0).kernel(n, dt), dt)["half_decay_s"]
    assert mid is not None and lo < mid < hi


def test_student_t_nll_reduces_to_gaussian_and_gradient_is_exact() -> None:
    from occamworm.analysis.scoring import _flatten, fit_noise, interval_coverage, noise_objective_t

    rng = np.random.default_rng(5)
    e = rng.standard_normal((30, 20)) * 0.1
    valid = rng.random((30, 20)) < 0.7
    s = rng.uniform(0.01, 0.1, 30)
    sig = np.sqrt(0.5**2 * s**2 + 0.05**2)
    assert ar1_nll(e, valid, sig, 0.4, nu=1e7) == pytest.approx(ar1_nll(e, valid, sig, 0.4), abs=1e-4)
    x = np.array([0.3, -1.0, 0.4, 1.2])
    flat = _flatten(e, valid)
    f, g = noise_objective_t(x, flat, s, 0.1)
    m = NoiseModel(float(np.exp(x[0])), float(np.exp(x[1]) * 0.1), float(np.tanh(x[2])), float(np.exp(x[3])))
    assert f == pytest.approx(ar1_nll(e, valid, m.sigma(s), m.phi, m.nu).sum() / valid.sum(), rel=1e-12)
    for k in range(4):
        d = np.zeros(4)
        d[k] = 1e-6
        num = (noise_objective_t(x + d, flat, s, 0.1)[0] - noise_objective_t(x - d, flat, s, 0.1)[0]) / 2e-6
        assert g[k] == pytest.approx(num, rel=1e-5, abs=1e-9)
    # heavy-tailed data: the t fit finds small nu, a better likelihood and calibrated coverage
    h = rng.standard_t(3, (400, 30)) * 0.1
    vh = np.ones_like(h, dtype=bool)
    sh = np.full(400, 0.05)
    tm, gm = fit_noise(h, vh, sh, family="student_t_ar1"), fit_noise(h, vh, sh)
    assert tm.nu is not None and 2.0 < tm.nu < 6.0 and gm.nu is None
    assert ar1_nll(h, vh, tm.sigma(sh), tm.phi, tm.nu).sum() < ar1_nll(h, vh, gm.sigma(sh), gm.phi).sum()
    cov = interval_coverage(h, vh, tm.sigma(sh), nu=tm.nu)
    assert abs(cov["0.50"] - 0.5) < 0.03 and abs(cov["0.95"] - 0.95) < 0.02
    assert tm.to_json()["family"] == "student_t_ar1"
