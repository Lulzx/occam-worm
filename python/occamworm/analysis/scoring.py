"""Masked proper scoring under temporally correlated noise (§11.2, OW-005).

Residuals ``e = y - mu`` of one trace are modelled as a stationary Gaussian AR(1) process sampled on the volume
grid, with per-trace marginal variance ``sigma_n^2 = a^2 s_n^2 + b^2``. ``s_n`` is the trace's own pre-stimulus SD
(an allowed input, §2.4); ``a``, ``b`` and the lag-one correlation ``phi`` are fitted on training residuals only.

AR(1) is Markov, so the likelihood of the observed samples of a masked trace is exact without imputation:
the first observed sample is ``N(0, sigma^2)`` and each later one, ``d`` volumes after the previous observed
sample, is ``N(phi^d e_prev, sigma^2 (1 - phi^(2d)))``. This equals the multivariate-normal marginal of the
observed subset, which the unit tests check.

``block_nll`` is the block-aggregated alternative: block means of observed residuals scored with their exact
variance under the fitted process, blocks treated as independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class NoiseModel:
    a: float
    b: float
    phi: float

    def sigma(self, s: FloatArray) -> FloatArray:
        s = np.nan_to_num(np.asarray(s, dtype=np.float64), nan=0.0)
        return np.asarray(np.sqrt(self.a**2 * s**2 + self.b**2))

    def to_json(self) -> dict[str, float | str]:
        return {"a": self.a, "b": self.b, "phi": self.phi, "family": "gaussian_ar1"}


def _previous_valid(valid: BoolArray) -> npt.NDArray[np.int64]:
    """Index of the previous valid sample in the same row, or -1."""
    n, t = valid.shape
    idx = np.where(valid, np.arange(t)[None, :], -1)
    run = np.maximum.accumulate(idx, axis=1)
    prev = np.full((n, t), -1, dtype=np.int64)
    prev[:, 1:] = run[:, :-1]
    return prev


def ar1_nll(resid: FloatArray, valid: BoolArray, sigma: FloatArray, phi: float) -> FloatArray:
    """Exact negative log likelihood of each trace's observed residuals; returns one value per trace."""
    e = np.where(valid, resid, 0.0).astype(np.float64)
    prev = _previous_valid(valid)
    has_prev = valid & (prev >= 0)
    first = valid & (prev < 0)
    rows = np.arange(e.shape[0])[:, None]
    e_prev = e[rows, np.maximum(prev, 0)]
    gap = np.where(has_prev, np.arange(e.shape[1])[None, :] - prev, 1)
    rho = np.power(phi, gap)
    var_c = np.maximum(1.0 - rho**2, 1e-12)
    s2 = (np.asarray(sigma, dtype=np.float64) ** 2)[:, None]
    term_first = 0.5 * (LOG_2PI + np.log(s2) + e**2 / s2)
    term_cond = 0.5 * (LOG_2PI + np.log(s2 * var_c) + (e - rho * e_prev) ** 2 / (s2 * var_c))
    out = np.where(first, term_first, 0.0) + np.where(has_prev, term_cond, 0.0)
    return np.asarray(out.sum(axis=1))


def block_nll(resid: FloatArray, valid: BoolArray, sigma: FloatArray, phi: float, block: int) -> FloatArray:
    """Block-aggregated Gaussian score per trace (proper for block means; ignores cross-block correlation)."""
    n, t = resid.shape
    total = np.zeros(n)
    lag = np.abs(np.arange(block)[:, None] - np.arange(block)[None, :])
    corr = np.power(phi, lag)
    s2 = np.asarray(sigma, dtype=np.float64) ** 2
    for start in range(0, t, block):
        v = valid[:, start : start + block]
        m = v.sum(axis=1)
        ok = m > 0
        if not ok.any():
            continue
        w = v.astype(np.float64)
        k = w.shape[1]
        mean = np.where(ok, (np.where(v, resid[:, start : start + block], 0.0)).sum(axis=1) / np.maximum(m, 1), 0.0)
        quad = np.einsum("na,ab,nb->n", w, corr[:k, :k], w)
        var = s2 * quad / np.maximum(m, 1) ** 2
        total += np.where(ok, 0.5 * (LOG_2PI + np.log(np.where(ok, var, 1.0)) + mean**2 / np.where(ok, var, 1.0)), 0.0)
    return total


def fit_noise(resid: FloatArray, valid: BoolArray, s: FloatArray, max_traces: int = 50_000) -> NoiseModel:
    """Maximum-likelihood ``(a, b, phi)`` on training residuals. Uses a deterministic subsample when large."""
    n = resid.shape[0]
    if n > max_traces:
        keep = np.linspace(0, n - 1, max_traces).round().astype(np.int64)
        resid, valid, s = resid[keep], valid[keep], s[keep]
    s = np.nan_to_num(np.asarray(s, dtype=np.float64), nan=0.0)
    ev = np.where(valid, resid, np.nan)
    scale = float(np.sqrt(np.nanmean(ev**2))) if valid.any() else 1.0

    def unpack(x: FloatArray) -> NoiseModel:
        return NoiseModel(a=float(np.exp(x[0])), b=float(np.exp(x[1]) * scale), phi=float(np.tanh(x[2])))

    def objective(x: FloatArray) -> float:
        m = unpack(x)
        return float(ar1_nll(resid, valid, m.sigma(s), m.phi).sum() / max(valid.sum(), 1))

    best = None
    for x0 in ([0.0, np.log(0.5), 0.5], [np.log(0.5), np.log(0.1), 1.0]):
        res = optimize.minimize(objective, np.asarray(x0, dtype=np.float64), method="L-BFGS-B",
                                options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    return unpack(best.x)


def interval_coverage(
    resid: FloatArray, valid: BoolArray, sigma: FloatArray, levels: tuple[float, ...] = (0.5, 0.8, 0.95)
) -> dict[str, float]:
    """Fraction of observed samples inside central marginal predictive intervals."""
    z = np.abs(resid) / np.asarray(sigma)[:, None]
    zv = z[valid]
    return {f"{lv:.2f}": float(np.mean(zv <= stats.norm.ppf(0.5 + lv / 2))) for lv in levels}


def per_animal(values: FloatArray, animals: npt.NDArray[np.object_]) -> dict[str, float]:
    """Sum of per-trace scores by animal."""
    out: dict[str, float] = {}
    for a, v in zip(animals, values, strict=True):
        out[str(a)] = out.get(str(a), 0.0) + float(v)
    return out
