"""Masked proper scoring under temporally correlated noise (§11.2, OW-005).

Residuals ``e = y - mu`` of one trace are modelled as a stationary Gaussian AR(1) process sampled on the volume
grid, with per-trace marginal variance ``sigma_n^2 = a^2 s_n^2 + b^2``. ``s_n`` is the trace's own pre-stimulus SD
(an allowed input, §2.4); ``a``, ``b`` and the lag-one correlation ``phi`` are fitted on training residuals only.

AR(1) is Markov, so the likelihood of the observed samples of a masked trace is exact without imputation:
the first observed sample is ``N(0, sigma^2)`` and each later one, ``d`` volumes after the previous observed
sample, is ``N(phi^d e_prev, sigma^2 (1 - phi^(2d)))``. This equals the multivariate-normal marginal of the
observed subset, which the unit tests check.

Optional heavy-tailed variant (``nu`` set, family ``student_t_ar1``, open decision 3): the same recursion with
Student-t innovations. The standardized innovation ``z = (e - rho e_prev) / (sigma sqrt(1 - rho^2))`` (``z = e / sigma``
for the first observed sample) is i.i.d. t with ``nu`` degrees of freedom, so ``sigma`` is a scale, not the SD
(SD = sigma sqrt(nu / (nu - 2)) for nu > 2). The process is a conditional-t AR(1), Markov like the Gaussian one, so
masking stays exact; as ``nu`` -> infinity it is the Gaussian model. Gaussian remains the default everywhere.

``block_nll`` is the block-aggregated alternative: block means of observed residuals scored with their exact
variance under the fitted process, blocks treated as independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize, special, stats

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class NoiseModel:
    a: float
    b: float
    phi: float
    nu: float | None = None  # None: Gaussian innovations; else Student-t degrees of freedom

    def sigma(self, s: FloatArray) -> FloatArray:
        s = np.nan_to_num(np.asarray(s, dtype=np.float64), nan=0.0)
        return np.asarray(np.sqrt(self.a**2 * s**2 + self.b**2))

    def to_json(self) -> dict[str, float | str]:
        if self.nu is None:
            return {"a": self.a, "b": self.b, "phi": self.phi, "family": "gaussian_ar1"}
        return {"a": self.a, "b": self.b, "phi": self.phi, "nu": self.nu, "family": "student_t_ar1"}


def _previous_valid(valid: BoolArray) -> npt.NDArray[np.int64]:
    """Index of the previous valid sample in the same row, or -1."""
    n, t = valid.shape
    idx = np.where(valid, np.arange(t)[None, :], -1)
    run = np.maximum.accumulate(idx, axis=1)
    prev = np.full((n, t), -1, dtype=np.int64)
    prev[:, 1:] = run[:, :-1]
    return prev


def _log_t_norm(nu: float) -> float:
    """Constant of the Student-t NLL: -lgamma((nu+1)/2) + lgamma(nu/2) + log(nu pi)/2."""
    return float(-special.gammaln(0.5 * (nu + 1.0)) + special.gammaln(0.5 * nu) + 0.5 * np.log(nu * np.pi))


def ar1_nll(resid: FloatArray, valid: BoolArray, sigma: FloatArray, phi: float, nu: float | None = None) -> FloatArray:
    """Exact negative log likelihood of each trace's observed residuals; returns one value per trace.

    Gaussian innovations by default; Student-t with ``nu`` degrees of freedom (scale ``sigma``) if ``nu`` is given.
    """
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
    if nu is None:
        term_first = 0.5 * (LOG_2PI + np.log(s2) + e**2 / s2)
        term_cond = 0.5 * (LOG_2PI + np.log(s2 * var_c) + (e - rho * e_prev) ** 2 / (s2 * var_c))
    else:
        c = _log_t_norm(nu)
        term_first = c + 0.5 * np.log(s2) + 0.5 * (nu + 1.0) * np.log1p(e**2 / (s2 * nu))
        term_cond = (
            c + 0.5 * np.log(s2 * var_c) + 0.5 * (nu + 1.0) * np.log1p((e - rho * e_prev) ** 2 / (s2 * var_c * nu))
        )
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


Flat = tuple[FloatArray, FloatArray, FloatArray, BoolArray, npt.NDArray[np.int64]]


def _flatten(resid: FloatArray, valid: BoolArray) -> Flat:
    """Observed samples as flat arrays: value, previous observed value, gap, is-first flag, trace index."""
    prev = _previous_valid(valid)
    rows, cols = np.nonzero(valid)
    p = prev[rows, cols]
    first = p < 0
    e = resid[rows, cols].astype(np.float64)
    e_prev = np.where(first, 0.0, resid[rows, np.maximum(p, 0)]).astype(np.float64)
    gap = np.where(first, 1, cols - p).astype(np.float64)
    return e, e_prev, gap, first, rows


def noise_objective(x: FloatArray, flat: Flat, s: FloatArray, scale: float) -> tuple[float, FloatArray]:
    """Mean NLL per observed sample and its gradient in ``x = (log a, log(b/scale), atanh phi)``."""
    e, e_prev, gap, first, rows = flat
    a, b, phi = float(np.exp(x[0])), float(np.exp(x[1]) * scale), float(np.tanh(x[2]))
    sn2 = s[rows] ** 2
    s2 = a**2 * sn2 + b**2
    rho = np.where(first, 0.0, np.power(phi, gap))
    one_m = np.maximum(1.0 - rho**2, 1e-12)
    v = s2 * one_m
    r = e - rho * e_prev
    n = max(e.size, 1)
    f = 0.5 * (LOG_2PI + np.log(v) + r**2 / v)
    df_dv = 0.5 * (1.0 / v - r**2 / v**2)
    df_ds2 = df_dv * one_m
    df_drho = df_dv * (-2.0 * s2 * rho) + (r / v) * (-e_prev)
    drho_dphi = np.where(first, 0.0, gap * np.power(phi, np.maximum(gap - 1.0, 0.0)))
    g0 = float(np.sum(df_ds2 * 2.0 * a**2 * sn2))
    g1 = float(np.sum(df_ds2 * 2.0 * b**2))
    g2 = float(np.sum(df_drho * drho_dphi)) * (1.0 - phi**2)
    return float(f.sum()) / n, np.array([g0, g1, g2]) / n


def noise_objective_t(x: FloatArray, flat: Flat, s: FloatArray, scale: float) -> tuple[float, FloatArray]:
    """Student-t counterpart of ``noise_objective``; ``x = (log a, log(b/scale), atanh phi, log nu)``."""
    e, e_prev, gap, first, rows = flat
    a, b, phi, nu = float(np.exp(x[0])), float(np.exp(x[1]) * scale), float(np.tanh(x[2])), float(np.exp(x[3]))
    sn2 = s[rows] ** 2
    s2 = a**2 * sn2 + b**2
    rho = np.where(first, 0.0, np.power(phi, gap))
    one_m = np.maximum(1.0 - rho**2, 1e-12)
    v = s2 * one_m
    r = e - rho * e_prev
    n = max(e.size, 1)
    q = r**2 / (v * nu)
    u = 1.0 + q
    f = _log_t_norm(nu) + 0.5 * np.log(v) + 0.5 * (nu + 1.0) * np.log(u)
    df_dv = 0.5 / v * (1.0 - (nu + 1.0) * q / u)
    df_ds2 = df_dv * one_m
    df_dr = (nu + 1.0) * r / (v * nu * u)
    df_drho = df_dv * (-2.0 * s2 * rho) + df_dr * (-e_prev)
    drho_dphi = np.where(first, 0.0, gap * np.power(phi, np.maximum(gap - 1.0, 0.0)))
    df_dnu = (
        -0.5 * special.digamma(0.5 * (nu + 1.0))
        + 0.5 * special.digamma(0.5 * nu)
        + 0.5 / nu
        + 0.5 * np.log(u)
        - 0.5 * (nu + 1.0) * q / (nu * u)
    )
    g0 = float(np.sum(df_ds2 * 2.0 * a**2 * sn2))
    g1 = float(np.sum(df_ds2 * 2.0 * b**2))
    g2 = float(np.sum(df_drho * drho_dphi)) * (1.0 - phi**2)
    g3 = float(np.sum(df_dnu)) * nu
    return float(f.sum()) / n, np.array([g0, g1, g2, g3]) / n


def fit_noise(
    resid: FloatArray,
    valid: BoolArray,
    s: FloatArray,
    max_traces: int = 50_000,
    family: str = "gaussian_ar1",
) -> NoiseModel:
    """Maximum-likelihood noise model on training residuals. Uses a deterministic subsample when large.

    ``family`` is ``gaussian_ar1`` (``a, b, phi``) or ``student_t_ar1`` (adds ``nu``, bounded to [1, 1000]).
    """
    if family not in ("gaussian_ar1", "student_t_ar1"):
        raise ValueError(f"unknown noise family {family!r}")
    heavy = family == "student_t_ar1"
    n = resid.shape[0]
    if n > max_traces:
        keep = np.linspace(0, n - 1, max_traces).round().astype(np.int64)
        resid, valid, s = resid[keep], valid[keep], s[keep]
    s = np.nan_to_num(np.asarray(s, dtype=np.float64), nan=0.0)
    ev = np.where(valid, resid, np.nan)
    scale = float(np.sqrt(np.nanmean(ev**2))) if valid.any() else 1.0
    flat = _flatten(resid, valid)
    best = None
    starts = [[0.0, np.log(0.5), 0.5], [np.log(0.5), np.log(0.1), 1.0]]
    bounds = [(-12.0, 6.0), (-20.0, 6.0), (-4.0, 4.0)]
    if heavy:
        starts = [x0 + [float(np.log(nu0))] for x0, nu0 in zip(starts, (5.0, 30.0), strict=True)]
        bounds.append((0.0, float(np.log(1000.0))))
    for x0 in starts:
        res = optimize.minimize(
            noise_objective_t if heavy else noise_objective,
            np.asarray(x0, dtype=np.float64),
            args=(flat, s, scale),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 500},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    x = best.x
    nu = float(np.exp(x[3])) if heavy else None
    return NoiseModel(a=float(np.exp(x[0])), b=float(np.exp(x[1]) * scale), phi=float(np.tanh(x[2])), nu=nu)


def interval_coverage(
    resid: FloatArray,
    valid: BoolArray,
    sigma: FloatArray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
    nu: float | None = None,
) -> dict[str, float]:
    """Fraction of observed samples inside central marginal predictive intervals (Gaussian, or t with ``nu``)."""
    z = np.abs(resid) / np.asarray(sigma)[:, None]
    zv = z[valid]

    def q(p: float) -> float:
        return float(stats.norm.ppf(p) if nu is None else stats.t.ppf(p, nu))

    return {f"{lv:.2f}": float(np.mean(zv <= q(0.5 + lv / 2))) for lv in levels}


def per_animal(values: FloatArray, animals: npt.NDArray[np.object_]) -> dict[str, float]:
    """Sum of per-trace scores by animal."""
    out: dict[str, float] = {}
    for a, v in zip(animals, values, strict=True):
        out[str(a)] = out.get(str(a), 0.0) + float(v)
    return out
