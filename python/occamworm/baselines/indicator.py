"""Indicator stage, parameterized separately from neural kernels (§3.4, §4.5, OW-005).

The indicator impulse response is a difference of exponentials with peak 1:

    h(t) = (exp(-t / tau_d) - exp(-t / tau_r)) / peak,   0 < tau_r < tau_d,   t >= 0

with an optional saturating readout ``H(c) = c / (1 + c / c_sat)``.

Estimating it from autoresponses: the stimulated neuron's trace is the indicator filter applied to that neuron's
own (unknown) activity. Treating that activity as a pulse at the stimulation frame gives the slowest indicator
consistent with the data, i.e. an upper bound on indicator time constants; the report labels it so. No published
kinetic constant is used yet (audit: ``indicator.published_kinetics``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import optimize

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Indicator:
    tau_r: float  # s
    tau_d: float  # s
    c_sat: float | None = None  # saturation scale in dF/F; None = linear

    def kernel(self, n: int, dt: float) -> FloatArray:
        """Impulse response sampled at ``k * dt``, k = 0..n-1, peak-normalized (discrete convolution kernel)."""
        t = np.arange(n) * dt
        return difference_of_exponentials(t, self.tau_r, self.tau_d)

    def readout(self, c: FloatArray) -> FloatArray:
        if self.c_sat is None:
            return c
        return np.asarray(c / (1.0 + np.abs(c) / self.c_sat))

    def apply(self, x: FloatArray, dt: float) -> FloatArray:
        """Causal convolution of each row of ``x`` with the sampled kernel, then the readout."""
        x = np.atleast_2d(x)
        k = self.kernel(x.shape[-1], dt)
        out = np.stack([np.convolve(row, k)[: x.shape[-1]] for row in x])
        return self.readout(out)

    def to_json(self) -> dict[str, Any]:
        return {
            "family": "difference_of_exponentials",
            "tau_r_s": self.tau_r,
            "tau_d_s": self.tau_d,
            "c_sat": self.c_sat,
        }


def difference_of_exponentials(t: npt.ArrayLike, tau_r: float, tau_d: float) -> FloatArray:
    if not 0 < tau_r < tau_d:
        raise ValueError("need 0 < tau_r < tau_d")
    t = np.asarray(t, dtype=np.float64)
    raw = np.where(t >= 0, np.exp(-t / tau_d) - np.exp(-t / tau_r), 0.0)
    t_peak = tau_r * tau_d / (tau_d - tau_r) * np.log(tau_d / tau_r)
    peak = np.exp(-t_peak / tau_d) - np.exp(-t_peak / tau_r)
    return np.asarray(raw / peak)


def fit_to_mean_response(mean: FloatArray, dt: float) -> tuple[Indicator, float]:
    """Least-squares fit of ``A * h(t)`` to a mean impulse-like response sampled from t = 0. Returns (fit, A)."""
    t = np.arange(mean.size) * dt
    ok = np.isfinite(mean)

    def resid(x: FloatArray) -> FloatArray:
        tau_r = float(np.exp(x[0]))
        tau_d = tau_r + float(np.exp(x[1]))
        h = difference_of_exponentials(t[ok], tau_r, tau_d)
        amp = float(h @ mean[ok]) / max(float(h @ h), 1e-12)
        return np.asarray(mean[ok] - amp * h)

    best = None
    for r0, d0 in ((0.2, 2.0), (0.5, 5.0), (1.0, 10.0), (0.1, 1.0)):
        res = optimize.least_squares(resid, np.log([r0, d0]), method="lm")
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    tau_r = float(np.exp(best.x[0]))
    tau_d = tau_r + float(np.exp(best.x[1]))
    h = difference_of_exponentials(t[ok], tau_r, tau_d)
    return Indicator(tau_r, tau_d), float(h @ mean[ok] / (h @ h))


def estimate_from_autoresponses(
    traces: FloatArray,
    valid: npt.NDArray[np.bool_],
    animals: Sequence[str],
    dt: float,
    reps: int = 200,
    seed: int = 20261008,
) -> dict[str, Any]:
    """Fit the indicator to the amplitude-normalized mean autoresponse (post-stimulus samples, t >= 0).

    ``traces`` are training-fold autoresponses only. Each trace is normalized by its peak over the first 10 s so
    that strong trials do not dominate; uncertainty comes from an animal-level bootstrap.
    """
    y = np.where(valid, traces, np.nan)
    early = min(y.shape[1], int(round(10.0 / dt)))
    peak = np.nanmax(y[:, :early], axis=1)
    keep = np.isfinite(peak) & (peak > 0)
    yn = y[keep] / peak[keep, None]
    an = np.asarray(animals, dtype=object)[keep]
    uniq = sorted(set(an))
    by_animal = np.stack([np.nanmean(yn[an == a], axis=0) for a in uniq])
    fit, amp = fit_to_mean_response(np.nanmean(by_animal, axis=0), dt)
    rng = np.random.Generator(np.random.PCG64(seed))
    taus = []
    for _ in range(reps):
        pick = by_animal[rng.integers(0, len(uniq), len(uniq))]
        f, _ = fit_to_mean_response(np.nanmean(pick, axis=0), dt)
        taus.append((f.tau_r, f.tau_d))
    tr = np.array(taus).reshape(-1, 2)

    def ci(col: int) -> list[float] | None:
        if not reps:
            return None
        return [float(np.quantile(tr[:, col], 0.025)), float(np.quantile(tr[:, col], 0.975))]

    return {
        "indicator": fit,
        "amplitude": amp,
        "n_traces": int(keep.sum()),
        "n_animals": len(uniq),
        "tau_r_ci95": ci(0),
        "tau_d_ci95": ci(1),
        "interpretation": "upper bound: autoresponse = indicator * (stimulated neuron's own activity)",
    }


def resolvable_bandwidth(
    ind: Indicator, signal_amplitude: float, sigma: float, phi: float, dt: float
) -> dict[str, Any]:
    """Highest frequency at which a kernel component of the given amplitude stays above the noise (§4.5 step 2).

    SNR(f) = (A |H(f)| / |H(0)|)^2 / S(f), with H the indicator transfer function and S the AR(1) noise spectrum
    normalized to the per-sample variance sigma^2: S(f) = sigma^2 (1 - phi^2) / |1 - phi e^{-i 2 pi f dt}|^2.
    The resolvable bandwidth is the largest f <= Nyquist with SNR(f) >= 1; 1 / (2 pi f) is the matching timescale.
    """
    f = np.linspace(0.0, 0.5 / dt, 2001)
    w = 2 * np.pi * f
    gain = 1.0 / np.sqrt((1 + (w * ind.tau_r) ** 2) * (1 + (w * ind.tau_d) ** 2))
    noise = sigma**2 * (1 - phi**2) / np.abs(1 - phi * np.exp(-1j * w * dt)) ** 2
    snr = (signal_amplitude * gain) ** 2 / noise
    ok = np.nonzero(snr >= 1.0)[0]
    f_res = float(f[ok[-1]]) if ok.size else 0.0
    return {
        "f_resolvable_hz": f_res,
        "timescale_s": (1.0 / (2 * np.pi * f_res)) if f_res > 0 else None,
        "nyquist_hz": float(0.5 / dt),
        "signal_amplitude": signal_amplitude,
        "noise_sigma": sigma,
        "noise_phi": phi,
    }
