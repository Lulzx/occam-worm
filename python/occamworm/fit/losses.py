"""Masked observation losses on sampled traces (negative log-likelihoods, in nats).

Predictions, targets and masks have shape ``(B, S, K)``: trials, sample ticks, reported neurons. ``mask`` is true
where the target was observed; masked entries never enter the loss. A loss is *bound* to a mask (static) and then
maps ``(pred, target)`` to a scalar, so one compiled objective serves many target datasets.

* :class:`GaussianLoss`: independent Gaussian noise. With a known ``sigma`` this is the full Gaussian NLL; with
  ``sigma=None`` the noise level is profiled out, ``NLL = n/2 (1 + log(2 pi RSS / n))``, so that its Hessian still
  gives asymptotic standard errors (and the fit is the least-squares fit).
* :class:`AR1Loss`: the exact masked Gaussian AR(1) likelihood of ``occamworm.analysis.scoring.ar1_nll``
  reimplemented in JAX. Residuals of each trace (trial, neuron) follow a stationary AR(1) process with marginal SD
  ``sigma`` (scalar, or array broadcastable to ``(B, K)``) and lag-one correlation ``phi`` on the sample grid;
  observed samples are scored with the exact conditional of the previous observed sample.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from occamworm.sim import jaxsim  # noqa: F401  (float64)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
LOG_2PI = math.log(2.0 * math.pi)
BoundLoss = Callable[[jax.Array, jax.Array], jax.Array]


class ObservationLoss(Protocol):
    @property
    def name(self) -> str: ...

    def bind(self, mask: BoolArray) -> BoundLoss:
        """The NLL as a function ``(pred, target) -> scalar`` for a fixed observation mask ``(B, S, K)``."""
        ...

    def sample_noise(self, rng: np.random.Generator, mask: BoolArray) -> FloatArray:
        """Noise ``(B, S, K)`` drawn from the loss's own noise model (used by parametric bootstrap and tests)."""
        ...

    def to_json(self) -> dict[str, Any]: ...


def _as_mask(mask: Any) -> BoolArray:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"mask must have shape (B, S, K), got {m.shape}")
    return m


@dataclass(frozen=True)
class GaussianLoss:
    """Gaussian observation noise; ``sigma=None`` profiles the noise SD out (least squares)."""

    sigma: float | None = None
    name: str = "gaussian"

    def bind(self, mask: BoolArray) -> BoundLoss:
        w = jnp.asarray(_as_mask(mask), dtype=jnp.float64)
        n = float(np.sum(mask))
        sigma = self.sigma

        def nll(pred: jax.Array, target: jax.Array) -> jax.Array:
            rss = jnp.sum(w * jnp.where(w > 0, target - pred, 0.0) ** 2)
            if sigma is None:
                return 0.5 * n * (1.0 + LOG_2PI + jnp.log(jnp.maximum(rss, 1e-300) / max(n, 1.0)))
            return 0.5 * rss / sigma**2 + n * (math.log(sigma) + 0.5 * LOG_2PI)

        return nll

    def sample_noise(self, rng: np.random.Generator, mask: BoolArray) -> FloatArray:
        if self.sigma is None:
            raise ValueError("sample_noise needs a known sigma")
        return np.asarray(rng.normal(0.0, self.sigma, size=_as_mask(mask).shape), dtype=np.float64)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "sigma": self.sigma, "family": "gaussian"}


@dataclass(frozen=True)
class AR1Structure:
    """Static index arrays of the masked AR(1) likelihood (flat over valid samples)."""

    rows: npt.NDArray[np.int64]  # trace index of each valid sample
    cols: npt.NDArray[np.int64]  # sample position
    prev: npt.NDArray[np.int64]  # sample position of the previous valid sample of the same trace (0 if none)
    first: BoolArray
    gap: FloatArray  # grid distance to the previous valid sample (1 for the first sample)
    n_traces: int


def ar1_structure(valid: BoolArray) -> AR1Structure:
    """Structure of a ``(traces, samples)`` mask."""
    n, t = valid.shape
    idx = np.where(valid, np.arange(t)[None, :], -1)
    run = np.maximum.accumulate(idx, axis=1)
    prev_all = np.full((n, t), -1, dtype=np.int64)
    prev_all[:, 1:] = run[:, :-1]
    rows, cols = np.nonzero(valid)
    p = prev_all[rows, cols]
    first = p < 0
    return AR1Structure(
        rows=rows.astype(np.int64),
        cols=cols.astype(np.int64),
        prev=np.where(first, 0, p).astype(np.int64),
        first=first,
        gap=np.where(first, 1.0, cols - p).astype(np.float64),
        n_traces=n,
    )


def _traces(a: Any) -> Any:
    """``(B, S, K)`` to one row per trace ``(B*K, S)``."""
    b, s, k = a.shape
    return a.transpose(0, 2, 1).reshape(b * k, s)


@dataclass(frozen=True)
class AR1Loss:
    """Masked Gaussian AR(1) NLL (the §11.2 scoring family) for fixed ``sigma`` and ``phi``."""

    sigma: float | FloatArray
    phi: float
    name: str = "ar1"

    def __post_init__(self) -> None:
        if not -1.0 < self.phi < 1.0:
            raise ValueError("phi must lie in (-1, 1)")

    def _sigma_per_trace(self, shape: tuple[int, int, int]) -> FloatArray:
        b, _, k = shape
        s = np.broadcast_to(np.asarray(self.sigma, dtype=np.float64), (b, k))
        return np.asarray(s.reshape(b * k), dtype=np.float64)

    def bind(self, mask: BoolArray) -> BoundLoss:
        m = _as_mask(mask)
        st = ar1_structure(_traces(m))
        s2 = jnp.asarray(self._sigma_per_trace(m.shape)[st.rows] ** 2)
        rho = np.where(st.first, 0.0, np.power(self.phi, st.gap))
        var_c = jnp.asarray(np.maximum(1.0 - rho**2, 1e-12))
        rho_j = jnp.asarray(rho)
        first = jnp.asarray(st.first)
        rows, cols, prev = jnp.asarray(st.rows), jnp.asarray(st.cols), jnp.asarray(st.prev)
        valid = jnp.asarray(_traces(m))

        def nll(pred: jax.Array, target: jax.Array) -> jax.Array:
            resid = jnp.where(valid, _traces(target - pred), 0.0)
            e = resid[rows, cols]
            e_prev = jnp.where(first, 0.0, resid[rows, prev])
            v = s2 * var_c
            r = e - rho_j * e_prev
            return jnp.sum(0.5 * (LOG_2PI + jnp.log(v) + r**2 / v))

        return nll

    def sample_noise(self, rng: np.random.Generator, mask: BoolArray) -> FloatArray:
        m = _as_mask(mask)
        b, s, k = m.shape
        sd = self._sigma_per_trace(m.shape)
        z = rng.standard_normal((b * k, s))
        e = np.empty_like(z)
        e[:, 0] = z[:, 0]
        for t in range(1, s):
            e[:, t] = self.phi * e[:, t - 1] + math.sqrt(1.0 - self.phi**2) * z[:, t]
        e *= sd[:, None]
        return np.asarray(e.reshape(b, k, s).transpose(0, 2, 1), dtype=np.float64)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "family": "gaussian_ar1", "sigma": np.asarray(self.sigma).tolist(), "phi": self.phi}
