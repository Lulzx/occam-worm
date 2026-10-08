"""Fit problem: a program, a graph, stimuli, masked target traces and a loss, as a differentiable objective.

The objective is a function of the unconstrained ``raw`` vector of the fitted parameters,
``raw -> theta = transforms.forward(raw) -> JAX simulation -> loss``. Parameters that are not fitted keep the
values of ``theta_base`` (default: the declared values of the IR). The target is an argument of the compiled
functions, so replicate datasets (parametric bootstrap, repeated recovery tests) reuse one compilation.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from occamworm.fit.losses import BoundLoss, ObservationLoss
from occamworm.fit.transforms import Kind, ParamTransforms
from occamworm.sim.ir import Program
from occamworm.sim.jaxsim import JaxSimulator

FloatArray = npt.NDArray[np.float64]


class FitProblem:
    """Static data of one fit. ``stimulus`` is ``(T, N)`` or ``(B, T, N)``; ``target`` and ``mask`` ``(B, S, K)``."""

    def __init__(
        self,
        program: Program,
        simulator: JaxSimulator,
        stimulus: npt.ArrayLike,
        target: npt.ArrayLike,
        loss: ObservationLoss,
        mask: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        free: Iterable[str] | None = None,
        theta_base: Sequence[float] | None = None,
        kind: Kind | None = None,
    ) -> None:
        self.program = program
        self.simulator = simulator
        self.loss = loss
        self.stimulus = np.asarray(stimulus, dtype=np.float64)
        self.init = None if init is None else np.asarray(init, dtype=np.float64)
        self.batched = self.stimulus.ndim == 3 or (self.init is not None and self.init.ndim == 3)
        tgt = np.asarray(target, dtype=np.float64)
        self.target = tgt if tgt.ndim == 3 else tgt[None]
        self.mask = (
            np.isfinite(self.target) if mask is None else np.asarray(mask, dtype=bool).reshape(self.target.shape)
        )
        self.target = np.where(self.mask, self.target, 0.0)
        s, k = len(simulator.sample_ticks), len(simulator.reported)
        if self.target.shape[1:] != (s, k):
            raise ValueError(f"target must have shape (B, {s}, {k}), got {self.target.shape}")
        self.transforms = ParamTransforms.from_program(program, free, kind)
        if len(self.transforms) == 0:
            raise ValueError("no trainable parameters to fit")
        self.theta_base = np.asarray(program.default_theta() if theta_base is None else theta_base, dtype=np.float64)
        self.free_index = np.asarray([program.parameter_index(n) for n in self.transforms.names], dtype=np.int64)
        self.n_trials = self.target.shape[0]
        self._bound: BoundLoss = loss.bind(self.mask)
        self._compiled: dict[str, Callable[..., Any]] = {}

    # -- parameters -----------------------------------------------------------------------------------------------

    @property
    def free_names(self) -> tuple[str, ...]:
        return self.transforms.names

    def theta_full(self, theta_free: jax.Array | FloatArray) -> jax.Array:
        """Full IR-order parameter vector with the fitted entries replaced."""
        return jnp.asarray(self.theta_base).at[jnp.asarray(self.free_index)].set(theta_free)

    def theta_free(self, theta_full: Sequence[float] | FloatArray) -> FloatArray:
        return np.asarray(theta_full, dtype=np.float64)[self.free_index]

    # -- objective ------------------------------------------------------------------------------------------------

    def predict(self, theta_free: jax.Array | FloatArray) -> jax.Array:
        """Simulated observation ``(B, S, K)`` for the fitted parameters."""
        obs = self.simulator.observe(self.theta_full(theta_free), self.stimulus, self.init)
        return obs if self.batched else obs[None]

    def nll_theta(self, theta_free: jax.Array, target: jax.Array) -> jax.Array:
        return self._bound(self.predict(theta_free), target)

    def nll_raw(self, raw: jax.Array, target: jax.Array) -> jax.Array:
        return self.nll_theta(self.transforms.forward(raw), target)

    def _jit(self, key: str, make: Callable[[], Callable[..., Any]]) -> Callable[..., Any]:
        if key not in self._compiled:
            self._compiled[key] = jax.jit(make())
        return self._compiled[key]

    def value_and_grad_raw(self, raw: FloatArray, target: FloatArray | None = None) -> tuple[float, FloatArray]:
        """NLL and its gradient with respect to ``raw`` (one forward and one backward simulation)."""
        fn = self._jit("vg_raw", lambda: jax.value_and_grad(self.nll_raw, argnums=0))
        value, grad = fn(jnp.asarray(raw), jnp.asarray(self.target if target is None else target))
        return float(value), np.asarray(grad, dtype=np.float64)

    def nll_at(self, theta_free: FloatArray, target: FloatArray | None = None) -> float:
        fn = self._jit("nll_theta", lambda: self.nll_theta)
        return float(fn(jnp.asarray(theta_free), jnp.asarray(self.target if target is None else target)))

    def hessian_theta(self, theta_free: FloatArray, target: FloatArray | None = None) -> FloatArray:
        """Hessian of the NLL with respect to the natural parameters (forward-over-reverse)."""
        fn = self._jit("hess_theta", lambda: jax.hessian(self.nll_theta, argnums=0))
        return np.asarray(fn(jnp.asarray(theta_free), jnp.asarray(self.target if target is None else target)))

    def with_target(self, target: npt.ArrayLike) -> FitProblem:
        """The same problem on another target (same shape and mask); shares the compiled functions."""
        other = copy.copy(self)
        tgt = np.asarray(target, dtype=np.float64)
        tgt = tgt if tgt.ndim == 3 else tgt[None]
        if tgt.shape != self.target.shape:
            raise ValueError("replicate target must have the shape of the original target")
        other.target = np.where(self.mask, tgt, 0.0)
        return other

    # -- checks ---------------------------------------------------------------------------------------------------

    def check_gradient(self, raw: FloatArray | None = None, h: float = 1e-6) -> tuple[float, FloatArray, FloatArray]:
        """Gradient of the objective against central finite differences.

        Returns ``(max relative error, analytic gradient, finite-difference gradient)`` with the error scaled by
        ``max(|g|, 1e-3 max|g|)`` per component, evaluated at ``raw`` (default: the declared parameter values).
        """
        r = self.transforms.inverse(self.theta_free(self.theta_base)) if raw is None else np.asarray(raw, np.float64)
        _, g = self.value_and_grad_raw(r)
        fd = np.zeros_like(r)
        for k in range(r.size):
            d = np.zeros_like(r)
            d[k] = h
            fd[k] = (self.value_and_grad_raw(r + d)[0] - self.value_and_grad_raw(r - d)[0]) / (2 * h)
        scale = np.maximum(np.abs(fd), 1e-3 * max(np.abs(fd).max(), 1e-300))
        return float(np.max(np.abs(g - fd) / scale)), g, fd
