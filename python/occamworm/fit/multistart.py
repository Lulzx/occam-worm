"""Multi-start L-BFGS-B fitting on transformed parameters, with the optimisation budget recorded.

Start points are deterministic given the seed: start 0 is the declared (midpoint) parameter vector, the others are
a Latin-hypercube sample of the bounded box (``scipy.stats.qmc``, seeded), kept away from the bounds. Every start
runs L-BFGS-B on the unconstrained ``raw`` vector with the exact JAX gradient. The budget (function-and-gradient
evaluations, iterations, simulated steps, wall time) is counted per start and in total. A wall-clock cap stops
launching new starts once exceeded (a started L-BFGS-B run is bounded by ``maxiter`` and ``maxfun``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy import optimize
from scipy.stats import qmc

from occamworm.fit.problem import FitProblem

FloatArray = npt.NDArray[np.float64]
EDGE = 0.02  # random starts avoid the outer 2% of every bounded interval


@dataclass(frozen=True)
class FitBudget:
    """Per-start limits and the global wall-clock cap (``None``: unlimited)."""

    maxiter: int = 200
    maxfun: int = 1000
    ftol: float = 1e-14
    gtol: float = 1e-9
    max_wall_seconds: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "maxiter_per_start": self.maxiter,
            "maxfun_per_start": self.maxfun,
            "ftol": self.ftol,
            "gtol": self.gtol,
            "max_wall_seconds": self.max_wall_seconds,
        }


@dataclass
class StartResult:
    index: int
    theta0: FloatArray
    theta: FloatArray  # fitted parameters, natural units
    nll: float
    n_iterations: int
    n_fun_grad_evals: int
    wall_seconds: float
    success: bool
    message: str
    grad_norm: float


@dataclass
class FitResult:
    problem: FitProblem
    seed: int
    budget: FitBudget
    starts: list[StartResult]
    best: int
    wall_seconds: float
    n_starts_requested: int
    stopped_by_wall_cap: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def best_start(self) -> StartResult:
        return self.starts[self.best]

    @property
    def theta_free(self) -> FloatArray:
        return self.best_start.theta

    @property
    def theta_full(self) -> FloatArray:
        return np.asarray(self.problem.theta_full(self.best_start.theta), dtype=np.float64)

    @property
    def estimates(self) -> dict[str, float]:
        return {p.source_name: float(v) for p, v in zip(self.problem.program.parameters, self.theta_full, strict=True)}

    @property
    def n_fun_grad_evals(self) -> int:
        return sum(s.n_fun_grad_evals for s in self.starts)

    @property
    def n_iterations(self) -> int:
        return sum(s.n_iterations for s in self.starts)

    @property
    def n_simulated_steps(self) -> int:
        """Trial-steps simulated: each evaluation runs every trial for ``n_steps`` ticks (plus its backward pass)."""
        return int(self.n_fun_grad_evals * self.problem.n_trials * self.problem.simulator.n_steps)


def start_points(problem: FitProblem, n_starts: int, seed: int) -> FloatArray:
    """``(n_starts, k)`` starting values (natural units). Row 0 is the declared value; deterministic in ``seed``."""
    tr = problem.transforms
    lo, hi = tr.lower(), tr.upper()
    span = np.asarray([t.span() for t in tr.transforms])
    declared = np.clip(problem.theta_free(problem.theta_base), lo + 1e-9 * span, hi)
    rows = [declared]
    if n_starts > 1:
        sampler = qmc.LatinHypercube(d=len(tr), seed=seed)
        u = sampler.random(n_starts - 1)
        rows.extend(lo + (EDGE + (1.0 - 2.0 * EDGE) * u[i]) * span for i in range(n_starts - 1))
    return np.asarray(rows, dtype=np.float64)


def fit_local(problem: FitProblem, theta0: FloatArray, budget: FitBudget, index: int = 0) -> StartResult:
    """One L-BFGS-B run from ``theta0`` (natural units)."""
    tr = problem.transforms
    raw0 = tr.inverse(theta0)
    evals = 0
    # L-BFGS-B takes a unit-Hessian first step: with a summed NLL (gradients of order 1e4) that step jumps to the
    # corner of the box, where a saturated transform has no gradient. The per-observation NLL is O(1).
    scale = float(max(int(problem.mask.sum()), 1))

    def objective(raw: FloatArray) -> tuple[float, FloatArray]:
        nonlocal evals
        evals += 1
        value, grad = problem.value_and_grad_raw(raw)
        return value / scale, grad / scale

    t0 = time.perf_counter()
    res = optimize.minimize(
        objective,
        raw0,
        jac=True,
        method="L-BFGS-B",
        bounds=tr.raw_bounds(),
        options={"maxiter": budget.maxiter, "maxfun": budget.maxfun, "ftol": budget.ftol, "gtol": budget.gtol},
    )
    wall = time.perf_counter() - t0
    theta = np.asarray(tr.forward(jnp.asarray(res.x)), dtype=np.float64)
    return StartResult(
        index=index,
        theta0=np.asarray(theta0, dtype=np.float64),
        theta=theta,
        nll=float(res.fun) * scale,
        n_iterations=int(res.nit),
        n_fun_grad_evals=evals,
        wall_seconds=wall,
        success=bool(res.success),
        message=str(res.message),
        grad_norm=float(np.max(np.abs(res.jac))) * scale,
    )


def fit_multistart(problem: FitProblem, n_starts: int = 8, seed: int = 0, budget: FitBudget | None = None) -> FitResult:
    """Multi-start fit; the best start (lowest NLL, ties to the earlier start) gives the estimate."""
    budget = budget if budget is not None else FitBudget()
    starts = start_points(problem, n_starts, seed)
    t0 = time.perf_counter()
    done: list[StartResult] = []
    capped = False
    for i, theta0 in enumerate(starts):
        if budget.max_wall_seconds is not None and done and time.perf_counter() - t0 > budget.max_wall_seconds:
            capped = True
            break
        done.append(fit_local(problem, theta0, budget, i))
    best = min(range(len(done)), key=lambda i: (done[i].nll, i))
    return FitResult(
        problem=problem,
        seed=seed,
        budget=budget,
        starts=done,
        best=best,
        wall_seconds=time.perf_counter() - t0,
        n_starts_requested=n_starts,
        stopped_by_wall_cap=capped,
    )
