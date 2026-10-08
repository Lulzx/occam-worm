"""Statistical uncertainty of fitted parameters: inverse Hessian and parametric bootstrap.

``hessian_standard_errors`` inverts the Hessian of the NLL in the natural parameters at the optimum (observed
Fisher information, valid for an interior optimum of a correctly specified likelihood). Components within a
relative ``bound_tol`` of a bound, a non-positive-definite Hessian or a non-finite inverse are flagged and their
standard error is ``nan``. ``parametric_bootstrap`` simulates replicate datasets from the fitted model plus
noise drawn from the loss's own noise model, refits each from the fitted values and reports the spread of the
estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from occamworm.fit.losses import GaussianLoss, ObservationLoss
from occamworm.fit.multistart import FitBudget, FitResult, StartResult, fit_local
from occamworm.fit.problem import FitProblem

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HessianSE:
    names: tuple[str, ...]
    se: FloatArray
    covariance: FloatArray
    hessian: FloatArray
    positive_definite: bool
    at_bound: tuple[bool, ...]
    condition_number: float

    def to_dict(self) -> dict[str, float]:
        return {n: float(s) for n, s in zip(self.names, self.se, strict=True)}


def hessian_standard_errors(
    problem: FitProblem, theta_free: FloatArray, target: FloatArray | None = None, bound_tol: float = 1e-4
) -> HessianSE:
    theta = np.asarray(theta_free, dtype=np.float64)
    h = problem.hessian_theta(theta, target)
    h = 0.5 * (h + h.T)
    lo, hi = problem.transforms.lower(), problem.transforms.upper()
    width = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
    at_bound = tuple(bool(b) for b in (theta - lo < bound_tol * width) | (hi - theta < bound_tol * width))
    eig = np.linalg.eigvalsh(h)
    pd = bool(np.min(eig) > 0.0)
    k = len(theta)
    if pd:
        cov = np.linalg.inv(h)
        cov = 0.5 * (cov + cov.T)
        se = np.sqrt(np.diag(cov))
        cond = float(np.max(eig) / np.min(eig))
    else:
        cov = np.full((k, k), np.nan)
        se = np.full(k, np.nan)
        cond = float("inf")
    return HessianSE(problem.free_names, se, cov, h, pd, at_bound, cond)


@dataclass
class BootstrapResult:
    names: tuple[str, ...]
    estimates: FloatArray  # (n_rep, k)
    se: FloatArray
    bias: FloatArray
    fits: list[StartResult] = field(repr=False)
    n_fun_grad_evals: int = 0
    wall_seconds: float = 0.0


def parametric_bootstrap(
    problem: FitProblem, fit: FitResult, n_rep: int, seed: int, budget: FitBudget | None = None
) -> BootstrapResult:
    """Refit ``n_rep`` replicate datasets simulated from the fitted model (single start at the estimate)."""
    rng = np.random.default_rng(seed)
    theta_hat = fit.theta_free
    mean = np.asarray(problem.predict(theta_hat), dtype=np.float64)
    budget = budget if budget is not None else fit.budget
    noise_model: ObservationLoss = problem.loss
    if isinstance(noise_model, GaussianLoss) and noise_model.sigma is None:  # profiled noise: use the residual SD
        resid = (problem.target - mean)[problem.mask]
        noise_model = GaussianLoss(float(np.sqrt(np.mean(resid**2))))
    fits: list[StartResult] = []
    for _ in range(n_rep):
        noise = noise_model.sample_noise(rng, problem.mask)
        replicate = problem.with_target(mean + noise)
        fits.append(fit_local(replicate, theta_hat, budget))
    est = np.asarray([f.theta for f in fits])
    return BootstrapResult(
        names=problem.free_names,
        estimates=est,
        se=est.std(axis=0, ddof=1) if n_rep > 1 else np.full(est.shape[1], np.nan),
        bias=est.mean(axis=0) - theta_hat,
        fits=fits,
        n_fun_grad_evals=sum(f.n_fun_grad_evals for f in fits),
        wall_seconds=sum(f.wall_seconds for f in fits),
    )
