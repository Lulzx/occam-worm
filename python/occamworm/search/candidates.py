"""Candidates for the nested search (OW-012).

A candidate is scored inside a ``TrainingView`` only: ``inner_select`` fits on inner-training animals, scores the
inner-validation animals and picks its ridge; ``fit_final`` refits on the whole view and returns the frozen state;
``score`` is called by the ``LockedEvaluator`` alone, with the frozen state and the held-out data. Fitted pair
kernels are keyed by (target, responder) labels, so held-out pairs never seen in training get a zero kernel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from occamworm.analysis.dataset import PRE
from occamworm.analysis.scoring import NoiseModel, ar1_nll
from occamworm.baselines.data import M, TaskData, aggregate, compute_stats, predict, trial_designs
from occamworm.baselines.evaluate import noise_for, rows_of, score_rows
from occamworm.baselines.indicator import Indicator, estimate_from_autoresponses
from occamworm.baselines.kernels import Fit, fit_family
from occamworm.search.views import TrainingView

FloatArray = npt.NDArray[np.float64]
BITS_PER_FLOAT = 32  # declared precision of fitted kernel coefficients for L_params (§5.7)


class CandidateRejected(ValueError):
    """A candidate refused before fitting (e.g. a WRL program failing a structural or stability gate)."""


@dataclass
class InnerResult:
    key: str
    ridge: float
    inner_nll: float
    val_animal_nll: dict[str, float]
    l_struct_bits: float
    l_params_bits: float
    n_params: int
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def l_total_bits(self) -> float:
        return self.l_struct_bits + self.l_params_bits


class Candidate(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def l_struct_bits(self) -> float: ...

    def inner_select(self, view: TrainingView) -> InnerResult: ...

    def fit_final(self, view: TrainingView, ridge: float) -> dict[str, Any]: ...

    def score(self, fitted: dict[str, Any], test: TaskData) -> dict[str, float]: ...


def _indicator(data: TaskData) -> Indicator | None:
    if data.task != "T2b":
        return None
    ok = data.trial_auto_ok
    tr = data.trial_auto[ok, PRE:]
    est = estimate_from_autoresponses(
        tr, np.ones_like(tr, dtype=bool), [data.animals[a] for a in data.trial_animal[ok]], data.dt, reps=0
    )
    ind: Indicator = est["indicator"]
    return ind


@dataclass
class KernelCandidate:
    """A baseline family (B0-B3) treated as a search candidate; L_struct is a fixed declared cost per family."""

    family: str
    l_struct_bits: float
    ridge_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    kind: str = "kernel"

    @property
    def key(self) -> str:
        return f"kernel:{self.family}"

    def _fit(self, data: TaskData, ps: Any, lam: float, warm: Fit | None) -> Fit:
        return fit_family(self.family, ps, data.pair_target, len(data.targets), lam, init=warm)

    def inner_select(self, view: TrainingView) -> InnerResult:
        t0 = time.time()
        data = view.data
        designs = trial_designs(data, _indicator(data))
        stats = compute_stats(data, designs)
        grid = (0.0,) if self.family == "B0" else self.ridge_grid
        totals = dict.fromkeys(grid, 0.0)
        per_animal: dict[float, dict[str, float]] = {lam: {} for lam in grid}
        n_params = 0
        for itrain, ival in view.inner:
            ps = aggregate(stats, itrain, len(data.pairs))
            fits: dict[float, Fit] = {}
            warm: Fit | None = None
            for lam in sorted(grid):
                warm = fits[lam] = self._fit(data, ps, lam, warm)
            noise = noise_for(data, designs, fits[grid[len(grid) // 2]], rows_of(data, itrain))
            vrows = rows_of(data, ival)
            animals = np.asarray(data.animals, dtype=object)[data.trace_animal[vrows]]
            for lam, f in fits.items():
                nll, _, _ = score_rows(data, designs, f, noise, vrows)
                totals[lam] += float(nll.sum())
                for a, x in zip(animals, nll, strict=True):
                    per_animal[lam][a] = per_animal[lam].get(a, 0.0) + float(x)
                n_params = f.n_params
        best = min(grid, key=lambda lam: (totals[lam], lam))
        return InnerResult(
            self.key,
            best,
            totals[best],
            per_animal[best],
            self.l_struct_bits,
            float(n_params * BITS_PER_FLOAT),
            n_params,
            time.time() - t0,
        )

    def fit_final(self, view: TrainingView, ridge: float) -> dict[str, Any]:
        data = view.data
        ind = _indicator(data)
        designs = trial_designs(data, ind)
        ps = aggregate(compute_stats(data, designs), np.ones(len(data.animals), dtype=bool), len(data.pairs))
        fit = self._fit(data, ps, ridge, None)
        noise = noise_for(data, designs, fit, np.arange(data.n_traces))
        live = ps.present
        return {
            "family": self.family,
            "task": data.task,
            "history": data.history,
            "ridge": ridge,
            "kernels": {f"{t}|{r}": fit.c[p].tolist() for p, (t, r) in enumerate(data.pairs) if live[p]},
            "beta": fit.beta.tolist(),
            "noise": noise.to_json(),
            "indicator": ind.to_json() if ind else None,
            "n_params": fit.n_params,
        }

    def score(self, fitted: dict[str, Any], test: TaskData) -> dict[str, float]:
        return score_kernels(fitted, test)


def score_kernels(fitted: dict[str, Any], test: TaskData) -> dict[str, float]:
    """Per-animal NLL of frozen pair kernels (keyed ``target|responder``), baseline and noise on held-out data."""
    ind = None
    if fitted["indicator"]:
        ind = Indicator(fitted["indicator"]["tau_r_s"], fitted["indicator"]["tau_d_s"])
    designs = trial_designs(test, ind)
    c = np.zeros((len(test.pairs), M))
    for p, (t, r) in enumerate(test.pairs):
        k = fitted["kernels"].get(f"{t}|{r}")
        if k is not None:
            c[p] = k
    beta = np.asarray(fitted["beta"], dtype=np.float64)
    rows = np.arange(test.n_traces)
    mu = predict(test, designs, c, beta, rows)
    n = fitted["noise"]
    noise = NoiseModel(float(n["a"]), float(n["b"]), float(n["phi"]))
    nll = ar1_nll(test.y[rows].astype(np.float64) - mu, test.m[rows], noise.sigma(test.s[rows]), noise.phi)
    out: dict[str, float] = {}
    for a, x in zip(np.asarray(test.animals, dtype=object)[test.trace_animal[rows]], nll, strict=True):
        out[a] = out.get(a, 0.0) + float(x)
    return out
