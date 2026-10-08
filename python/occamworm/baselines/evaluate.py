"""Nested animal-wise evaluation of kernel baselines (§7.4, §11, OW-006).

For one outer fold: the ridge strength of a family is chosen on the inner folds (validation NLL summed over inner
folds, noise model refitted on each inner training set), the family is refitted on the outer training animals,
the noise model is fitted on outer-training residuals, and the held-out animals are scored once. Held-out traces
enter only through ``score_rows``; fitting sees sufficient statistics of training animals only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from occamworm.analysis.scoring import NoiseModel, ar1_nll, fit_noise, interval_coverage
from occamworm.baselines.data import PairStats, Stats, TaskData, aggregate, predict
from occamworm.baselines.kernels import Fit, fit_family

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
LAM_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
NOISE_TRACES = 30_000
INNER_NOISE_TRACES = 10_000  # inner selection fits noise once per ridge value: ~500k samples for 3 parameters

Fitter = Callable[[PairStats, float, Fit | None], Fit]  # (stats, ridge, warm start) -> fit


@dataclass
class FoldResult:
    family: str
    lam_rel: float
    inner_scores: dict[float, float]
    noise: NoiseModel
    animal_nll: dict[str, float]
    animal_traces: dict[str, int]
    animal_samples: dict[str, int]
    mse: float
    coverage: dict[str, float]
    n_params: int
    fit_extra: dict[str, Any] = field(default_factory=dict)


def rows_of(data: TaskData, animal_mask: BoolArray) -> IntArray:
    return np.nonzero(animal_mask[data.trace_animal])[0]


def _subsample(rows: IntArray, n: int) -> IntArray:
    if rows.size <= n:
        return rows
    return rows[np.linspace(0, rows.size - 1, n).round().astype(np.int64)]


def residuals(data: TaskData, designs: FloatArray, fit: Fit, rows: IntArray) -> tuple[FloatArray, BoolArray]:
    mu = predict(data, designs, fit.c, fit.beta, rows)
    return data.y[rows].astype(np.float64) - mu, data.m[rows]


def noise_for(data: TaskData, designs: FloatArray, fit: Fit, rows: IntArray, n: int = NOISE_TRACES) -> NoiseModel:
    sub = _subsample(rows, n)
    e, v = residuals(data, designs, fit, sub)
    return fit_noise(e, v, data.s[sub])


def score_rows(
    data: TaskData, designs: FloatArray, fit: Fit, noise: NoiseModel, rows: IntArray
) -> tuple[FloatArray, FloatArray, BoolArray]:
    e, v = residuals(data, designs, fit, rows)
    return ar1_nll(e, v, noise.sigma(data.s[rows]), noise.phi), e, v


def make_fitter(data: TaskData, family: str) -> Fitter:
    def fitter(ps: PairStats, lam_rel: float, init: Fit | None) -> Fit:
        return fit_family(family, ps, data.pair_target, len(data.targets), lam_rel, init=init)

    return fitter


def evaluate_fold(
    data: TaskData,
    stats: Stats,
    designs: FloatArray,
    train: BoolArray,
    test: BoolArray,
    inner: Sequence[tuple[BoolArray, BoolArray]],
    family: str,
    fitter: Fitter | None = None,
    lam_grid: Sequence[float] = LAM_GRID,
) -> FoldResult:
    """Inner-loop ridge selection, outer refit, single scoring of the held-out animals."""
    if np.any(train & test):
        raise RuntimeError("outer training and test animals overlap")
    fitter = fitter or make_fitter(data, family)
    n_pairs = len(data.pairs)
    grid = (0.0,) if family == "B0" else tuple(lam_grid)
    inner_scores = {lam: 0.0 for lam in grid}
    last_inner: dict[float, Fit] = {}  # inner fits are on subsets of the outer training animals
    for itrain, ival in inner:
        if np.any(itrain & ival) or np.any((itrain | ival) & test):
            raise RuntimeError("inner folds leak the outer test animals")
        ps = aggregate(stats, itrain, n_pairs)
        fits: dict[float, Fit] = {}
        warm: Fit | None = None
        for lam in sorted(grid):  # warm-start along the ridge path, within this inner training set
            warm = fits[lam] = fitter(ps, lam, warm)
        trows_inner = rows_of(data, itrain)
        vrows = rows_of(data, ival)
        for lam, f in fits.items():
            # Each ridge value is scored with the noise fitted on its own training residuals, exactly as the
            # outer refit is; one shared noise model would favour whichever ridge value it was fitted on.
            noise = noise_for(data, designs, f, trows_inner, INNER_NOISE_TRACES)
            nll, _, _ = score_rows(data, designs, f, noise, vrows)
            inner_scores[lam] += float(nll.sum())
        last_inner = fits
    lam_best = min(grid, key=lambda lam: (inner_scores[lam], lam))
    ps = aggregate(stats, train, n_pairs)
    fit = fitter(ps, lam_best, last_inner.get(lam_best))
    noise = noise_for(data, designs, fit, rows_of(data, train))
    trows = rows_of(data, test)
    nll, e, v = score_rows(data, designs, fit, noise, trows)
    animals = np.asarray(data.animals, dtype=object)[data.trace_animal[trows]]
    animal_nll: dict[str, float] = {}
    animal_traces: dict[str, int] = {}
    animal_samples: dict[str, int] = {}
    for a, x, vv in zip(animals, nll, v.sum(axis=1), strict=True):
        animal_nll[a] = animal_nll.get(a, 0.0) + float(x)
        animal_traces[a] = animal_traces.get(a, 0) + 1
        animal_samples[a] = animal_samples.get(a, 0) + int(vv)
    mse = float(np.sum(np.where(v, e, 0.0) ** 2) / max(int(v.sum()), 1))
    extra = {k: val for k, val in fit.extra.items() if k == "lam"}
    return FoldResult(
        family=family,
        lam_rel=lam_best,
        inner_scores=inner_scores,
        noise=noise,
        animal_nll=animal_nll,
        animal_traces=animal_traces,
        animal_samples=animal_samples,
        mse=mse,
        coverage=interval_coverage(e, v, noise.sigma(data.s[trows])) if trows.size else {},
        n_params=fit.n_params,
        fit_extra=extra,
    )


def grouped_inner(train: BoolArray, n_inner: int, order_key: Callable[[int], str]) -> list[tuple[BoolArray, BoolArray]]:
    """Inner folds grouped by animal over the training animals, dealt in a deterministic order."""
    idx = sorted(np.nonzero(train)[0].tolist(), key=order_key)
    out = []
    for f in range(n_inner):
        val = np.zeros_like(train)
        val[idx[f::n_inner]] = True
        out.append((train & ~val, val))
    return out
