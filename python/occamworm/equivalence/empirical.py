"""Observer-relative empirical equivalence (§9, OW-014).

An observer is a declared protocol: interventions, measurement operator, horizon and grid, a divergence and a
tolerance (§9.1). Here the measurement noise is the fitted AR(1) model shared by the candidates, so the divergence
between two Gaussian predictive distributions with equal covariance is the symmetric KL divergence

    d(M1, M2 | a) = 0.5 * (mu1 - mu2)' Sigma^{-1} (mu1 - mu2)

computed exactly by AR(1) whitening. Equivalence is *empirical* over the tested interventions only: the stored
result is ``max_a d`` with the intervention list, never a claim about the supremum over all inputs (§9.1). The
indistinguishability graph is reported through its maximal cliques, not transitive closures (§9.4).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from occamworm.analysis.uncertainty import paired_difference

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Observer:
    name: str
    level: str  # F0..F5 (§9.2)
    interventions: tuple[Any, ...]
    measurement: str
    horizon_volumes: int
    epsilon: float  # nats


def whitened(x: FloatArray, sigma: FloatArray, phi: float) -> FloatArray:
    """Whiten AR(1) residual-scale rows: Sigma^{-1/2} x for stationary AR(1) with per-row scale."""
    x = np.asarray(x, dtype=np.float64) / np.asarray(sigma)[:, None]
    out = np.empty_like(x)
    out[:, 0] = x[:, 0]
    out[:, 1:] = (x[:, 1:] - phi * x[:, :-1]) / np.sqrt(1 - phi**2)
    return out


def divergence(mu1: FloatArray, mu2: FloatArray, sigma: FloatArray, phi: float) -> float:
    z = whitened(mu1 - mu2, sigma, phi)
    return float(0.5 * np.sum(z**2))


def distance_matrix(
    predict: Sequence[Callable[[Any], FloatArray]], observer: Observer, sigma: FloatArray, phi: float
) -> FloatArray:
    """max over the observer's interventions of the pairwise divergence (empirical sup)."""
    n = len(predict)
    d = np.zeros((n, n))
    for a in observer.interventions:
        mus = [np.asarray(p(a), dtype=np.float64) for p in predict]
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = max(d[i, j], divergence(mus[i], mus[j], sigma, phi))
    return d


def maximal_cliques(adj: npt.NDArray[np.bool_]) -> list[list[int]]:
    """Bron-Kerbosch with pivoting; deterministic order."""
    n = adj.shape[0]
    nbr = [set(np.nonzero(adj[i])[0].tolist()) - {i} for i in range(n)]
    out: list[list[int]] = []

    def bk(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            out.append(sorted(r))
            return
        pivot = max(p | x, key=lambda u: (len(nbr[u] & p), -u))
        for v in sorted(p - nbr[pivot]):
            bk(r | {v}, p & nbr[v], x & nbr[v])
            p = p - {v}
            x = x | {v}

    bk(set(), set(range(n)), set())
    return sorted(out, key=lambda c: (-len(c), c))


def indistinguishability(d: FloatArray, epsilon: float) -> dict[str, Any]:
    adj = d <= epsilon
    np.fill_diagonal(adj, True)
    return {
        "epsilon": epsilon,
        "edges": int((adj.sum() - adj.shape[0]) // 2),
        "maximal_cliques": maximal_cliques(adj),
        "note": "empirical over the tested interventions only; cliques, not transitive closures (§9.4)",
    }


def operational_necessity(
    full: dict[str, float], ablated: dict[str, float], margin: float, reps: int = 2000, seed: int = 20261008
) -> dict[str, Any]:
    """§9.5: a feature is operationally necessary if removing it worsens held-out per-animal NLL by more than
    ``margin`` with the lower 95% bound above the margin."""
    d = paired_difference(full, ablated, reps=reps, seed=seed)  # ablated -> full; positive favours ``ablated``
    loss = {"mean": -d["mean"], "ci": [-d["mean_ci"][1], -d["mean_ci"][0]]}
    return {
        "loss_from_ablation": loss,
        "margin": margin,
        "necessary": loss["ci"][0] > margin,
        "n_animals": d["n_animals"],
    }
