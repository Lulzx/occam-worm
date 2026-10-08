"""Expected-information-gain experiment ranking (§8.3, OW-014).

For each feasible intervention ``a`` the hypotheses predict measurement means (with parameter particles); the
measurement noise is the fitted Gaussian AR(1) model. The expected information gain

    I(R; Y_a | D) = E_{R, Y_a} [ log p(Y_a | R, a) - log p(Y_a | a) ]

is estimated by predictive Monte Carlo: draw R from the posterior weights, a particle, and Y_a = mu + noise;
score Y_a under every hypothesis (exact masked AR(1) likelihood); average. The Monte Carlo standard error is
reported with every estimate, so predictions that diverge visibly but sit inside large noise get a low, honestly
uncertain gain. Utility = EIG - cost_weight * cost. Proposals are research suggestions, not lab actuation (§8.4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from occamworm.analysis.scoring import ar1_nll

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Intervention:
    """One menu entry (§8.4). ``design`` is whatever the hypotheses' predict functions understand."""

    intervention_type: str
    target_neurons: tuple[str, ...]
    waveform: str
    duration_and_timing: str
    proposed_readout: str
    cost: float = 0.0
    feasible_controls: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    design: Any = None


@dataclass
class ExperimentProposal:
    intervention: Intervention
    expected_information_gain: float  # nats
    eig_standard_error: float
    utility: float
    predictions_by_hypothesis: dict[str, FloatArray] = field(default_factory=dict)


def _ar1_draw(rng: np.random.Generator, sigma: FloatArray, phi: float, shape: tuple[int, int]) -> FloatArray:
    n, t = shape
    e = np.empty((n, t))
    e[:, 0] = rng.standard_normal(n)
    for k in range(1, t):
        e[:, k] = phi * e[:, k - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(n)
    return e * sigma[:, None]


def expected_information_gain(
    means: Sequence[FloatArray],
    weights: FloatArray,
    sigma: FloatArray,
    phi: float,
    samples: int = 400,
    seed: int = 0,
) -> tuple[float, float]:
    """EIG (nats) and its Monte Carlo standard error.

    ``means[m]`` has shape (particles, traces, T): predicted measurement means of hypothesis m under one
    intervention; ``sigma`` (traces,) and ``phi`` define the AR(1) measurement noise.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    n_traces, t = means[0].shape[1:]
    valid = np.ones((n_traces, t), dtype=bool)
    logw = np.log(np.maximum(w, 1e-300))
    vals = np.empty(samples)
    for s in range(samples):
        m = int(rng.choice(len(means), p=w))
        k = int(rng.integers(means[m].shape[0]))
        y = means[m][k] + _ar1_draw(rng, sigma, phi, (n_traces, t))
        # log p(y | hypothesis) = log mean over particles of the AR(1) likelihood
        logp = np.array(
            [
                logsumexp([-ar1_nll(y - mu, valid, sigma, phi).sum() for mu in means[j]]) - np.log(means[j].shape[0])
                for j in range(len(means))
            ]
        )
        vals[s] = logp[m] - logsumexp(logw + logp)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(samples))


def rank_interventions(
    menu: Sequence[Intervention],
    predict: Sequence[Any],
    names: Sequence[str],
    weights: FloatArray,
    sigma: FloatArray,
    phi: float,
    cost_weight: float = 0.0,
    samples: int = 400,
    seed: int = 0,
) -> list[ExperimentProposal]:
    """``predict[m](intervention)`` -> (particles, traces, T) means for hypothesis m. Highest utility first."""
    out = []
    for i, a in enumerate(menu):
        means = [np.asarray(p(a), dtype=np.float64) for p in predict]
        eig, se = expected_information_gain(means, weights, sigma, phi, samples, seed + i)
        out.append(
            ExperimentProposal(
                a, eig, se, eig - cost_weight * a.cost, {n: m.mean(axis=0) for n, m in zip(names, means, strict=True)}
            )
        )
    return sorted(out, key=lambda p: (-p.utility, p.intervention.target_neurons))
