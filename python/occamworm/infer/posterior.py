"""Approximate posterior over candidate programs (§7.7, §8.2; OW-014).

    p(R | D) proportional to p(D_val | R) * 2^(-L_struct(R))

``p(D_val | R)`` is the cross-validated predictive likelihood (inner-validation NLL), so parameter complexity is
already paid for out of sample; ``L_params`` is reported, not charged twice (§3.5). The posterior is a labelled
approximation over the evaluated candidates only (§7.7: "maintain likelihood-weighted particles").
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
LN2 = math.log(2.0)


@dataclass
class Hypothesis:
    """§8.2 hypothesis record. ``predict(intervention)`` returns parameter particles (K, n) of predicted means."""

    program_hash: str
    grammar_version: str
    l_struct_bits: float
    l_params_bits: float
    validation_nll: float
    training_nll: float | None = None
    heldout_score_summary: dict[str, Any] | None = None
    biological_constraint_violations: list[str] = field(default_factory=list)
    posterior_weight: float = 0.0
    predict: Callable[[Any], FloatArray] | None = None


@dataclass(frozen=True)
class HypothesisEdge:
    source_program_hash: str
    target_program_hash: str
    mutation_operator: str
    edit_cost_bits: float
    evaluation_round: int


def posterior_weights(hyps: Sequence[Hypothesis], temperature: float = 1.0) -> FloatArray:
    """Normalized weights; ``temperature`` > 1 flattens the likelihood (a declared robustness option)."""
    logw = np.array([-h.validation_nll / temperature - h.l_struct_bits * LN2 for h in hyps], dtype=np.float64)
    logw -= np.max(logw)
    w = np.exp(logw)
    w /= w.sum()
    for h, x in zip(hyps, w, strict=True):
        h.posterior_weight = float(x)
    return w


def effective_number(w: FloatArray) -> float:
    return float(1.0 / np.sum(np.asarray(w) ** 2))


def top_diverse(
    hyps: Sequence[Hypothesis], distance: Callable[[Hypothesis, Hypothesis], float], k: int, min_distance: float
) -> list[Hypothesis]:
    """Greedy: highest posterior weight first, skipping candidates closer than ``min_distance`` to one chosen."""
    chosen: list[Hypothesis] = []
    for h in sorted(hyps, key=lambda x: (-x.posterior_weight, x.program_hash)):
        if all(distance(h, c) >= min_distance for c in chosen):
            chosen.append(h)
        if len(chosen) == k:
            break
    return chosen
