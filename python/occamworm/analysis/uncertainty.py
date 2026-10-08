"""Animal-level bootstrap of paired score differences (§11.5)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def paired_difference(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    reps: int = 2000,
    seed: int = 20261008,
    level: float = 0.95,
) -> dict[str, Any]:
    """``Delta NLL_a = NLL_a(baseline) - NLL_a(candidate)`` per animal; positive favours the candidate.

    Resamples animals (the independence unit) with replacement; reports the mean and median difference with
    percentile intervals and the number of animal clusters.
    """
    animals = sorted(set(baseline) & set(candidate))
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate were scored on different animals")
    d = np.array([baseline[a] - candidate[a] for a in animals], dtype=np.float64)
    if d.size == 0:
        return {"n_animals": 0}
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, d.size, size=(reps, d.size))
    boot_mean = d[idx].mean(axis=1)
    boot_median = np.median(d[idx], axis=1)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {
        "n_animals": int(d.size),
        "mean": float(d.mean()),
        "mean_ci": [float(np.quantile(boot_mean, lo)), float(np.quantile(boot_mean, hi))],
        "median": float(np.median(d)),
        "median_ci": [float(np.quantile(boot_median, lo)), float(np.quantile(boot_median, hi))],
        "fraction_animals_favouring_candidate": float(np.mean(d > 0)),
        "max_single_animal_share": float(np.max(np.abs(d)) / np.sum(np.abs(d))) if np.any(d) else 0.0,
        "level": level,
        "reps": reps,
        "seed": seed,
    }
