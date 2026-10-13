"""Locked analysis of the M4 search (PREREGISTRATION_M4.md §7, §8 and §10; OW-017).

Inputs are per held-out animal and come from frozen selections only: summed held-out NLL of the rule track, the
baseline track and B0; the same two tracks with the traces of the top-amplitude pairs removed; per-sample NLL on
non-responsive pairs; pooled interval coverage; and each animal's recording day. ``h2_verdict`` applies the
pre-registered criteria with the thresholds of the search config's ``analysis`` block and reports every one of them,
whether or not H2 passes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from occamworm.analysis.uncertainty import paired_difference

NominalCoverage = Mapping[str, float]  # level ("0.50", "0.80", "0.95") -> observed fraction


@dataclass(frozen=True)
class H2Inputs:
    baseline: Mapping[str, float]  # animal -> summed held-out NLL, baseline track
    rule: Mapping[str, float]  # animal -> summed held-out NLL, rule track
    b0: Mapping[str, float]  # animal -> summed held-out NLL, B0
    day: Mapping[str, str]  # animal -> recording day
    baseline_trimmed: Mapping[str, float]  # without traces of the top-amplitude pairs
    rule_trimmed: Mapping[str, float]
    null_baseline: Mapping[str, tuple[float, int]]  # animal -> (NLL, samples) on non-responsive pairs
    null_rule: Mapping[str, tuple[float, int]]
    coverage_baseline: NominalCoverage
    coverage_rule: NominalCoverage


def _diff(baseline: Mapping[str, float], rule: Mapping[str, float]) -> dict[str, float]:
    if set(baseline) != set(rule):
        raise ValueError("tracks were scored on different animals")
    return {a: baseline[a] - rule[a] for a in sorted(baseline)}


def cluster_bootstrap_mean(
    diff: Mapping[str, float], cluster: Mapping[str, str], reps: int, seed: int, level: float
) -> list[float]:
    """Percentile interval of the per-animal mean, resampling whole clusters (recording days) with replacement."""
    groups: dict[str, list[float]] = {}
    for a, d in diff.items():
        groups.setdefault(cluster[a], []).append(d)
    keys = sorted(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, len(keys), size=(reps, len(keys)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return [float(np.quantile(boot, lo)), float(np.quantile(boot, hi))]


def drop_one_min_mean(diff: Mapping[str, float], group: Mapping[str, str]) -> dict[str, Any]:
    """Smallest mean difference after dropping each group in turn (an animal, or a recording day)."""
    worst, which = np.inf, ""
    for g in sorted(set(group[a] for a in diff)):
        rest = [d for a, d in diff.items() if group[a] != g]
        m = float(np.mean(rest)) if rest else np.nan
        if m < worst:
            worst, which = m, g
    return {"min_mean": float(worst), "dropped": which}


def null_margin_upper(
    rule: Mapping[str, tuple[float, int]], baseline: Mapping[str, tuple[float, int]], reps: int, seed: int, level: float
) -> dict[str, float]:
    """Mean per-sample NLL difference (rule - baseline) on non-responsive pairs, with its animal-bootstrap upper bound.

    The per-sample mean is a ratio of sums over animals (NLL difference / samples), so animals with more null samples
    weigh more; the bootstrap resamples animals.
    """
    animals = sorted(set(rule) & set(baseline))
    if set(rule) != set(baseline):
        raise ValueError("tracks were scored on different animals")
    d = np.array([rule[a][0] - baseline[a][0] for a in animals], dtype=np.float64)
    n = np.array([rule[a][1] for a in animals], dtype=np.float64)
    if not n.sum():
        return {"mean_per_sample": float("nan"), "upper": float("nan"), "samples": 0}
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, d.size, size=(reps, d.size))
    boot = d[idx].sum(axis=1) / np.maximum(n[idx].sum(axis=1), 1.0)
    return {
        "mean_per_sample": float(d.sum() / n.sum()),
        "upper": float(np.quantile(boot, 1 - (1 - level) / 2)),
        "samples": int(n.sum()),
    }


def gap_fraction(
    b0: Mapping[str, float],
    rule: Mapping[str, float],
    baseline: Mapping[str, float],
    reps: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    """(B0 - rule) / (B0 - baseline) on summed held-out NLL, with an animal-bootstrap interval."""
    animals = sorted(b0)
    num = np.array([b0[a] - rule[a] for a in animals], dtype=np.float64)
    den = np.array([b0[a] - baseline[a] for a in animals], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, num.size, size=(reps, num.size))
    boot = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {
        "estimate": float(num.sum() / den.sum()),
        "ci": [float(np.quantile(boot, lo)), float(np.quantile(boot, hi))],
    }


def calibration_check(rule: NominalCoverage, baseline: NominalCoverage, margin: float) -> dict[str, Any]:
    """Per level: |coverage - nominal| of the rule track may exceed the baseline track's by at most ``margin``."""
    rows = {}
    for level in sorted(rule):
        nominal = float(level)
        err_rule, err_base = abs(rule[level] - nominal), abs(baseline[level] - nominal)
        rows[level] = {"rule_error": err_rule, "baseline_error": err_base, "ok": err_rule <= err_base + margin}
    return {"levels": rows, "ok": all(r["ok"] for r in rows.values())}


def h2_verdict(x: H2Inputs, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Every §7 criterion of the pre-registration, and whether H2 is supported (criteria 1-3 all pass)."""
    boot = cfg["bootstrap"]
    reps, seed, level = int(boot["replicates"]), int(boot["seed"]), float(boot["level"])
    diff = _diff(x.baseline, x.rule)
    primary = paired_difference(x.baseline, x.rule, reps, seed, level)
    c1 = primary["mean"] > 0 and primary["mean_ci"][0] > 0
    by_animal = drop_one_min_mean(diff, {a: a for a in diff})
    by_day = drop_one_min_mean(diff, x.day)
    trimmed = float(np.mean(list(_diff(x.baseline_trimmed, x.rule_trimmed).values())))
    c2 = primary["median"] > 0 and by_animal["min_mean"] > 0 and by_day["min_mean"] > 0 and trimmed > 0
    calibration = calibration_check(x.coverage_rule, x.coverage_baseline, float(cfg["calibration_margin"]))
    null = null_margin_upper(x.null_rule, x.null_baseline, reps, seed, level)
    c3 = calibration["ok"] and null["upper"] <= float(cfg["null_nll_margin_nats_per_sample"])
    return {
        "primary": primary,
        "day_clustered_mean_ci": cluster_bootstrap_mean(diff, x.day, reps, seed, level),
        "criteria": {
            "1_improvement": {"ok": bool(c1)},
            "2_robustness": {
                "ok": bool(c2),
                "median": primary["median"],
                "drop_one_animal": by_animal,
                "drop_one_day": by_day,
                "mean_without_top_amplitude_pairs": trimmed,
            },
            "3_no_deterioration": {"ok": bool(c3), "calibration": calibration, "null_pairs": null},
        },
        "gap_fraction": gap_fraction(x.b0, x.rule, x.baseline, reps, seed, level),
        "h2_supported": bool(c1 and c2 and c3),
    }
