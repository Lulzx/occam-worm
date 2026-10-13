"""The locked M4 analysis applies the pre-registered criteria (PREREGISTRATION_M4.md §7)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from occamworm.search.analysis import (
    H2Inputs,
    cluster_bootstrap_mean,
    drop_one_min_mean,
    gap_fraction,
    h2_verdict,
    null_margin_upper,
)

CFG = json.loads((Path(__file__).parents[3] / "configs" / "searches" / "atlas-v1.json").read_text())["analysis"]
GOOD_COV = {"0.50": 0.51, "0.80": 0.81, "0.95": 0.95}


def _inputs(effect: float, n: int = 60, seed: int = 0) -> H2Inputs:
    rng = np.random.default_rng(seed)
    animals = [f"a{i:03d}" for i in range(n)]
    base = {a: 1000.0 + rng.normal(0, 5) for a in animals}
    rule = {a: base[a] - effect + rng.normal(0, 2) for a in animals}
    b0 = {a: base[a] + 50.0 for a in animals}
    null = {a: (100.0, 1000) for a in animals}
    return H2Inputs(
        baseline=base,
        rule=rule,
        b0=b0,
        day={a: f"d{i // 3}" for i, a in enumerate(animals)},
        baseline_trimmed=base,
        rule_trimmed=rule,
        null_baseline=null,
        null_rule=null,
        coverage_baseline=GOOD_COV,
        coverage_rule=GOOD_COV,
    )


def test_clear_improvement_supports_h2() -> None:
    v = h2_verdict(_inputs(effect=3.0), CFG)
    assert v["h2_supported"]
    assert all(c["ok"] for c in v["criteria"].values())
    lo, hi = v["day_clustered_mean_ci"]
    assert lo > 0 and hi > lo
    assert v["gap_fraction"]["estimate"] == pytest.approx((50 + 3) / 50, rel=0.02)


def test_no_improvement_fails_h2() -> None:
    v = h2_verdict(_inputs(effect=0.0), CFG)
    assert not v["criteria"]["1_improvement"]["ok"] and not v["h2_supported"]


def test_one_animal_cannot_carry_the_result() -> None:
    x = _inputs(effect=0.0)
    rule = dict(x.rule)
    rule["a000"] = x.baseline["a000"] - 5000.0  # one animal fitted extremely well
    v = h2_verdict(dataclasses.replace(x, rule=rule), CFG)
    crit = v["criteria"]["2_robustness"]
    assert crit["drop_one_animal"]["dropped"] == "a000" and not crit["ok"]
    assert not v["h2_supported"]


def test_calibration_and_null_margins_are_enforced() -> None:
    x = _inputs(effect=3.0)
    bad_cov = {"0.50": 0.62, "0.80": 0.81, "0.95": 0.95}
    v = h2_verdict(dataclasses.replace(x, coverage_rule=bad_cov), CFG)
    assert not v["criteria"]["3_no_deterioration"]["calibration"]["ok"] and not v["h2_supported"]
    worse_null = {a: (100.0 + 0.05 * 1000, 1000) for a in x.null_rule}  # +0.05 nats per sample
    v = h2_verdict(dataclasses.replace(x, null_rule=worse_null), CFG)
    assert v["criteria"]["3_no_deterioration"]["null_pairs"]["upper"] > 0.02 and not v["h2_supported"]


def test_cluster_bootstrap_is_wider_when_days_share_the_effect() -> None:
    rng = np.random.default_rng(1)
    animals = [f"a{i}" for i in range(60)]
    day = {a: f"d{i // 6}" for i, a in enumerate(animals)}
    shift = {f"d{k}": rng.normal(0, 3) for k in range(10)}
    diff = {a: 1.0 + shift[day[a]] + rng.normal(0, 0.3) for a in animals}
    per_day = cluster_bootstrap_mean(diff, day, 2000, 0, 0.95)
    per_animal = cluster_bootstrap_mean(diff, {a: a for a in animals}, 2000, 0, 0.95)
    assert per_day[1] - per_day[0] > 1.5 * (per_animal[1] - per_animal[0])


def test_helpers() -> None:
    assert drop_one_min_mean({"a": 10.0, "b": 1.0, "c": 1.0}, {"a": "x", "b": "y", "c": "y"}) == {
        "min_mean": 1.0,
        "dropped": "x",
    }
    up = null_margin_upper({"a": (11.0, 10), "b": (21.0, 10)}, {"a": (10.0, 10), "b": (20.0, 10)}, 200, 0, 0.95)
    assert up["mean_per_sample"] == pytest.approx(0.1) and up["samples"] == 20
    g = gap_fraction({"a": 10.0}, {"a": 5.0}, {"a": 0.0}, 10, 0, 0.95)
    assert g["estimate"] == pytest.approx(0.5)
