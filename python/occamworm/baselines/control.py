"""Indicator-kinetics control (§4.5, OW-005/OW-006).

1. Indicator impulse response from autoresponses, with an animal-level bootstrap, plus the per-fold estimates the
   T2b benchmark actually used (training animals only).
2. Resolvable kernel bandwidth from that indicator and the noise fitted by B0.
3. Shared (B1) kernel timescales in T2b next to the indicator's.
4. False-sharing rate: synthetic data with *distinct* pair kernels behind the estimated indicator and the measured
   noise, run through the same nested evaluation; the rate is how often B1 or a B2 bank wins on held-out animals.

The interpretation rule of §4.5 is applied with the threshold frozen in the experiment config (open decision 14).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from occamworm.analysis.dataset import PRE, dataset_dir, load_windows
from occamworm.baselines.benchmark import eligible_targets, load_experiment
from occamworm.baselines.data import KERNEL_BASIS, aggregate, build_task, compute_stats, trial_designs
from occamworm.baselines.evaluate import evaluate_fold, grouped_inner
from occamworm.baselines.indicator import Indicator, estimate_from_autoresponses, resolvable_bandwidth
from occamworm.baselines.kernels import fit_shared
from occamworm.baselines.summary import load_rows
from occamworm.baselines.synthetic import SyntheticSpec, generate

FloatArray = npt.NDArray[np.float64]
SHARING = ("B1", "B2-K1", "B2-K2")
CONTROL_FAMILIES = ("B1", "B2-K1", "B2-K2", "B3")


def _timescales(kernel: FloatArray, dt: float) -> dict[str, float | None]:
    """Peak latency and half-decay after the peak of |kernel| (seconds)."""
    k = np.abs(kernel)
    if not np.any(k > 0):
        return {"peak_s": None, "half_decay_s": None}
    p = int(np.argmax(k))
    after = np.nonzero(k[p:] <= 0.5 * k[p])[0]
    return {"peak_s": p * dt, "half_decay_s": float(after[0] * dt) if after.size else None}


def false_sharing_rate(
    ind: Indicator,
    noise_a: float,
    noise_phi: float,
    s_range: tuple[float, float],
    missing: float,
    task: str,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    wins: dict[str, int] = {f: 0 for f in CONTROL_FAMILIES}
    for k in range(reps):
        spec = SyntheticSpec(
            "independent",
            n_animals=12,
            n_targets=4,
            n_responders=8,
            reps=1,
            noise_a=noise_a,
            noise_phi=noise_phi,
            s_range=s_range,
            missing=missing,
            indicator=ind,
            task=task,
            seed=seed + k,
        )
        data, _ = generate(spec)
        designs = trial_designs(data, ind if task == "T2b" else None)
        stats = compute_stats(data, designs)
        n = len(data.animals)
        totals = dict.fromkeys(CONTROL_FAMILIES, 0.0)
        for fold in range(4):
            test = np.zeros(n, dtype=bool)
            test[fold::4] = True
            names = list(data.animals)
            inner = grouped_inner(~test, 3, names.__getitem__)
            for fam in CONTROL_FAMILIES:
                r = evaluate_fold(data, stats, designs, ~test, test, inner, fam, lam_grid=(0.1, 1.0, 10.0, 100.0))
                totals[fam] += sum(r.animal_nll.values())
        wins[min(totals, key=lambda f: totals[f])] += 1
    rate = sum(wins[f] for f in SHARING) / reps
    return {"task": task, "replicates": reps, "wins": wins, "false_sharing_rate": rate}


def run_control(root: Path, out: Path, replicates: int | None = None) -> dict[str, Any]:
    exp = load_experiment(root)
    ctl = exp["indicator_control"]
    reps = replicates or int(ctl["false_sharing_replicates"])
    w = load_windows(dataset_dir(root, exp["data"]["dataset"]))
    elig = eligible_targets(root, exp)
    data = build_task(w, elig, "T2b", history=False)
    ok = data.trial_auto_ok
    traces = data.trial_auto[ok, PRE:]
    animals = [data.animals[a] for a in data.trial_animal[ok]]
    est = estimate_from_autoresponses(traces, np.ones_like(traces, dtype=bool), animals, data.dt, reps=200)
    ind: Indicator = est["indicator"]
    rows = load_rows(out / "folds.jsonl") if (out / "folds.jsonl").exists() else []
    fold_inds = [
        r["indicator"] for r in rows if r.get("family") == "B0" and r.get("task") == "T2b" and "indicator" in r
    ]
    b0 = [r["noise"] for r in rows if r.get("family") == "B0" and r.get("task") == "T2b" and not r["history"]]
    if b0:
        noise_a = float(np.median([n["a"] for n in b0]))
        noise_b = float(np.median([n["b"] for n in b0]))
        noise_phi = float(np.median([n["phi"] for n in b0]))
    else:
        noise_a, noise_b, noise_phi = 1.4, 0.12, 0.4  # placeholder until the benchmark has run; flagged below
    s = data.s[data.s > 0]
    sigma_typ = float(np.median(np.sqrt(noise_a**2 * s**2 + noise_b**2)))
    pairs = pq.read_table(root / exp["data"]["eligibility"].replace("target_eligibility", "pair_coverage")).to_pylist()
    amp = float(np.median([abs(p["mean_amplitude"]) for p in pairs if abs(p["t_statistic"] or 0) >= 3]))
    bandwidth = resolvable_bandwidth(ind, amp, sigma_typ, noise_phi, data.dt)

    # shared kernels in T2b (descriptive: all animals, most frequently selected ridge)
    lam_rel = 10.0
    b1_rows = [r for r in rows if r.get("family") == "B1" and r.get("task") == "T2b" and not r["history"]]
    if b1_rows:
        lams = [r["lam_rel"] for r in b1_rows]
        lam_rel = max(set(lams), key=lams.count)
    designs = trial_designs(data, ind)
    ps = aggregate(compute_stats(data, designs), np.ones(len(data.animals), dtype=bool), len(data.pairs))
    fit = fit_shared(ps, data.pair_target, len(data.targets), lam_rel)
    lags = np.arange(KERNEL_BASIS.shape[0])
    ind_k = ind.kernel(lags.size, data.dt)
    ind_ts = _timescales(ind_k, data.dt)
    kern_rows: list[dict[str, Any]] = []
    for j, target in enumerate(data.targets):
        neural = KERNEL_BASIS @ fit.extra["kernels"][j]
        observed = np.convolve(ind_k, neural)[: lags.size]
        kern_rows.append(
            {"target": target, "neural": _timescales(neural, data.dt), "observed": _timescales(observed, data.dt)}
        )
    boot = []
    rng = np.random.default_rng(int(ctl["false_sharing_seed"]))
    uniq = sorted(set(animals))
    an = np.asarray(animals, dtype=object)
    for _ in range(100):
        pick = set(rng.choice(uniq, len(uniq)))
        sel = np.array([a in pick for a in an])
        e = estimate_from_autoresponses(
            traces[sel], np.ones_like(traces[sel], dtype=bool), list(an[sel]), data.dt, reps=0
        )
        boot.append(_timescales(e["indicator"].kernel(lags.size, data.dt), data.dt)["half_decay_s"] or np.nan)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    obs_decay = [k["observed"]["half_decay_s"] for k in kern_rows if k["observed"]["half_decay_s"] is not None]
    differs = float(np.mean([(d < lo) or (d > hi) for d in obs_decay])) if obs_decay else None

    s_lo, s_hi = (float(np.quantile(s, 0.1)), float(np.quantile(s, 0.9)))
    missing = float(1.0 - data.m.mean())
    fs = [
        false_sharing_rate(ind, noise_a, noise_phi, (s_lo, s_hi), missing, task, reps, int(ctl["false_sharing_seed"]))
        for task in ("T2a", "T2b")
    ]
    threshold = float(ctl["false_sharing_threshold"])
    low = all(f["false_sharing_rate"] <= threshold for f in fs)
    result = {
        "indicator": {
            **ind.to_json(),
            "amplitude": est["amplitude"],
            "n_traces": est["n_traces"],
            "n_animals": est["n_animals"],
            "tau_r_ci95": est["tau_r_ci95"],
            "tau_d_ci95": est["tau_d_ci95"],
            "interpretation": est["interpretation"],
            "timescales": ind_ts,
            "half_decay_ci95": [float(lo), float(hi)],
        },
        "fold_indicators": {
            "n": len(fold_inds),
            "tau_r_s": [
                float(np.min([f["tau_r_s"] for f in fold_inds])),
                float(np.max([f["tau_r_s"] for f in fold_inds])),
            ]
            if fold_inds
            else None,
            "tau_d_s": [
                float(np.min([f["tau_d_s"] for f in fold_inds])),
                float(np.max([f["tau_d_s"] for f in fold_inds])),
            ]
            if fold_inds
            else None,
        },
        "noise": {
            "a": noise_a,
            "b": noise_b,
            "phi": noise_phi,
            "typical_sigma": sigma_typ,
            "source": "median of B0 T2b fold fits" if b0 else "placeholder (benchmark not run)",
        },
        "resolvable_bandwidth": bandwidth,
        "shared_kernels_t2b": {
            "lam_rel": lam_rel,
            "per_target": kern_rows,
            "fraction_observed_half_decay_outside_indicator_ci": differs,
        },
        "false_sharing": fs,
        "threshold": threshold,
        "interpretation": {
            "false_sharing_low": low,
            "kernels_differ_from_indicator": differs,
            "h1_claim_allowed_as_neural": bool(low and differs is not None and differs > 0.5),
            "rule": "§4.5: neural claim only if (a) shared kernels differ from the indicator beyond its uncertainty "
            "and (b) the false-sharing rate is at or below the frozen threshold",
        },
    }
    (out / "indicator_control.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
