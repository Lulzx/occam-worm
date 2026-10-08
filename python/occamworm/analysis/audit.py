"""M0 data-audit gate (§2.3, OW-003).

Every number in the report is computed here from the normalized import and the frozen thresholds in
``configs/datasets/eligibility.json``; the Markdown report is rendered from ``audit.json`` and never edited by hand.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from occamworm.analysis.dataset import POST, PRE, Windows, dataset_dir, load_windows
from occamworm.analysis.splits import SplitSpec, build_split, target_coverage

AUDIT_VERSION = "audit-v1"
INDICATOR = {
    "name": "GCaMP6s, nuclear-localized",
    "strain": "AML462 (pan-neuronal nuclear GCaMP6s, GUR-3/PRDX-2 optogenetics, NeuroPAL)",
    "reference": "Randi et al. 2023, Nature 623:406 (arXiv:2208.04790), Methods; CGC strain AML462",
    "published_kinetics": None,
    "published_kinetics_note": (
        "No kinetic parameters for nuclear GCaMP6s in this preparation are taken from the literature yet; "
        "the indicator impulse response is estimated from autoresponses (OW-005)."
    ),
}


def load_config(root: Path) -> dict[str, Any]:
    return dict(json.loads((root / "configs" / "datasets" / "eligibility.json").read_text()))


def _git_revision(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def response_amplitude(w: Windows, cfg: dict[str, Any]) -> npt.NDArray[np.float64]:
    """Mean dF/F over the response offsets; NaN where too few samples are valid."""
    lo, hi = cfg["response_offsets"]
    cols = slice(PRE + lo, PRE + hi + 1)
    v = w.valid[:, cols]
    n = v.sum(axis=1)
    total = np.where(v, w.dff[:, cols], 0.0).sum(axis=1, dtype=np.float64)
    need = cfg["min_response_valid_fraction"] * (hi - lo + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= need, total / np.maximum(n, 1), np.nan)


def _quantiles(x: npt.ArrayLike) -> dict[str, float | int | None]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    q = np.quantile(a, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "n": int(a.size),
        "min": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "max": float(q[4]),
    }


def _strata(counts: list[int]) -> dict[str, int]:
    edges = [("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, 10**9)]
    return {name: sum(lo <= c <= hi for c in counts) for name, lo, hi in edges}


def _animal_bootstrap_mean(values: list[float], reps: int, seed: int) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    if a.size < 2:
        return {"n_animals": int(a.size), "mean": float(a.mean()) if a.size else None, "ci95": None}
    rng = np.random.Generator(np.random.PCG64(seed))
    boots = a[rng.integers(0, a.size, size=(reps, a.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"n_animals": int(a.size), "mean": float(a.mean()), "ci95": [float(lo), float(hi)]}


def _slope(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.ptp(x[ok]) == 0:
        return None
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def _variance_components(groups: list[list[float]]) -> tuple[float, float] | None:
    """One-way random-effects ANOVA estimates (within, between) for per-animal lists of trial values."""
    groups = [g for g in groups if g]
    n = sum(len(g) for g in groups)
    k = len(groups)
    if k < 2 or n - k < 1:
        return None
    means = [float(np.mean(g)) for g in groups]
    grand = float(np.mean(np.concatenate([np.asarray(g) for g in groups])))
    ss_w = sum(float(np.sum((np.asarray(g) - m) ** 2)) for g, m in zip(groups, means, strict=True))
    ss_b = sum(len(g) * (m - grand) ** 2 for g, m in zip(groups, means, strict=True))
    ms_w, ms_b = ss_w / (n - k), ss_b / (k - 1)
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    return ms_w, max(0.0, (ms_b - ms_w) / n0)


def _autoresponse_kinetics(w: Windows, cfg: dict[str, Any], rows: npt.NDArray[np.int64]) -> dict[str, Any]:
    """Peak latency and half-decay time of strong autoresponses: the data constraining indicator kinetics."""
    dt = w.volume_interval_s
    peaks, half_decays, rises = [], [], []
    for r in rows:
        y = w.dff[r, PRE:].astype(np.float64)
        v = w.valid[r, PRE:]
        if v[:20].sum() < 15 or not np.isfinite(w.pre_sd[r]) or w.pre_sd[r] <= 0:
            continue
        early = np.where(v[:20], y[:20], -np.inf)
        k = int(np.argmax(early))
        peak = early[k]
        if peak < cfg["autoresponse_min_peak_over_noise"] * float(w.pre_sd[r]):
            continue
        peaks.append(k * dt)
        above = np.nonzero(v[: k + 1] & (y[: k + 1] >= 0.5 * peak))[0]
        if above.size:
            rises.append(float(above[0]) * dt)
        after = np.nonzero(v[k:] & (y[k:] <= 0.5 * peak))[0]
        if after.size:
            half_decays.append(float(after[0]) * dt)
    return {
        "n_strong_autoresponses": len(peaks),
        "peak_latency_s": _quantiles(peaks),
        "time_to_half_peak_s": _quantiles(rises),
        "half_decay_after_peak_s": _quantiles(half_decays),
        "half_decay_censored": len(peaks) - len(half_decays),
        "note": (
            "Window resolution is the 0.5 s volume interval; decay is censored at the window end or next stimulation."
        ),
    }


def run_audit(root: Path, out_dir: Path, windows: Windows | None = None) -> dict[str, Any]:
    cfg = load_config(root)
    directory = dataset_dir(root)
    w = windows or load_windows(directory)
    t, a, nrn = w.trials, w.animals, w.neurons
    amp = response_amplitude(w, cfg)
    n_trials = len(t["trial_id"])
    trial_animal = np.asarray(t["animal_id"], dtype=object)
    w_animal = trial_animal[w.trial_index]
    target_nid = np.asarray(t["stim_target_neuron_id"], dtype=object)
    w_target = target_nid[w.trial_index]
    code = np.asarray(t["stim_target_code"], dtype=object)

    # --- counts by genotype and target code
    genotypes = Counter(str(g) for g in a["genotype"])
    counts = {
        "animals": len(set(str(x) for x in a["animal_id"])),
        "recordings": len(a["recording_id"]),
        "sessions": len(set(str(x) for x in a["session_id"])),
        "trials": n_trials,
        "animals_by_genotype": dict(sorted(genotypes.items())),
        "trials_by_target_code": dict(sorted(Counter(str(c) for c in code).items())),
        "identified_targets": len({str(x) for x in target_nid if x is not None}),
        "trials_with_identified_target": int(sum(x is not None for x in target_nid)),
        "rois": len(nrn["roi_id"]),
        "roi_label_status": dict(sorted(Counter(str(s) for s in nrn["label_status"]).items())),
        "animal_id_assumption": "one recording per animal (flag animal_id_assumed_one_recording_per_animal)",
        "genotype_coverage_note": (
            "Only wild-type is imported; the unc-31 export is registered for M6 and not counted here. "
            "No sample size is assumed for any genotype."
        ),
    }

    # --- autoresponses and missing data
    tgt_rows = np.nonzero(w.is_target)[0]
    ar_frac = np.asarray(t["autoresponse_valid_fraction"], dtype=np.float64)
    ar_ok = np.isfinite(ar_frac) & (ar_frac >= cfg["autoresponse_min_valid_fraction"])
    tgt_amp_ok = np.isfinite(amp[tgt_rows])
    quality = {
        "trials_with_autoresponse": int(np.isfinite(ar_frac).sum()),
        "autoresponses_valid": int(ar_ok.sum()),
        "autoresponse_window_amplitude_defined": int(tgt_amp_ok.sum()),
        "sham_events": 0,
        "sham_note": (
            "The export contains no sham (light-off) stimulations; B0 is fit from pre-stimulus and non-responding data."
        ),
        "window_samples": int(w.valid.size),
        "window_samples_valid_fraction": float(w.valid.mean()),
        "windows_without_baseline": int(np.isnan(w.f0).sum()),
        "windows_without_response_amplitude": int(np.isnan(amp).sum()),
        "trial_qc_flags": dict(sorted(Counter(f for fl in t["trial_qc_flags"] for f in (fl or [])).items())),
        "uncertain_identity_rois": int(sum(s != "name" for s in nrn["label_status"])),
    }

    # --- pairs: (identified target, identified responder), responder != target
    pair_trials: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    named = np.array([x is not None for x in w.neuron_id]) & np.array([x is not None for x in w_target])
    rows = np.nonzero(named & ~w.is_target & np.isfinite(amp))[0]
    for r in rows:
        tg, rs = str(w_target[r]), str(w.neuron_id[r])
        if tg != rs:
            pair_trials[(tg, rs)][str(w_animal[r])].append(float(amp[r]))
    pair_rows: list[dict[str, Any]] = []
    for (tg, rs), by_animal in sorted(pair_trials.items()):
        vals = [v for vs in by_animal.values() for v in vs]
        means = [float(np.mean(vs)) for vs in by_animal.values()]
        n = len(vals)
        sd = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        tstat = float(np.mean(vals) / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan")
        vc = _variance_components(list(by_animal.values()))
        pair_rows.append(
            {
                "target": tg,
                "responder": rs,
                "n_trials": n,
                "n_animals": len(by_animal),
                "mean_amplitude": float(np.mean(vals)),
                "sd_amplitude": sd,
                "t_statistic": tstat,
                "between_animal_sd_of_means": float(np.std(means, ddof=1)) if len(means) > 1 else float("nan"),
                "within_animal_variance": vc[0] if vc else float("nan"),
                "between_animal_variance": vc[1] if vc else float("nan"),
            }
        )
    pairs_tbl = pa.Table.from_pylist(pair_rows)
    n_animals_per_pair = [p["n_animals"] for p in pair_rows]
    vc_rows = [p for p in pair_rows if np.isfinite(p["within_animal_variance"])]
    icc = [
        p["between_animal_variance"] / (p["between_animal_variance"] + p["within_animal_variance"])
        for p in vc_rows
        if p["between_animal_variance"] + p["within_animal_variance"] > 0
    ]
    pairs = {
        "pairs_observed": len(pair_rows),
        "pairs_by_animal_count": _strata(n_animals_per_pair),
        "pairs_by_trial_count": _strata([p["n_trials"] for p in pair_rows]),
        "pairs_with_min_animals": {str(k): int(sum(c >= k for c in n_animals_per_pair)) for k in (2, 3, 5)},
        "fraction_pairs_with_min_animals": {
            str(k): (float(np.mean([c >= k for c in n_animals_per_pair])) if pair_rows else 0.0) for k in (2, 3, 5)
        },
        "variance_decomposition": {
            "pairs_estimable": len(vc_rows),
            "median_between_animal_fraction": float(np.median(icc)) if icc else None,
            "pooled_within_animal_variance": float(np.mean([p["within_animal_variance"] for p in vc_rows]))
            if vc_rows
            else None,
            "pooled_between_animal_variance": float(np.mean([p["between_animal_variance"] for p in vc_rows]))
            if vc_rows
            else None,
            "note": "One-way random-effects ANOVA per (target, responder) pair with >= 2 animals and a repeat.",
        },
    }

    # --- per target: responders observed in enough animals
    resp_animals: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for (tg, rs), by_animal in pair_trials.items():
        resp_animals[tg][rs] |= set(by_animal)
    target_animals: dict[str, set[str]] = defaultdict(set)
    for k in range(n_trials):
        if target_nid[k] is not None:
            target_animals[str(target_nid[k])].add(str(trial_animal[k]))
    target_ar_animals: dict[str, set[str]] = defaultdict(set)
    for k in np.nonzero(ar_ok)[0]:
        if target_nid[k] is not None:
            target_ar_animals[str(target_nid[k])].add(str(trial_animal[k]))
    responsive = {
        (p["target"], p["responder"])
        for p in pair_rows
        if p["n_animals"] >= cfg["responder_min_animals"]
        and np.isfinite(p["t_statistic"])
        and abs(p["t_statistic"]) >= cfg["responsive_pair_min_abs_t"]
    }
    target_rows: list[dict[str, Any]] = []
    for tg in sorted(target_animals):
        multi = sorted(rs for rs, an in resp_animals[tg].items() if len(an) >= cfg["responder_min_animals"])
        reasons = []
        if len(target_ar_animals[tg]) < cfg["target_min_animals"]:
            reasons.append(f"fewer than {cfg['target_min_animals']} animals with a valid autoresponse")
        if len(multi) < cfg["target_min_responders"]:
            reasons.append(
                f"fewer than {cfg['target_min_responders']} identified responders in "
                f">= {cfg['responder_min_animals']} animals"
            )
        target_rows.append(
            {
                "target": tg,
                "n_animals": len(target_animals[tg]),
                "n_animals_valid_autoresponse": len(target_ar_animals[tg]),
                "n_trials": int(sum(1 for k in range(n_trials) if target_nid[k] == tg)),
                "responders_any": len(resp_animals[tg]),
                "responders_multi_animal": len(multi),
                "responsive_pairs": sum(1 for rs in multi if (tg, rs) in responsive),
                "eligible": not reasons,
                "exclusion_reasons": reasons,
            }
        )
    eligible = [r["target"] for r in target_rows if r["eligible"]]
    targets = {
        "identified_targets": len(target_rows),
        "eligible_targets": len(eligible),
        "eligible_with_responsive_pair": sum(1 for r in target_rows if r["eligible"] and r["responsive_pairs"] > 0),
        "distinct_responders_per_target": _quantiles([r["responders_any"] for r in target_rows]),
        "multi_animal_responders_per_target": _quantiles([r["responders_multi_animal"] for r in target_rows]),
        "exclusions": dict(sorted(Counter(x for r in target_rows for x in r["exclusion_reasons"]).items())),
    }

    # --- stimulation timing and history
    isi = np.asarray(t["time_since_previous_stim_s"], dtype=np.float64)
    prev = np.asarray(t["previous_stim_target_label"], dtype=object)
    cur = np.asarray(t["stim_target_label"], dtype=object)
    same_prev = sum(1 for p, c in zip(prev, cur, strict=True) if p is not None and c is not None and p == c)
    post_len = np.minimum(np.asarray(t["next_stim_frame"], dtype=np.float64) - np.asarray(t["stim_frame"]), POST)
    stimulus = {
        "volume_interval_s": w.volume_interval_s,
        "sampling_rate_hz": 1.0 / w.volume_interval_s,
        "window": {"pre_volumes": PRE, "post_volumes": POST, "response_offsets": cfg["response_offsets"]},
        "stimulus_duration_s_known": int(sum(x is not None and np.isfinite(x) for x in t["stimulus_duration_s"])),
        "stimulus_amplitude_known": int(sum(x is not None and np.isfinite(x) for x in t["stimulus_amplitude"])),
        "optogenetics_type": dict(Counter(str(x) for x in t["optogenetics_type"])),
        "optogenetics_n_pulses": dict(sorted(Counter(str(x) for x in t["optogenetics_n_pulses"]).items())),
        "optogenetics_n_trains": dict(sorted(Counter(str(x) for x in t["optogenetics_n_trains"]).items())),
        "inter_stimulus_interval_s": _quantiles(isi),
        "post_window_volumes_before_next_stim": _quantiles(post_len),
        "stimulations_per_recording": _quantiles(np.asarray(a["n_stimulations"], dtype=np.float64)),
        "recording_duration_s": _quantiles(np.asarray(a["n_frames"], dtype=np.float64) * w.volume_interval_s),
        "trials_whose_previous_target_label_is_the_same": same_prev,
    }

    # --- within-recording drift (per-animal slopes, animal-level bootstrap)
    stim_idx = np.asarray(t["stim_index_in_recording"], dtype=np.float64)
    elapsed = np.asarray(t["time_since_recording_start_s"], dtype=np.float64)
    tgt_amp = np.full(n_trials, np.nan)
    tgt_amp[w.trial_index[tgt_rows]] = amp[tgt_rows]
    tgt_amp[~ar_ok] = np.nan
    resp_abs = np.full(n_trials, np.nan)
    resp_rows = np.nonzero(~w.is_target & np.isfinite(amp))[0]
    sums = np.bincount(w.trial_index[resp_rows], weights=np.abs(amp[resp_rows]), minlength=n_trials)
    cnts = np.bincount(w.trial_index[resp_rows], minlength=n_trials)
    resp_abs[cnts > 0] = sums[cnts > 0] / cnts[cnts > 0]
    f0_log = np.full(n_trials, np.nan)
    f0_rows = np.nonzero(np.isfinite(w.f0) & (w.f0 > 0))[0]
    s = np.bincount(w.trial_index[f0_rows], weights=np.log(w.f0[f0_rows].astype(np.float64)), minlength=n_trials)
    c = np.bincount(w.trial_index[f0_rows], minlength=n_trials)
    f0_log[c > 0] = s[c > 0] / c[c > 0]
    trials_by_animal: dict[str, list[int]] = defaultdict(list)
    for k in range(n_trials):
        trials_by_animal[str(trial_animal[k])].append(k)
    drift: dict[str, Any] = {}
    reps, seed = cfg["bootstrap_replicates"], cfg["bootstrap_seed"]
    for name, y, x, unit in [
        ("autoresponse_amplitude_vs_stim_index", tgt_amp, stim_idx, "dF/F per stimulation"),
        ("mean_abs_responder_amplitude_vs_stim_index", resp_abs, stim_idx, "dF/F per stimulation"),
        ("mean_log_baseline_f0_vs_elapsed_time", f0_log, elapsed / 60.0, "log units per minute"),
    ]:
        slopes = [s_ for ks in trials_by_animal.values() if (s_ := _slope(x[ks], y[ks])) is not None]
        drift[name] = {"unit": unit, **_animal_bootstrap_mean(slopes, reps, seed)}

    # --- split scheme coverage (eligible targets only)
    all_animals = sorted(set(str(x) for x in a["animal_id"]))
    elig_animals = {str(tg): target_ar_animals[str(tg)] for tg in eligible}
    schemes: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for sch in cfg["candidate_schemes"]:
        spec = SplitSpec(sch["name"], sch["outer"], sch["outer_folds"], sch["inner_folds"], cfg["split_seed"])
        split = build_split(all_animals, spec)
        cov = target_coverage(split, elig_animals)
        for row in cov:
            coverage_rows.append({"scheme": sch["name"], **row})
        schemes.append(
            {
                **spec.to_json(),
                "n_outer_folds": len(split.folds),
                "eligible_targets_covered": sum(r["covered"] for r in cov),
                "eligible_targets_tested_without_inner_coverage": sum(
                    1 for r in cov if r["outer_test_folds"] and not r["covered"]
                ),
                "median_test_animals_per_target": float(np.median([r["test_animals"] for r in cov])) if cov else None,
            }
        )
    best = max(schemes, key=lambda s: (s["eligible_targets_covered"], -s["n_outer_folds"]))
    covered_best = {r["target"] for r in coverage_rows if r["scheme"] == best["name"] and r["covered"]}
    go_targets = sorted(
        r["target"] for r in target_rows if r["eligible"] and r["responsive_pairs"] > 0 and r["target"] in covered_best
    )
    split_selection = {
        "rule": cfg["scheme_selection_rule"],
        "chosen": best["name"],
        "schemes": schemes,
        "decided_before_any_model_scored": True,
    }
    go = {
        "criterion": (
            "at least one specified collection of stimulus targets admits animal-held-out evaluation with "
            "nontrivial response variability: eligible targets covered by the chosen split that have at "
            f"least one responder with |t| >= {cfg['responsive_pair_min_abs_t']} across "
            f">= {cfg['responder_min_animals']} animals"
        ),
        "targets": go_targets,
        "decision": "go" if go_targets else "no-go",
    }

    # --- ID intersection across functional atlas, anatomy and molecular annotations (OW-013)
    intersection = _id_intersection(root, {str(x) for x in nrn["neuron_id"] if x is not None})

    kinetics = _autoresponse_kinetics(w, cfg, tgt_rows[ar_ok[w.trial_index[tgt_rows]]])
    manifest_sha = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    result: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "dataset": directory.name,
        "dataset_manifest_sha256": manifest_sha,
        "config": cfg,
        "code_revision": _git_revision(root),
        "window": w.meta,
        "counts": counts,
        "quality": quality,
        "pairs": pairs,
        "targets": targets,
        "stimulus": stimulus,
        "drift": drift,
        "indicator": {**INDICATOR, "autoresponse_kinetics": kinetics},
        "id_intersection": intersection,
        "split_selection": split_selection,
        "go_no_go": go,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pairs_tbl, out_dir / "pair_coverage.parquet")
    pq.write_table(pa.Table.from_pylist(target_rows), out_dir / "target_eligibility.parquet")
    pq.write_table(pa.Table.from_pylist(coverage_rows), out_dir / "fold_coverage.parquet")
    tables = {}
    for name in ("pair_coverage", "target_eligibility", "fold_coverage"):
        p = out_dir / f"{name}.parquet"
        tables[name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": pq.read_metadata(p).num_rows}
    result["tables"] = tables
    (out_dir / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=False, default=_json_default) + "\n")
    from occamworm.analysis.audit_report import render

    (out_dir / "report.md").write_text(render(result, target_rows))
    return result


def _json_default(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))


def _id_intersection(root: Path, atlas_ids: set[str]) -> dict[str, Any]:
    """Atlas labels resolved through the OW-013 alias audit, against anatomy and molecular annotations."""
    ann = root / "data" / "normalized" / "annotations-v1"
    needed = ("edges.parquet", "neurons.parquet", "unresolved.parquet")
    if not all((ann / f).exists() for f in needed):
        return {"status": "pending OW-013", "atlas_ids": len(atlas_ids)}
    edges = pq.read_table(ann / "edges.parquet", columns=["source_neuron_id", "target_neuron_id", "edge_kind"])
    kind = np.asarray(edges.column("edge_kind").to_pylist(), dtype=object)
    syn = (kind == "chem") | (kind == "gap")
    anat = {str(x) for x in np.asarray(edges.column("source_neuron_id").to_pylist(), dtype=object)[syn]}
    anat |= {str(x) for x in np.asarray(edges.column("target_neuron_id").to_pylist(), dtype=object)[syn]}
    neurons = pq.read_table(ann / "neurons.parquet", columns=["neuron_id", "has_transmitter_assignment"]).to_pylist()
    molecular = {str(r["neuron_id"]) for r in neurons if r["has_transmitter_assignment"]}
    audit = pq.read_table(ann / "unresolved.parquet", columns=["label", "resolved", "resolution", "neuron_id"])
    resolution: dict[str, Any] = {}
    atlas: set[str] = set()
    for r in audit.to_pylist():
        if r["label"] in atlas_ids:
            resolution[r["resolution"]] = resolution.get(r["resolution"], 0) + 1
            if r["resolved"] and r["neuron_id"]:
                atlas.add(str(r["neuron_id"]))
    return {
        "status": "computed",
        "atlas_labels": len(atlas_ids),
        "atlas_label_resolution": dict(sorted(resolution.items())),
        "atlas_ids": len(atlas),
        "anatomy_ids": len(anat),
        "molecular_ids": len(molecular),
        "molecular_definition": "neurons with a neurotransmitter assignment (OW-013 neurons.parquet)",
        "atlas_and_anatomy": len(atlas & anat),
        "atlas_and_molecular": len(atlas & molecular),
        "all_three": len(atlas & anat & molecular),
        "atlas_not_in_anatomy": sorted(atlas - anat),
    }
