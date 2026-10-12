"""Gaussian vs Student-t AR(1) observation model on inner folds only (open decision 3, OW-006).

For a few outer folds of the frozen split, each family is refitted at its inner-selected ridge on every inner
training set, both noise models are fitted on that inner training set's residuals, and the inner-validation animals
are scored. Outer test animals are never loaded into a fit or a score (the same leakage check as
``evaluate_fold``). Diagnostics on where the heavy tails sit use inner-validation residuals only.

    python -m occamworm.baselines.noise_evidence --folds 0 1 2 --families B0 B1 B3 --workers 2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from occamworm.analysis.dataset import PRE, dataset_dir, load_windows
from occamworm.analysis.scoring import interval_coverage
from occamworm.analysis.splits import read_split
from occamworm.baselines.benchmark import eligible_targets, fold_masks, load_experiment
from occamworm.baselines.data import TaskData, aggregate, build_task, compute_stats, trial_designs
from occamworm.baselines.evaluate import INNER_NOISE_TRACES, make_fitter, noise_for, rows_of, score_rows
from occamworm.sources.cli import find_root

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
TIME_BINS = ((2, 4), (4, 6), (6, 10), (10, 20), (20, 40), (40, 60))
SEG = 9  # samples of a fully observed segment used for the autocorrelation comparison (pre-stimulus has 9)
_STATE: dict[str, Any] = {}


def _init(root: str, exp: dict[str, Any]) -> None:
    r = Path(root)
    w = load_windows(dataset_dir(r, exp["data"]["dataset"]))
    data = build_task(w, eligible_targets(r, exp), "T2a", False)
    designs = trial_designs(data, None)
    _STATE.update(
        root=r,
        exp=exp,
        w=w,
        data=data,
        designs=designs,
        stats=compute_stats(data, designs),
        split=read_split(r / exp["splits"]["split"]),
    )


def segment_autocorr(x: FloatArray, valid: BoolArray, lo: int, hi: int) -> tuple[float, float, int]:
    """Pooled lag-1 autocorrelation (sum of products / sum of squares) of demeaned fully observed segments."""
    seg, ok = x[:, lo:hi], valid[:, lo:hi].all(axis=1)
    seg = seg[ok]
    seg = seg - seg.mean(axis=1, keepdims=True)
    den = float((seg**2).sum())
    return float((seg[:, 1:] * seg[:, :-1]).sum()), den, int(ok.sum())


def tail_diagnostics(z: FloatArray, valid: BoolArray, animals: list[str], targets: list[int]) -> dict[str, Any]:
    """Where |z| > 3 happens (z = residual / Gaussian sigma), pooled over validation traces."""
    big = (np.abs(z) > 3) & valid
    out: dict[str, Any] = {"n_samples": int(valid.sum()), "n_big": int(big.sum())}
    out["by_time"] = {f"{lo}-{hi - 1}": [int(big[:, lo:hi].sum()), int(valid[:, lo:hi].sum())] for lo, hi in TIME_BINS}
    per_trace = big.sum(axis=1)
    order = np.sort(per_trace)[::-1]
    tot = max(int(per_trace.sum()), 1)
    out["share_in_top_1pct_traces"] = float(order[: max(len(order) // 100, 1)].sum() / tot)
    out["share_in_top_10pct_traces"] = float(order[: max(len(order) // 10, 1)].sum() / tot)
    out["frac_traces_with_any"] = float((per_trace > 0).mean())
    by_animal: dict[str, list[int]] = {}
    for a, nb, nv in zip(animals, per_trace, valid.sum(axis=1), strict=True):
        c = by_animal.setdefault(a, [0, 0])
        c[0] += int(nb)
        c[1] += int(nv)
    out["by_animal"] = by_animal
    by_target: dict[int, list[int]] = {}
    for t, nb, nv in zip(targets, per_trace, valid.sum(axis=1), strict=True):
        c = by_target.setdefault(int(t), [0, 0])
        c[0] += int(nb)
        c[1] += int(nv)
    out["by_target"] = {str(k): v for k, v in by_target.items()}
    zv = z[valid]
    out["excess_kurtosis"] = float(np.mean(zv**4) / np.mean(zv**2) ** 2 - 3.0)
    return out


def _outer_fold(i: int, families: tuple[str, ...], lams: dict[str, float]) -> dict[str, Any]:
    data: TaskData = _STATE["data"]
    designs, stats, w = _STATE["designs"], _STATE["stats"], _STATE["w"]
    train, test, inner = fold_masks(_STATE["split"], data, i)
    n_pairs = len(data.pairs)
    animal_of_trace = np.asarray(data.animals, dtype=object)[data.trace_animal]
    result: dict[str, Any] = {"fold": i, "families": {}}
    # pre-stimulus autocorrelation of raw dF/F in the inner-validation animals of this fold (never the test animal)
    win_animal = np.asarray(w.trials["animal_id"], dtype=object)[w.trial_index]
    for fam in families:
        fitter = make_fitter(data, fam)
        acc: dict[str, Any] = {
            k: 0.0
            for k in ("nll_g", "nll_t", "n", "cov_g_50", "cov_g_80", "cov_g_95", "cov_t_50", "cov_t_80", "cov_t_95")
        }
        per_inner: list[dict[str, float]] = []
        animal_diff: dict[str, float] = {}
        z_all, v_all, e_all, an_all, tg_all = [], [], [], [], []
        for itrain, ival in inner:
            if np.any(itrain & ival) or np.any((itrain | ival) & test):
                raise RuntimeError("inner folds leak the outer test animals")
            ps = aggregate(stats, itrain, n_pairs)
            fit = fitter(ps, lams[fam], None)
            trows = rows_of(data, itrain)
            vrows = rows_of(data, ival)
            ng = noise_for(data, designs, fit, trows, INNER_NOISE_TRACES)
            nt = noise_for(data, designs, fit, trows, INNER_NOISE_TRACES, family="student_t_ar1")
            nll_g, e, v = score_rows(data, designs, fit, ng, vrows)
            nll_t, _, _ = score_rows(data, designs, fit, nt, vrows)
            sg = ng.sigma(data.s[vrows])
            st = nt.sigma(data.s[vrows])
            n = int(v.sum())
            cg = interval_coverage(e, v, sg)
            ct = interval_coverage(e, v, st, nu=nt.nu)
            per_inner.append(
                {
                    "nll_g": float(nll_g.sum()),
                    "nll_t": float(nll_t.sum()),
                    "n": n,
                    "nu": float(nt.nu or 0.0),
                    "phi_g": ng.phi,
                    "phi_t": nt.phi,
                    "a_g": ng.a,
                    "b_g": ng.b,
                    "a_t": nt.a,
                    "b_t": nt.b,
                    **{f"cov_g_{k[2:]}": c for k, c in cg.items()},
                    **{f"cov_t_{k[2:]}": c for k, c in ct.items()},
                }
            )
            acc["nll_g"] += float(nll_g.sum())
            acc["nll_t"] += float(nll_t.sum())
            acc["n"] += n
            for lv in ("50", "80", "95"):
                acc[f"cov_g_{lv}"] += cg[f"0.{lv}"] * n
                acc[f"cov_t_{lv}"] += ct[f"0.{lv}"] * n
            an = animal_of_trace[vrows]
            for a in np.unique(an):
                m = an == a
                animal_diff[str(a)] = float((nll_t[m] - nll_g[m]).sum())
            z_all.append(e / sg[:, None])
            v_all.append(v)
            e_all.append(e)
            an_all.extend(an.tolist())
            tg_all.extend(data.trial_target[data.trace_trial[vrows]].tolist())
        for k in list(acc):
            if k.startswith("cov_"):
                acc[k] /= acc["n"]
        z, v, e = np.concatenate(z_all), np.concatenate(v_all), np.concatenate(e_all)
        ac = {
            name: segment_autocorr(e, v, lo, lo + SEG)
            for name, lo in (("early_2_10", 2), ("mid_20_28", 20), ("late_40_48", 40))
        }
        val_animals = {a for a in an_all}
        pre_rows = np.nonzero(~w.is_target & np.isin(win_animal, sorted(val_animals)))[0]
        pre_x = w.dff[pre_rows, : PRE - 1].astype(np.float64)
        pre_ok = w.valid[pre_rows, : PRE - 1]
        ac["pre_-10_-2"] = segment_autocorr(pre_x, pre_ok, 0, SEG)
        result["families"][fam] = {
            "lam": lams[fam],
            "total": acc,
            "inner": per_inner,
            "animal_dnll": animal_diff,
            "tails": tail_diagnostics(z, v, [str(a) for a in an_all], tg_all),
            "autocorr": {k: {"num": a, "den": b, "n_segments": c} for k, (a, b, c) in ac.items()},
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--families", nargs="+", default=["B0", "B1", "B3"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--benchmark", type=Path, default=None, help="folds.jsonl whose inner-selected ridge to reuse")
    ap.add_argument("--out", type=Path, default=Path("artifacts/noise-evidence-v1/results.json"))
    args = ap.parse_args()
    root = find_root(Path.cwd())
    exp = load_experiment(root)
    bench = args.benchmark or root / "artifacts" / "baselines-v1" / "folds.jsonl"
    lams: dict[int, dict[str, float]] = {}
    for line in bench.read_text().splitlines():
        r = json.loads(line)
        if r.get("config") == "T2a-nohist" and r.get("family") in args.families and r["fold"] in args.folds:
            lams.setdefault(r["fold"], {})[r["family"]] = float(r["lam_rel"])
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init, initargs=(str(root), exp)) as pool:
        futs = [pool.submit(_outer_fold, i, tuple(args.families), lams[i]) for i in args.folds]
        results = [f.result() for f in futs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
