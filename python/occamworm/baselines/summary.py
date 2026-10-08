"""Summaries of the baseline benchmark (§4.3, §11.5): paired animal-level comparisons from ``folds.jsonl``.

Every number in ``summary.json`` and ``report.md`` is computed here from the fold log; T2a and T2b are reported
separately and never averaged. The *selected baseline* of an outer fold is the (family, history) pair with the
best inner-validation score in that fold, i.e. a nested choice that never looks at the held-out animal.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from occamworm.analysis.uncertainty import paired_difference
from occamworm.baselines.benchmark import load_experiment

CONTRASTS = [
    ("B0", "B1", "stimulus response, shared kernel per target"),
    ("B0", "B3", "stimulus response, independent kernels"),
    ("B3", "B1", "H1: shared per-target kernel vs independent kernels"),
    ("B3", "B1d", "H1 with per-pair delays"),
    ("B3", "B2-K1", "one global kernel vs independent kernels"),
    ("B3", "B2-K2", "two global kernels vs independent kernels"),
    ("B3", "B2-K4", "four global kernels vs independent kernels"),
    ("B3", "B2-K8", "eight global kernels vs independent kernels"),
    ("B0", "B4-fixed", "anatomical network (fixed) vs null"),
    ("B0", "B4-learned", "anatomical network (learned) vs null"),
    ("B3", "B4-learned", "anatomical network (learned) vs independent kernels"),
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _per_animal(rows: list[dict[str, Any]], key: str = "animal_nll") -> dict[str, float]:
    out: dict[str, float] = {}
    for r in rows:
        for a, v in r[key].items():
            if a in out:
                raise ValueError(f"animal {a} scored twice")
            out[a] = float(v)
    return out


def summarize(root: Path, out: Path) -> dict[str, Any]:
    exp = load_experiment(root)
    rep = exp["report"]
    rows = [r for r in load_rows(out / "folds.jsonl") if "family" in r]
    by: dict[tuple[str, bool, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["task"], r["history"], r["family"])].append(r)

    tables: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    for task in ("T2a", "T2b"):
        fams = sorted({f for (t, _, f) in by if t == task})
        if not fams:
            continue
        common: set[str] | None = None
        for key, rs in by.items():
            if key[0] == task:
                animals = set(_per_animal(rs))
                common = animals if common is None else common & animals
        common = common or set()
        table = []
        for hist in (False, True):
            for fam in fams:
                rs = by.get((task, hist, fam), [])
                if not rs:
                    continue
                nll = _per_animal(rs)
                samples = _per_animal(rs, "animal_samples")
                traces = _per_animal(rs, "animal_traces")
                tot = sum(nll[a] for a in common)
                n_s = sum(samples[a] for a in common)
                table.append(
                    {
                        "family": fam,
                        "history": hist,
                        "n_animals": len(common),
                        "nll_total": tot,
                        "nll_per_sample": tot / max(n_s, 1),
                        "nll_per_trace": tot / max(sum(traces[a] for a in common), 1),
                        "mse": float(np.mean([r["mse"] for r in rs])),
                        "coverage": {
                            lv: float(np.mean([r["coverage"][lv] for r in rs if r["coverage"]]))
                            for lv in ("0.50", "0.80", "0.95")
                        },
                        "lam_rel_mode": max({r["lam_rel"] for r in rs}, key=[r["lam_rel"] for r in rs].count),
                        "n_params_median": float(np.median([r["n_params"] for r in rs])),
                        "seconds": float(sum(r["seconds"] for r in rs)),
                    }
                )
        tables[task] = table

        res = []
        for hist in (False, True):
            for base, cand, what in CONTRASTS:
                rb, rc = by.get((task, hist, base)), by.get((task, hist, cand))
                if not rb or not rc:
                    continue
                b = {a: v for a, v in _per_animal(rb).items() if a in common}
                c = {a: v for a, v in _per_animal(rc).items() if a in common}
                d = paired_difference(b, c, reps=rep["bootstrap_replicates"], level=rep["confidence_level"])
                res.append({"baseline": base, "candidate": cand, "history": hist, "question": what, **d})
            for fam in fams:
                r0, r1 = by.get((task, False, fam)), by.get((task, True, fam))
                if r0 and r1 and not hist:
                    b = {a: v for a, v in _per_animal(r0).items() if a in common}
                    c = {a: v for a, v in _per_animal(r1).items() if a in common}
                    d = paired_difference(b, c, reps=rep["bootstrap_replicates"], level=rep["confidence_level"])
                    res.append(
                        {"baseline": fam, "candidate": fam, "history": "no -> yes", "question": "history term", **d}
                    )
        contrasts[task] = res

        folds: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for (t, _, _), rs in by.items():
            if t == task:
                for r in rs:
                    folds[r["fold"]].append(r)
        picks = []
        nll_sel: dict[str, float] = {}
        for _fold, rs in sorted(folds.items()):
            best = min(rs, key=lambda r: (r["inner_best"], r["family"], r["history"]))
            picks.append((best["family"], best["history"]))
            for a, v in best["animal_nll"].items():
                if a in common:
                    nll_sel[a] = v
        counts: dict[str, int] = defaultdict(int)
        for f, h in picks:
            counts[f"{f}{'+hist' if h else ''}"] += 1
        selected[task] = {"choices": dict(sorted(counts.items())), "nll_total": sum(nll_sel.values())}

    result = {
        "experiment": exp["experiment"]["name"],
        "primary_task": exp["experiment"]["primary_task"],
        "tables": tables,
        "contrasts": contrasts,
        "selected_baseline": selected,
        "folds_logged": len(rows),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "report.md").write_text(render(result))
    return result


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1e4 else f"{x:,.0f}"
    return str(x)


def render(r: dict[str, Any]) -> str:
    lines = [
        f"# Baseline benchmark: {r['experiment']}",
        "",
        "> Generated by `python -m occamworm.baselines summarize` from `folds.jsonl`. Do not edit by hand. "
        "Spec: §4, §11. Positive ΔNLL favours the candidate.",
        "",
    ]
    for task, table in r["tables"].items():
        primary = " (primary)" if task == r["primary_task"] else ""
        lines += [
            f"## {task}{primary}",
            "",
            "| family | history | animals | NLL | NLL/sample | MSE | cov50 | cov80 | cov95 | ridge | params |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in table:
            cv = row["coverage"]
            lines.append(
                f"| {row['family']} | {'yes' if row['history'] else 'no'} | {row['n_animals']} | "
                f"{_f(row['nll_total'])} | {_f(row['nll_per_sample'], 4)} | {_f(row['mse'])} | {_f(cv['0.50'])} | "
                f"{_f(cv['0.80'])} | {_f(cv['0.95'])} | {_f(row['lam_rel_mode'])} | {_f(row['n_params_median'])} |"
            )
        lines += [
            "",
            "| baseline → candidate | history | question | ΔNLL mean (95% CI) | median | animals | "
            "favouring | max animal share |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in r["contrasts"].get(task, []):
            if not c.get("n_animals"):
                continue
            lo, hi = c["mean_ci"]
            lines.append(
                f"| {c['baseline']} → {c['candidate']} | "
                f"{ {False: 'no', True: 'yes'}.get(c['history'], c['history']) } | {c['question']} | "
                f"{_f(c['mean'])} ({_f(lo)} to {_f(hi)}) | {_f(c['median'])} | {c['n_animals']} | "
                f"{_f(c['fraction_animals_favouring_candidate'])} | {_f(c['max_single_animal_share'])} |"
            )
        sel = r["selected_baseline"].get(task)
        if sel:
            lines += ["", f"Nested selection of the strongest baseline per outer fold: {sel['choices']}.", ""]
    return "\n".join(lines) + "\n"
