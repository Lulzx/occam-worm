"""Frozen nested evaluation (§7.4, §7.8; OW-012).

Per outer fold: build the training view, run every candidate's inner selection inside it, rank by summed
inner-validation NLL (ties: shorter L_total), keep the (L_total, inner NLL) Pareto front, refit the winner on the
whole view, freeze the selection, and hand it to the locked evaluator, which scores the held-out animals once.
The search log records every attempted candidate and the budget (§7.8). ``search_fold`` has no parameter through
which held-out data could arrive.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from occamworm.analysis.splits import Split
from occamworm.baselines.data import TaskData
from occamworm.search.candidates import Candidate, InnerResult
from occamworm.search.locked import LockedEvaluator, freeze_selection
from occamworm.search.views import TrainingView, animals_hash, views_for_fold


def pareto_front(results: Sequence[InnerResult]) -> list[InnerResult]:
    """Candidates not dominated in (L_total bits, inner NLL); both smaller-is-better."""
    front = []
    for r in results:
        dominated = any(
            (o.l_total_bits <= r.l_total_bits and o.inner_nll <= r.inner_nll)
            and (o.l_total_bits < r.l_total_bits or o.inner_nll < r.inner_nll)
            for o in results
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: (r.l_total_bits, r.inner_nll))


def search_fold(
    view: TrainingView, candidates: Sequence[Candidate], max_candidates: int | None = None
) -> dict[str, Any]:
    t0 = time.time()
    budget = len(candidates) if max_candidates is None else min(max_candidates, len(candidates))
    results = [c.inner_select(view) for c in candidates[:budget]]
    ranked = sorted(results, key=lambda r: (r.inner_nll, r.l_total_bits, r.key))
    return {
        "ranked": ranked,
        "pareto": pareto_front(results),
        "log": {
            "candidates_offered": len(candidates),
            "candidates_attempted": budget,
            "inner_folds": len(view.inner),
            "seconds": time.time() - t0,
        },
    }


def run_fold(
    data: TaskData,
    split: Split,
    fold: int,
    candidates: Sequence[Candidate],
    out: Path,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    view, test = views_for_fold(data, split, fold)
    found = search_fold(view, candidates, max_candidates)
    winner: InnerResult = found["ranked"][0]
    by_key = {c.key: c for c in candidates}
    cand = by_key[winner.key]
    fitted = cand.fit_final(view, winner.ridge)
    sel_path = out / f"fold{fold:03d}" / "selection.json"
    freeze_selection(
        sel_path,
        {
            "fold": fold,
            "training_animals": list(view.data.animals),
            "training_animals_sha256": animals_hash(view.data.animals),
            "winner": {k: v for k, v in asdict(winner).items() if k != "val_animal_nll"},
            "candidate_kind": cand.kind,
            "fitted": fitted,
            "pareto": [r.key for r in found["pareto"]],
            "ranking": [
                {"key": r.key, "inner_nll": r.inner_nll, "l_total_bits": r.l_total_bits} for r in found["ranked"]
            ],
            "search_log": found["log"],
        },
    )
    evaluator = LockedEvaluator(fold, test, out / "locks")
    scored = evaluator.score(sel_path, lambda sel, t: cand.score(sel["fitted"], t))
    (out / f"fold{fold:03d}" / "outer.json").write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")
    return {"winner": winner.key, **scored, "search_log": found["log"]}
