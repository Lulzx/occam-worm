"""OW-012 definition of done: no test labels are reachable from the search API; injected leaks fail."""

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from occamworm.analysis.splits import Fold, LeakageError, Split, SplitSpec, build_split
from occamworm.baselines.data import TaskData
from occamworm.baselines.synthetic import SyntheticSpec, generate
from occamworm.search.candidates import InnerResult, KernelCandidate
from occamworm.search.locked import LockedEvaluator, freeze_selection, read_selection
from occamworm.search.nested import pareto_front, run_fold, search_fold
from occamworm.search.views import TrainingView, animals_hash, views_for_fold

DATA, _ = generate(SyntheticSpec("shared", n_animals=8, seed=3))
SPLIT = build_split(DATA.animals, SplitSpec("g4", "group_kfold", 4, 2, seed=1))


@dataclass
class Spy:
    """A candidate that records everything the search hands it."""

    key: str = "spy"
    kind: str = "spy"
    l_struct_bits: float = 1.0
    seen: list[Any] | None = None

    def inner_select(self, view: TrainingView) -> InnerResult:
        assert self.seen is not None
        self.seen.append(view)
        return InnerResult(self.key, 0.0, 0.0, {}, 1.0, 0.0, 0, 0.0)

    def fit_final(self, view: TrainingView, ridge: float) -> dict[str, Any]:
        return {}

    def score(self, fitted: dict[str, Any], test: TaskData) -> dict[str, float]:
        return {a: 0.0 for a in test.animals}


def test_training_view_holds_no_held_out_data() -> None:
    view, test = views_for_fold(DATA, SPLIT, 0)
    assert not set(view.data.animals) & set(test.animals)
    assert not set(view.data.trial_ids) & set(test.trial_ids)
    for name in ("y", "m", "s", "trial_input", "trial_auto"):
        assert not np.shares_memory(getattr(view.data, name), getattr(DATA, name))
    held_out_pairs_only = set(test.pairs) - set(view.data.pairs)
    assert not held_out_pairs_only & set(view.data.pairs)


def test_search_api_never_receives_held_out_animals() -> None:
    spy = Spy(seen=[])
    view, test = views_for_fold(DATA, SPLIT, 1)
    search_fold(view, [spy])
    for v in spy.seen or []:
        assert not set(v.data.animals) & set(test.animals)
    assert list(inspect.signature(search_fold).parameters) == ["view", "candidates", "max_candidates"]


def test_injected_inner_leak_raises() -> None:
    f0 = SPLIT.folds[0]
    itr, iv = f0.inner[0]
    leaked = Fold(f0.index, f0.train, f0.test, ((itr, (*iv, f0.test[0])), *f0.inner[1:]))
    with pytest.raises(LeakageError):
        views_for_fold(DATA, Split(SPLIT.spec, SPLIT.animals, (leaked, *SPLIT.folds[1:])), 0)


def selection(tmp: Path, fold: int, animals: list[str]) -> Path:
    p = tmp / f"sel{fold}-{len(animals)}.json"
    freeze_selection(p, {"fold": fold, "training_animals": animals, "training_animals_sha256": animals_hash(animals)})
    return p


def test_locked_evaluator_refuses_tampering_leaks_and_rescoring(tmp_path: Path) -> None:
    view, test = views_for_fold(DATA, SPLIT, 2)
    ev = LockedEvaluator(2, test, tmp_path / "locks")

    def scorer(sel: dict[str, Any], t: TaskData) -> dict[str, float]:
        return {a: 0.0 for a in t.animals}

    leaky = selection(tmp_path, 2, [*view.data.animals, test.animals[0]])
    with pytest.raises(LeakageError, match="held-out"):
        ev.score(leaky, scorer)
    wrong_fold = selection(tmp_path, 3, list(view.data.animals))
    with pytest.raises(LeakageError, match="fold"):
        ev.score(wrong_fold, scorer)
    good = selection(tmp_path, 2, list(view.data.animals))
    tampered = tmp_path / "tampered.json"
    tampered.write_text(good.read_text().replace('"fold": 2', '"fold": 2, "extra": 1'))
    with pytest.raises(LeakageError, match="modified"):
        read_selection(tampered)
    assert ev.score(good, scorer)["fold"] == 2
    with pytest.raises(LeakageError, match="once"):
        ev.score(good, scorer)
    with pytest.raises(LeakageError):
        freeze_selection(good, {"fold": 2})  # frozen selections are never rewritten


def test_nested_run_scores_each_fold_once_with_a_kernel_candidate(tmp_path: Path) -> None:
    cands = [KernelCandidate("B0", 4.0), KernelCandidate("B1", 12.0), KernelCandidate("B3", 20.0)]
    res = run_fold(DATA, SPLIT, 0, cands, tmp_path)
    assert res["winner"] in {c.key for c in cands}
    assert set(res["animal_nll"]) == set(SPLIT.folds[0].test)
    assert res["search_log"]["candidates_attempted"] == 3
    with pytest.raises(LeakageError):
        run_fold(DATA, SPLIT, 0, cands, tmp_path)  # second outer scoring of the same fold


def test_pareto_front() -> None:
    r = [
        InnerResult(k, 0, nll, {}, bits, 0, 0, 0)
        for k, nll, bits in [("a", 10, 5), ("b", 8, 9), ("c", 11, 9), ("d", 7, 20)]
    ]
    assert [x.key for x in pareto_front(r)] == ["a", "b", "d"]
