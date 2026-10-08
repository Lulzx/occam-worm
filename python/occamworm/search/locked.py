"""Frozen selections and the locked outer evaluator (§7.4, §11.8; OW-012).

A selection is frozen as JSON with its own sha256 and the hash of the training animals it was chosen on. The
``LockedEvaluator`` owns the held-out data; it scores a selection once, after checking that the file is intact,
that its training animals are disjoint from the held-out animals, and that this fold has not been scored before
(a lock file records the first scoring). Nothing else in the search package receives held-out data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from occamworm.analysis.splits import LeakageError
from occamworm.baselines.data import TaskData
from occamworm.search.views import animals_hash


def _digest(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def freeze_selection(path: Path, body: dict[str, Any]) -> str:
    if path.exists():
        raise LeakageError(f"{path} exists; a frozen selection is never rewritten")
    sealed = {**body, "selection_sha256": _digest(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return str(sealed["selection_sha256"])


def read_selection(path: Path) -> dict[str, Any]:
    sealed = json.loads(path.read_text())
    body = {k: v for k, v in sealed.items() if k != "selection_sha256"}
    if _digest(body) != sealed.get("selection_sha256"):
        raise LeakageError(f"{path} was modified after it was frozen")
    return dict(sealed)


class LockedEvaluator:
    def __init__(self, fold: int, test: TaskData, lock_dir: Path) -> None:
        self.fold = fold
        self._test = test
        self._lock = lock_dir / f"fold{fold:03d}.lock"

    @property
    def test_animals_sha256(self) -> str:
        return animals_hash(self._test.animals)

    def score(
        self, selection_path: Path, scorer: Callable[[dict[str, Any], TaskData], dict[str, float]]
    ) -> dict[str, Any]:
        sel = read_selection(selection_path)
        if sel["fold"] != self.fold:
            raise LeakageError(f"selection is for fold {sel['fold']}, evaluator holds fold {self.fold}")
        if set(sel["training_animals"]) & set(self._test.animals):
            raise LeakageError("the selection was trained on a held-out animal")
        if animals_hash(sel["training_animals"]) != sel["training_animals_sha256"]:
            raise LeakageError("training animal list does not match its hash")
        if self._lock.exists():
            raise LeakageError(f"fold {self.fold} was already scored ({self._lock}); outer folds are scored once")
        self._lock.parent.mkdir(parents=True, exist_ok=True)
        self._lock.write_text(json.dumps({"selection_sha256": sel["selection_sha256"]}) + "\n")
        per_animal = scorer(sel, self._test)
        return {"fold": self.fold, "selection_sha256": sel["selection_sha256"], "animal_nll": per_animal}
