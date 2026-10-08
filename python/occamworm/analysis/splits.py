"""Animal-wise outer/inner splits (§11.1) with leakage validation (§11.8).

Fold assignment depends only on the animal ids and the seed: animals are ordered by
``sha256(f"{seed}:{scope}:{animal_id}")`` and dealt round-robin into folds. No library RNG is involved, so a
split is reproducible across numpy versions and platforms. Every session and trial of an animal follows the
animal (an animal is the minimum independence unit).

A frozen split is written once as ``data/splits/<name>/split.json`` and is never overwritten with different
content.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1.0"


class LeakageError(RuntimeError):
    """Raised when a split lets information cross from test to training data."""


@dataclass(frozen=True)
class SplitSpec:
    name: str
    outer: str  # "group_kfold" or "leave_one_animal_out"
    outer_folds: int | None  # None for leave-one-animal-out
    inner_folds: int
    seed: int

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "outer": self.outer,
            "outer_folds": self.outer_folds,
            "inner": "group_kfold",
            "inner_folds": self.inner_folds,
            "seed": self.seed,
            "unit": "animal_id",
        }


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[str, ...]
    test: tuple[str, ...]
    inner: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]  # (inner_train, inner_val) pairs


@dataclass(frozen=True)
class Split:
    spec: SplitSpec
    animals: tuple[str, ...]
    folds: tuple[Fold, ...]

    def outer_fold_of(self) -> dict[str, int]:
        return {a: f.index for f in self.folds for a in f.test}


def _rank_key(seed: int, scope: str, animal: str) -> str:
    return hashlib.sha256(f"{seed}:{scope}:{animal}".encode()).hexdigest()


def group_kfold(animals: Iterable[str], k: int, seed: int, scope: str = "outer") -> list[tuple[str, ...]]:
    """Deal animals into ``k`` folds in hash order; returns the test animals of each fold, each sorted."""
    unique = sorted(set(animals))
    if k < 2:
        raise ValueError("need at least 2 folds")
    if len(unique) < k:
        raise ValueError(f"{len(unique)} animals cannot fill {k} folds")
    ordered = sorted(unique, key=lambda a: _rank_key(seed, scope, a))
    return [tuple(sorted(ordered[i::k])) for i in range(k)]


def build_split(animals: Sequence[str], spec: SplitSpec) -> Split:
    unique = tuple(sorted(set(animals)))
    if spec.outer == "group_kfold":
        if spec.outer_folds is None:
            raise ValueError("group_kfold needs outer_folds")
        tests = group_kfold(unique, spec.outer_folds, spec.seed, "outer")
    elif spec.outer == "leave_one_animal_out":
        tests = [(a,) for a in unique]
    else:
        raise ValueError(f"unknown outer scheme {spec.outer!r}")
    folds = []
    for i, test in enumerate(tests):
        held = set(test)
        train = tuple(a for a in unique if a not in held)
        inner_tests = group_kfold(train, spec.inner_folds, spec.seed, f"inner{i}")
        inner = tuple((tuple(a for a in train if a not in set(v)), v) for v in inner_tests)
        folds.append(Fold(index=i, train=train, test=test, inner=inner))
    split = Split(spec=spec, animals=unique, folds=tuple(folds))
    check_partition(split)
    return split


def check_partition(split: Split) -> None:
    """Outer tests partition the animals; every train/test pair (outer and inner) is disjoint."""
    seen: dict[str, int] = {}
    for f in split.folds:
        for a in f.test:
            if a in seen:
                raise LeakageError(f"animal {a} is in outer test folds {seen[a]} and {f.index}")
            seen[a] = f.index
        if set(f.train) & set(f.test):
            both = sorted(set(f.train) & set(f.test))
            raise LeakageError(f"outer fold {f.index}: animals in both train and test: {both}")
        for j, (itrain, ival) in enumerate(f.inner):
            if set(itrain) & set(ival):
                raise LeakageError(f"outer fold {f.index} inner {j}: animals in both inner train and validation")
            if (set(itrain) | set(ival)) != set(f.train):
                raise LeakageError(f"outer fold {f.index} inner {j}: inner folds do not cover the outer training set")
            if set(ival) & set(f.test) or set(itrain) & set(f.test):
                raise LeakageError(f"outer fold {f.index} inner {j}: outer test animal inside the inner loop")
    if set(seen) != set(split.animals):
        raise LeakageError("outer test folds do not cover every animal")


def check_trials(split: Split, trial_animal: Mapping[str, str], trial_session: Mapping[str, str]) -> None:
    """Sessions and trials stay with their animal; trial ids are unique; every trial's animal is in the split.

    ``trial_animal`` and ``trial_session`` map trial id to animal id and session id.
    """
    fold_of = split.outer_fold_of()
    session_fold: dict[str, int] = {}
    session_animal: dict[str, str] = {}
    for trial, animal in trial_animal.items():
        if animal not in fold_of:
            raise LeakageError(f"trial {trial} belongs to animal {animal}, which is not in the split")
        session = trial_session[trial]
        fold = fold_of[animal]
        if session_animal.setdefault(session, animal) != animal:
            raise LeakageError(f"session {session} is shared by animals {session_animal[session]} and {animal}")
        if session_fold.setdefault(session, fold) != fold:
            raise LeakageError(f"session {session} spans outer folds {session_fold[session]} and {fold}")


def check_disjoint_ids(train_ids: Iterable[str], test_ids: Iterable[str], what: str) -> None:
    """Generic leakage assertion: no id may appear on both sides (§11.8)."""
    both = set(train_ids) & set(test_ids)
    if both:
        raise LeakageError(f"{len(both)} {what} appear in both training and test data, e.g. {sorted(both)[:3]}")


def split_to_json(split: Split, dataset_manifest_sha256: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "spec": split.spec.to_json(),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "animals": list(split.animals),
        "folds": [
            {
                "index": f.index,
                "test": list(f.test),
                "inner_validation": [list(v) for _, v in f.inner],
            }
            for f in split.folds
        ],
    }
    body["split_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def split_from_json(body: Mapping[str, Any]) -> Split:
    expected = body["split_sha256"]
    check = {k: v for k, v in body.items() if k != "split_sha256"}
    if hashlib.sha256(json.dumps(check, sort_keys=True).encode()).hexdigest() != expected:
        raise LeakageError("split file content does not match its split_sha256; frozen splits are immutable")
    s = body["spec"]
    spec = SplitSpec(s["name"], s["outer"], s["outer_folds"], s["inner_folds"], s["seed"])
    animals = tuple(body["animals"])
    folds = []
    for f in body["folds"]:
        test = tuple(f["test"])
        train = tuple(a for a in animals if a not in set(test))
        inner = tuple((tuple(a for a in train if a not in set(v)), tuple(v)) for v in f["inner_validation"])
        folds.append(Fold(f["index"], train, test, inner))
    split = Split(spec, animals, tuple(folds))
    check_partition(split)
    return split


def write_split(split: Split, dataset_manifest_sha256: str, out_dir: Path) -> Path:
    """Freeze a split. Rewriting identical content is a no-op; different content is refused."""
    body = split_to_json(split, dataset_manifest_sha256)
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    path = out_dir / "split.json"
    if path.exists():
        if path.read_text() != text:
            raise LeakageError(f"{path} exists with different content; frozen splits are never overwritten")
        return path
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return path


def read_split(path: Path) -> Split:
    return split_from_json(json.loads(path.read_text()))


def target_coverage(split: Split, target_animals: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    """For each target: outer test folds containing it, and per such fold how many inner folds can both fit and
    validate it (the target present in inner training *and* inner validation animals)."""
    rows = []
    for target in sorted(target_animals):
        animals = target_animals[target]
        test_folds = [f for f in split.folds if animals & set(f.test)]
        inner_ok = [sum(1 for itr, iv in f.inner if animals & set(itr) and animals & set(iv)) for f in test_folds]
        rows.append(
            {
                "target": target,
                "n_animals": len(animals),
                "outer_test_folds": len(test_folds),
                "test_animals": sum(len(animals & set(f.test)) for f in test_folds),
                "min_inner_folds_fit_and_validate": min(inner_ok) if inner_ok else 0,
                "mean_inner_folds_fit_and_validate": (sum(inner_ok) / len(inner_ok)) if inner_ok else 0.0,
                "covered": bool(test_folds) and min(inner_ok) >= 1,
            }
        )
    return rows
