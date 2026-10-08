"""Deliberate leaks must raise (§11.8, OW-004 definition of done)."""

import pytest

from occamworm.analysis.splits import (
    Fold,
    LeakageError,
    Split,
    SplitSpec,
    build_split,
    check_disjoint_ids,
    check_partition,
    check_trials,
)

ANIMALS = [f"a{i}" for i in range(10)]
SPEC = SplitSpec("g5", "group_kfold", 5, 2, seed=11)


def _leak_animal(split: Split) -> Split:
    """Copy one test animal of fold 0 into its training set."""
    f0 = split.folds[0]
    leaked = Fold(f0.index, (*f0.train, f0.test[0]), f0.test, f0.inner)
    return Split(split.spec, split.animals, (leaked, *split.folds[1:]))


def test_leaked_animal_in_outer_training_raises() -> None:
    with pytest.raises(LeakageError, match="both train and test"):
        check_partition(_leak_animal(build_split(ANIMALS, SPEC)))


def test_animal_tested_twice_raises() -> None:
    split = build_split(ANIMALS, SPEC)
    f1 = split.folds[1]
    dup = Fold(f1.index, f1.train, (*f1.test, split.folds[0].test[0]), f1.inner)
    with pytest.raises(LeakageError):
        check_partition(Split(split.spec, split.animals, (split.folds[0], dup, *split.folds[2:])))


def test_outer_test_animal_in_inner_loop_raises() -> None:
    split = build_split(ANIMALS, SPEC)
    f0 = split.folds[0]
    itrain, ival = f0.inner[0]
    bad = Fold(f0.index, f0.train, f0.test, ((itrain, (*ival, f0.test[0])), *f0.inner[1:]))
    with pytest.raises(LeakageError):
        check_partition(Split(split.spec, split.animals, (bad, *split.folds[1:])))


def test_session_shared_across_animals_raises() -> None:
    split = build_split(ANIMALS, SPEC)
    a, b = split.folds[0].test[0], split.folds[1].test[0]
    with pytest.raises(LeakageError, match="session"):
        check_trials(split, {"t1": a, "t2": b}, {"t1": "s", "t2": "s"})


def test_unknown_animal_raises() -> None:
    split = build_split(ANIMALS, SPEC)
    with pytest.raises(LeakageError):
        check_trials(split, {"t1": "ghost"}, {"t1": "s"})


def test_duplicate_trial_across_train_and_test_raises() -> None:
    with pytest.raises(LeakageError):
        check_disjoint_ids(["t1", "t2"], ["t2", "t3"], "trials")
