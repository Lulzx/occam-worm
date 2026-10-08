import json
from pathlib import Path

import pytest

from occamworm.analysis.splits import (
    LeakageError,
    SplitSpec,
    build_split,
    group_kfold,
    read_split,
    target_coverage,
    write_split,
)

ANIMALS = [f"a{i:02d}" for i in range(23)]


def test_group_kfold_is_a_balanced_partition() -> None:
    folds = group_kfold(ANIMALS, 5, seed=1)
    flat = [a for f in folds for a in f]
    assert sorted(flat) == sorted(ANIMALS)
    assert {len(f) for f in folds} == {4, 5}


def test_seed_stable_and_order_independent() -> None:
    spec = SplitSpec("g5", "group_kfold", 5, 3, seed=20261008)
    a = build_split(ANIMALS, spec)
    b = build_split(list(reversed(ANIMALS)), spec)
    assert a == b
    c = build_split(ANIMALS, SplitSpec("g5", "group_kfold", 5, 3, seed=7))
    assert [f.test for f in a.folds] != [f.test for f in c.folds]


def test_inner_folds_stay_inside_outer_training() -> None:
    split = build_split(ANIMALS, SplitSpec("g5", "group_kfold", 5, 3, seed=3))
    for f in split.folds:
        for itrain, ival in f.inner:
            assert not set(ival) & set(f.test)
            assert set(itrain) | set(ival) == set(f.train)


def test_loao_has_one_fold_per_animal() -> None:
    split = build_split(ANIMALS, SplitSpec("loao", "leave_one_animal_out", None, 5, seed=3))
    assert [f.test for f in split.folds] == [(a,) for a in sorted(ANIMALS)]


def test_frozen_split_round_trips_and_refuses_overwrite(tmp_path: Path) -> None:
    split = build_split(ANIMALS, SplitSpec("g5", "group_kfold", 5, 3, seed=3))
    path = write_split(split, "d" * 64, tmp_path)
    assert read_split(path) == split
    write_split(split, "d" * 64, tmp_path)  # identical content: no-op
    other = build_split(ANIMALS, SplitSpec("g5", "group_kfold", 5, 3, seed=4))
    with pytest.raises(LeakageError):
        write_split(other, "d" * 64, tmp_path)


def test_edited_split_file_is_rejected(tmp_path: Path) -> None:
    split = build_split(ANIMALS, SplitSpec("g5", "group_kfold", 5, 3, seed=3))
    path = write_split(split, "d" * 64, tmp_path)
    body = json.loads(path.read_text())
    body["folds"][0]["test"], body["folds"][1]["test"] = body["folds"][1]["test"], body["folds"][0]["test"]
    path.write_text(json.dumps(body))
    with pytest.raises(LeakageError):
        read_split(path)


def test_target_coverage_counts_inner_fit_and_validate() -> None:
    split = build_split(ANIMALS, SplitSpec("loao", "leave_one_animal_out", None, 2, seed=3))
    rows = target_coverage(split, {"ONE": {"a00"}, "TWO": {"a00", "a01"}, "MANY": set(ANIMALS[:6])})
    by = {r["target"]: r for r in rows}
    assert not by["ONE"]["covered"]  # held out, but never in training
    assert not by["TWO"]["covered"]  # training has one animal: cannot both fit and validate
    assert by["MANY"]["covered"] and by["MANY"]["outer_test_folds"] == 6
