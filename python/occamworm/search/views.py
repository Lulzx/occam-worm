"""Training-only views of task data (§7.4, OW-012).

``subset_task`` copies the trials and traces of the given animals into a new ``TaskData`` with its own animal,
target and pair lists, so nothing about other animals survives: not their traces, not their pair coverage, not
their identities. Every array is a fresh copy (no memory shared with the parent), which the leakage tests check.
The search API receives only ``TrainingView`` objects.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from occamworm.analysis.splits import LeakageError, Split
from occamworm.baselines.data import TaskData

BoolArray = npt.NDArray[np.bool_]


def subset_task(data: TaskData, animals: Sequence[str]) -> TaskData:
    keep = set(animals)
    unknown = keep - set(data.animals)
    if unknown:
        raise LeakageError(f"animals not in the task data: {sorted(unknown)[:3]}")
    a_keep = np.array([a in keep for a in data.animals])
    trials = np.nonzero(a_keep[data.trial_animal])[0]
    traces = np.nonzero(a_keep[data.trace_animal])[0]
    new_animals = sorted(keep)
    a_index = {a: i for i, a in enumerate(new_animals)}
    trial_pos = np.full(len(data.trial_ids), -1, dtype=np.int64)
    trial_pos[trials] = np.arange(trials.size)
    used_pairs = sorted({data.pairs[p] for p in np.unique(data.trace_pair[traces])})
    p_index = {p: i for i, p in enumerate(used_pairs)}
    targets = sorted({data.targets[t] for t in np.unique(data.trial_target[trials])} | {p[0] for p in used_pairs})
    t_index = {t: i for i, t in enumerate(targets)}
    return TaskData(
        task=data.task,
        history=data.history,
        dt=data.dt,
        animals=new_animals,
        targets=targets,
        pairs=used_pairs,
        pair_target=np.array([t_index[p[0]] for p in used_pairs], dtype=np.int64),
        trial_ids=[data.trial_ids[k] for k in trials],
        trial_animal=np.array([a_index[data.animals[data.trial_animal[k]]] for k in trials], dtype=np.int64),
        trial_target=np.array([t_index[data.targets[data.trial_target[k]]] for k in trials], dtype=np.int64),
        trial_input=data.trial_input[trials].copy(),
        trial_auto=data.trial_auto[trials].copy(),
        trial_auto_ok=data.trial_auto_ok[trials].copy(),
        trial_z=data.trial_z[trials].copy(),
        trace_trial=trial_pos[data.trace_trial[traces]],
        trace_pair=np.array([p_index[data.pairs[p]] for p in data.trace_pair[traces]], dtype=np.int64),
        y=data.y[traces].copy(),
        m=data.m[traces].copy(),
        s=data.s[traces].copy(),
        pre_slope=data.pre_slope[traces].copy(),
        meta={**data.meta, "subset_of": len(data.animals), "subset_animals": len(new_animals)},
    )


def animals_hash(animals: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(animals)).encode()).hexdigest()


@dataclass
class TrainingView:
    """Outer-training data of one fold, with inner folds as masks over the view's own animals."""

    fold: int
    data: TaskData
    inner: list[tuple[BoolArray, BoolArray]]

    @property
    def animals_sha256(self) -> str:
        return animals_hash(self.data.animals)


def views_for_fold(data: TaskData, split: Split, fold: int) -> tuple[TrainingView, TaskData]:
    """(training view, held-out data). Held-out data is returned for the locked evaluator only."""
    f = split.folds[fold]
    have = set(data.animals)
    train = [a for a in f.train if a in have]
    test = [a for a in f.test if a in have]
    if set(train) & set(test):
        raise LeakageError(f"fold {fold}: animals in both training and test")
    tv = subset_task(data, train)
    idx = {a: i for i, a in enumerate(tv.animals)}
    inner = []
    for itr, iv in f.inner:
        if set(iv) & set(test) or set(itr) & set(test):
            raise LeakageError(f"fold {fold}: outer test animal inside the inner loop")
        vm = np.zeros(len(tv.animals), dtype=bool)
        vm[[idx[a] for a in iv if a in idx]] = True
        if vm.any() and (~vm).any():
            inner.append((~vm, vm))
    return TrainingView(fold, tv, inner), subset_task(data, test)
