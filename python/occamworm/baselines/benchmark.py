"""Baseline benchmark on the frozen split (§4, §11; OW-006/OW-007).

For each task (T2a, T2b) and history variant, every outer fold of the frozen split is evaluated by
``evaluate_fold`` for every family. Folds run in worker processes; each worker rebuilds the task data once.
In T2b the indicator is re-estimated in every outer fold from that fold's training autoresponses, so the design
(and the statistics) are fold-specific. Results are appended to ``folds.jsonl`` one line per (config, fold,
family), so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
import tomllib
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from occamworm.analysis.dataset import PRE, dataset_dir, load_windows
from occamworm.analysis.provenance import file_sha256, git_revision, write_run_manifest
from occamworm.analysis.splits import Split, read_split
from occamworm.baselines.data import Stats, TaskData, build_task, compute_stats, trial_designs
from occamworm.baselines.evaluate import Fitter, FoldResult, evaluate_fold, make_fitter, rows_of
from occamworm.baselines.indicator import Indicator, estimate_from_autoresponses

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
KERNEL_FAMILIES = ("B0", "B1", "B1d", "B2-K1", "B2-K2", "B2-K4", "B2-K8", "B3")


@dataclass(frozen=True)
class Config:
    task: str
    history: bool

    @property
    def key(self) -> str:
        return f"{self.task}-{'hist' if self.history else 'nohist'}"


def load_experiment(root: Path, path: Path | None = None) -> dict[str, Any]:
    path = path or root / "configs" / "experiments" / "shared-timescales.toml"
    with path.open("rb") as fh:
        return tomllib.load(fh)


def eligible_targets(root: Path, exp: dict[str, Any]) -> set[str]:
    rows = pq.read_table(root / exp["data"]["eligibility"], columns=["target", "eligible"]).to_pylist()
    return {r["target"] for r in rows if r["eligible"]}


def fold_masks(split: Split, data: TaskData, i: int) -> tuple[BoolArray, BoolArray, list[tuple[BoolArray, BoolArray]]]:
    index = {a: k for k, a in enumerate(data.animals)}

    def mask(animals: Iterable[str]) -> BoolArray:
        m = np.zeros(len(data.animals), dtype=bool)
        m[[index[a] for a in animals if a in index]] = True
        return m

    f = split.folds[i]
    inner = [(mask(itr), mask(iv)) for itr, iv in f.inner]
    return mask(f.train), mask(f.test), [(itr, iv) for itr, iv in inner if iv.any() and itr.any()]


def fold_indicator(data: TaskData, train: BoolArray) -> Indicator:
    """Indicator from the outer-training autoresponses only (T2b design)."""
    ok = data.trial_auto_ok & train[data.trial_animal]
    traces = data.trial_auto[ok, PRE:]
    animals = [data.animals[a] for a in data.trial_animal[ok]]
    est = estimate_from_autoresponses(traces, np.ones_like(traces, dtype=bool), animals, data.dt, reps=0)
    ind: Indicator = est["indicator"]
    return ind


# --- worker state (one task configuration per worker process)
_STATE: dict[str, Any] = {}


def _init_worker(root: str, exp: dict[str, Any], cfg: Config) -> None:
    r = Path(root)
    w = load_windows(dataset_dir(r, exp["data"]["dataset"]))
    data = build_task(w, eligible_targets(r, exp), cfg.task, cfg.history)
    _STATE.clear()
    _STATE.update(root=r, exp=exp, cfg=cfg, data=data, split=read_split(r / exp["splits"]["split"]))
    if cfg.task == "T2a":
        designs = trial_designs(data)
        _STATE.update(designs=designs, stats=compute_stats(data, designs))


def run_inputs(root: Path, exp: dict[str, Any], families: tuple[str, ...]) -> dict[str, Any]:
    """Everything a benchmark result depends on (§14.7)."""
    ds = dataset_dir(root, exp["data"]["dataset"])
    split = json.loads((root / exp["splits"]["split"]).read_text())
    ann = root / "data" / "normalized" / "annotations-v1" / "manifest.json"
    return {
        "kind": "baseline_benchmark",
        "dataset_manifest_sha256": file_sha256(ds / "manifest.json"),
        "eligibility_sha256": file_sha256(root / exp["data"]["eligibility"]),
        "split_sha256": split["split_sha256"],
        "graph": {
            "annotations_manifest_sha256": file_sha256(ann) if ann.exists() else None,
            "reconstruction": "cook2019-herm",
        },
        "experiment_config_sha256": file_sha256(root / "configs" / "experiments" / "shared-timescales.toml"),
        "families": list(families),
        "code": git_revision(root),
        "seeds": {"split": split["spec"]["seed"], "bootstrap": exp["report"]["bootstrap_replicates"]},
    }


def fitter_for(family: str, data: TaskData, designs: FloatArray, train: BoolArray) -> Fitter:
    if family.startswith("B4"):
        from occamworm.baselines.linear_network import make_b4_fitter

        return make_b4_fitter(family, data, _STATE["root"])
    return make_fitter(data, family)


def _run_fold(i: int, families: tuple[str, ...]) -> list[dict[str, Any]]:
    data: TaskData = _STATE["data"]
    cfg: Config = _STATE["cfg"]
    train, test, inner = fold_masks(_STATE["split"], data, i)
    if not test.any():
        return [{"config": cfg.key, "fold": i, "skipped": "test animal has no eligible traces"}]
    extra: dict[str, Any] = {}
    if cfg.task == "T2b":
        ind = fold_indicator(data, train)
        designs = trial_designs(data, ind)
        stats: Stats = compute_stats(data, designs)
        extra["indicator"] = ind.to_json()
    else:
        designs, stats = _STATE["designs"], _STATE["stats"]
    scored = np.unique(data.trace_trial[rows_of(data, test)])
    extra["test_animals"] = [data.animals[a] for a in np.nonzero(test)[0]]
    extra["test_trials"] = [data.trial_ids[k] for k in scored]
    out = []
    for fam in families:
        t0 = time.time()
        r: FoldResult = evaluate_fold(
            data,
            stats,
            designs,
            train,
            test,
            inner,
            fam,
            fitter=fitter_for(fam, data, designs, train),
            lam_grid=tuple(_STATE["exp"]["models"]["ridge_grid"]),
        )
        out.append(
            {
                "config": cfg.key,
                "task": cfg.task,
                "history": cfg.history,
                "fold": i,
                "family": fam,
                "lam_rel": r.lam_rel,
                "inner_scores": {str(k): v for k, v in r.inner_scores.items()},
                "inner_best": min(r.inner_scores.values()),
                "noise": r.noise.to_json(),
                "animal_nll": r.animal_nll,
                "animal_traces": r.animal_traces,
                "animal_samples": r.animal_samples,
                "mse": r.mse,
                "coverage": r.coverage,
                "n_params": r.n_params,
                "seconds": round(time.time() - t0, 3),
                **extra,
            }
        )
    return out


def run(
    root: Path,
    out: Path,
    configs: list[Config],
    families: tuple[str, ...],
    workers: int,
    folds: list[int] | None = None,
) -> Path:
    exp = load_experiment(root)
    write_run_manifest(out, run_inputs(root, exp, families))
    log = out / "folds.jsonl"
    done: set[tuple[str, int, str]] = set()
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            if "family" in row:
                done.add((row["config"], row["fold"], row["family"]))
    split = read_split(root / exp["splits"]["split"])
    fold_ids = folds if folds is not None else list(range(len(split.folds)))
    ctx = mp.get_context("spawn")
    for cfg in configs:
        todo = [(i, tuple(f for f in families if (cfg.key, i, f) not in done)) for i in fold_ids]
        todo = [(i, fs) for i, fs in todo if fs]
        if not todo:
            continue
        print(f"{cfg.key}: {len(todo)} folds", file=sys.stderr, flush=True)
        with ProcessPoolExecutor(
            workers, mp_context=ctx, initializer=_init_worker, initargs=(str(root), exp, cfg)
        ) as pool:
            futures = [pool.submit(_run_fold, i, fs) for i, fs in todo]
            for n, fut in enumerate(as_completed(futures), 1):
                rows = fut.result()
                with log.open("a") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                if n % 10 == 0 or n == len(futures):
                    print(f"  {cfg.key}: {n}/{len(futures)}", file=sys.stderr, flush=True)
    return log
