"""Task data and sufficient statistics shared by the kernel baselines (§4.1, OW-006/OW-007).

A *trace* is one responder's post-stimulus window in one trial: ``y_n(t)``, t = 0..T-1 volumes from the
stimulation, with a validity mask. Every baseline predicts

    mu_n(t) = (X_n c_p)(t) + (Q_n beta)(t)

- ``X_n`` (T x M) is the stimulus design: the input convolved with a piecewise-linear kernel basis (tents on
  ``KERNEL_KNOTS``, lags 0..T-1). In T2a the input is the measured autoresponse of the stimulated neuron
  (gaps linearly interpolated: a documented transform of an input, never of a scored output). In T2b it is a
  unit impulse at the stimulation frame passed through the indicator, so ``c_p`` is a *neural* kernel.
- ``c_p`` (M) is the kernel of pair p = (target, responder); each model family constrains it differently.
- ``Q_n`` (T x H) holds shared nuisance columns: a drift/artifact curve (tents on ``DRIFT_KNOTS``) and, in history
  variants, the trace's own pre-stimulus slope and the stimulation index times coarse tents (carryover, §4.1).
  ``beta`` is shared by all traces.

Fitting only needs, per (animal, pair) entry, ``X'MX``, ``X'MQ`` and ``X'My``, and per animal ``Q'MQ``, ``Q'My``,
``y'My`` and the sample count; training sets are sums over their animals, so outer and inner folds never touch
held-out traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from occamworm.analysis.dataset import POST, PRE, Windows
from occamworm.baselines.indicator import Indicator

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

T = POST
KERNEL_KNOTS = np.array([0, 1, 2, 4, 6, 9, 13, 18, 25, 33, 43, 59])
DRIFT_KNOTS = np.array([0, 2, 5, 10, 20, 35, 59])
HISTORY_KNOTS = np.array([0, 10, 30, 59])
STIM_INDEX_SCALE = 50.0
MIN_AUTORESPONSE_VALID = 0.8


def tents(knots: IntArray, n: int) -> FloatArray:
    """Piecewise-linear interpolation basis: column m is 1 at knots[m], 0 at the other knots, flat-zero outside."""
    x = np.arange(n, dtype=np.float64)
    out = np.zeros((n, knots.size))
    for m in range(knots.size):
        e = np.zeros(knots.size)
        e[m] = 1.0
        out[:, m] = np.interp(x, knots, e, left=0.0, right=0.0)
    return out


KERNEL_BASIS = tents(KERNEL_KNOTS, T)  # (T, M): kernel(lag) = KERNEL_BASIS @ c
M = KERNEL_KNOTS.size


def conv_design(u: FloatArray, basis: FloatArray = KERNEL_BASIS) -> FloatArray:
    """Design for inputs ``u`` sampled at offsets -PRE..T-1: X[t, m] = sum_lag basis[lag, m] u[t - lag]."""
    u = np.atleast_2d(u)
    n_lag = basis.shape[0]
    out = np.zeros((u.shape[0], T, basis.shape[1]))
    for lag in range(n_lag):
        lo = PRE - lag  # u index of offset (0 - lag)
        src_start = max(lo, 0)
        dst_start = src_start - lo
        if dst_start >= T:
            break
        seg = u[:, src_start : src_start + T - dst_start]
        out[:, dst_start : dst_start + seg.shape[1], :] += seg[:, :, None] * basis[lag][None, None, :]
    return out


def impulse_design(indicator: Indicator | None, dt: float) -> FloatArray:
    """T2b design: unit impulse at offset 0, through the indicator if given (then ``c`` is a neural kernel)."""
    u = np.zeros((1, PRE + T))
    if indicator is None:
        u[0, PRE] = 1.0
    else:
        u[0, PRE:] = indicator.kernel(T, dt)
    return np.asarray(conv_design(u)[0], dtype=np.float64)


@dataclass
class TaskData:
    task: str  # "T2a" or "T2b"
    history: bool
    dt: float
    animals: list[str]
    targets: list[str]
    pairs: list[tuple[str, str]]
    pair_target: IntArray
    trial_ids: list[str]
    trial_animal: IntArray
    trial_target: IntArray
    trial_input: FloatArray  # (n_trials, PRE+T) interpolated autoresponse (T2a) or zeros (T2b)
    trial_auto: FloatArray  # (n_trials, PRE+T) interpolated autoresponse where available (indicator estimation)
    trial_auto_ok: BoolArray  # autoresponse tracked in >= MIN_AUTORESPONSE_VALID of the window
    trial_z: FloatArray  # stimulation index / STIM_INDEX_SCALE
    trace_trial: IntArray
    trace_pair: IntArray
    y: npt.NDArray[np.float32]
    m: BoolArray
    s: FloatArray  # pre-stimulus SD
    pre_slope: FloatArray  # dF/F per volume over the pre-stimulus window
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def trace_animal(self) -> IntArray:
        return self.trial_animal[self.trace_trial]

    @property
    def n_traces(self) -> int:
        return int(self.trace_trial.size)

    @property
    def H(self) -> int:
        return DRIFT_KNOTS.size + (2 * HISTORY_KNOTS.size if self.history else 0)

    def q_columns(self, rows: IntArray) -> FloatArray:
        """Nuisance design ``Q_n`` (len(rows), T, H)."""
        drift = tents(DRIFT_KNOTS, T)
        n = rows.size
        parts = [np.broadcast_to(drift, (n, T, drift.shape[1]))]
        if self.history:
            hist = tents(HISTORY_KNOTS, T)
            parts.append(self.pre_slope[rows, None, None] * hist[None])
            parts.append(self.trial_z[self.trace_trial[rows], None, None] * hist[None])
        return np.concatenate(parts, axis=2)


def build_task(
    w: Windows, eligible_targets: set[str], task: str, history: bool, animals_subset: set[str] | None = None
) -> TaskData:
    """Select trials and responder traces for a task (§3.6). ``animals_subset`` restricts animals (tests)."""
    if task not in ("T2a", "T2b"):
        raise ValueError(task)
    t = w.trials
    n_trials_all = len(t["trial_id"])
    target_nid = np.asarray(t["stim_target_neuron_id"], dtype=object)
    animal = np.asarray(t["animal_id"], dtype=object)
    ar_frac = np.asarray(t["autoresponse_valid_fraction"], dtype=np.float64)
    code = np.asarray(t["stim_target_code"], dtype=object)

    target_row = np.full(n_trials_all, -1, dtype=np.int64)
    tr = np.nonzero(w.is_target)[0]
    target_row[w.trial_index[tr]] = tr

    keep_trial = np.array([x is not None and str(x) in eligible_targets for x in target_nid])
    if animals_subset is not None:
        keep_trial &= np.array([str(a) in animals_subset for a in animal])
    if task == "T2a":
        ar_ok = np.nan_to_num(ar_frac, nan=0.0) >= MIN_AUTORESPONSE_VALID
        keep_trial &= (code == "roi") & (target_row >= 0) & ar_ok
    trials = np.nonzero(keep_trial)[0]
    trial_pos = np.full(n_trials_all, -1, dtype=np.int64)
    trial_pos[trials] = np.arange(trials.size)

    animals = sorted({str(animal[k]) for k in trials})
    a_index = {a: i for i, a in enumerate(animals)}
    targets = sorted({str(target_nid[k]) for k in trials})
    t_index = {x: i for i, x in enumerate(targets)}

    win_trial_pos = trial_pos[w.trial_index]
    resp = np.nonzero(
        (win_trial_pos >= 0)
        & ~w.is_target
        & np.array([n is not None for n in w.neuron_id])
        & w.valid[:, PRE:].any(axis=1)
    )[0]
    resp = resp[[str(w.neuron_id[r]) != str(target_nid[w.trial_index[r]]) for r in resp]]
    pair_keys = [(str(target_nid[w.trial_index[r]]), str(w.neuron_id[r])) for r in resp]
    pairs = sorted(set(pair_keys))
    p_index = {p: i for i, p in enumerate(pairs)}

    u = np.zeros((trials.size, PRE + T))
    auto_ok = np.zeros(trials.size, dtype=bool)
    grid = np.arange(PRE + T, dtype=np.float64)
    for i, k in enumerate(trials):
        r = target_row[k]
        if r < 0 or not np.isfinite(ar_frac[k]) or ar_frac[k] < MIN_AUTORESPONSE_VALID:
            continue
        v = w.valid[r]
        if v.any():
            u[i] = np.interp(grid, grid[v], w.dff[r][v].astype(np.float64))
            auto_ok[i] = True

    pre = w.dff[resp, :PRE].astype(np.float64)
    pv = w.valid[resp, :PRE]
    x = np.arange(PRE, dtype=np.float64)
    cnt = pv.sum(axis=1)
    xm = np.where(pv, x, 0).sum(axis=1) / np.maximum(cnt, 1)
    ym = np.where(pv, pre, 0).sum(axis=1) / np.maximum(cnt, 1)
    sxx = np.where(pv, (x - xm[:, None]) ** 2, 0).sum(axis=1)
    sxy = np.where(pv, (x - xm[:, None]) * (pre - ym[:, None]), 0).sum(axis=1)
    slope = np.where((cnt >= 3) & (sxx > 0), sxy / np.maximum(sxx, 1e-12), 0.0)

    stim_idx = np.asarray(t["stim_index_in_recording"], dtype=np.float64)
    return TaskData(
        task=task,
        history=history,
        dt=w.volume_interval_s,
        animals=animals,
        targets=targets,
        pairs=pairs,
        pair_target=np.array([t_index[p[0]] for p in pairs], dtype=np.int64),
        trial_ids=[str(t["trial_id"][k]) for k in trials],
        trial_animal=np.array([a_index[str(animal[k])] for k in trials], dtype=np.int64),
        trial_target=np.array([t_index[str(target_nid[k])] for k in trials], dtype=np.int64),
        trial_input=u if task == "T2a" else np.zeros_like(u),
        trial_auto=u,
        trial_auto_ok=auto_ok,
        trial_z=stim_idx[trials] / STIM_INDEX_SCALE,
        trace_trial=win_trial_pos[resp],
        trace_pair=np.array([p_index[p] for p in pair_keys], dtype=np.int64),
        y=w.dff[resp, PRE:],
        m=w.valid[resp, PRE:],
        s=np.nan_to_num(w.pre_sd[resp].astype(np.float64), nan=0.0),
        pre_slope=slope,
        meta={"window": w.meta, "eligible_targets": len(eligible_targets)},
    )


def trial_designs(data: TaskData, indicator: Indicator | None = None) -> FloatArray:
    """(n_trials, T, M) design per trial; in T2b every trial shares the impulse design."""
    if data.task == "T2a":
        return conv_design(data.trial_input)
    x = impulse_design(indicator, data.dt)
    return np.broadcast_to(x, (len(data.trial_ids), T, M))


@dataclass
class Stats:
    """Per-(animal, pair) and per-animal sufficient statistics."""

    entry_animal: IntArray
    entry_pair: IntArray
    G: FloatArray  # (E, M, M)
    XQ: FloatArray  # (E, M, H)
    Xy: FloatArray  # (E, M)
    QQ: FloatArray  # (A, H, H)
    Qy: FloatArray  # (A, H)
    yy: FloatArray  # (A,)
    n: FloatArray  # (A,) observed samples


def compute_stats(data: TaskData, designs: FloatArray) -> Stats:
    n_a, h = len(data.animals), data.H
    order = np.argsort(data.trace_trial, kind="stable")
    tt = data.trace_trial[order]
    bounds = np.searchsorted(tt, np.arange(len(data.trial_ids) + 1))
    key = data.trace_animal * len(data.pairs) + data.trace_pair
    uniq, entry_of = np.unique(key, return_inverse=True)
    e_n = uniq.size
    G = np.zeros((e_n, M, M))
    XQ = np.zeros((e_n, M, h))
    Xy = np.zeros((e_n, M))
    QQ = np.zeros((n_a, h, h))
    Qy = np.zeros((n_a, h))
    yy = np.zeros(n_a)
    nn = np.zeros(n_a)
    for k in range(len(data.trial_ids)):
        rows = order[bounds[k] : bounds[k + 1]]
        if rows.size == 0:
            continue
        x = designs[k]
        mm = data.m[rows].astype(np.float64)
        yv = np.where(data.m[rows], data.y[rows], 0.0).astype(np.float64)
        q = data.q_columns(rows)
        ent = entry_of[rows]
        np.add.at(G, ent, np.einsum("tm,rt,tk->rmk", x, mm, x, optimize=True))
        np.add.at(XQ, ent, np.einsum("tm,rt,rth->rmh", x, mm, q, optimize=True))
        np.add.at(Xy, ent, np.einsum("tm,rt->rm", x, yv))
        a = data.trial_animal[k]
        QQ[a] += np.einsum("rth,rt,rtk->hk", q, mm, q, optimize=True)
        Qy[a] += np.einsum("rth,rt->h", q, yv)
        yy[a] += float((yv**2).sum())
        nn[a] += float(mm.sum())
    return Stats(
        entry_animal=uniq // len(data.pairs),
        entry_pair=uniq % len(data.pairs),
        G=G,
        XQ=XQ,
        Xy=Xy,
        QQ=QQ,
        Qy=Qy,
        yy=yy,
        n=nn,
    )


@dataclass
class PairStats:
    G: FloatArray  # (P, M, M)
    XQ: FloatArray  # (P, M, H)
    Xy: FloatArray  # (P, M)
    QQ: FloatArray  # (H, H)
    Qy: FloatArray  # (H,)
    yy: float
    n: float
    present: BoolArray  # (P,) pair observed in these animals


def aggregate(stats: Stats, animals: BoolArray, n_pairs: int) -> PairStats:
    """Sum statistics over the selected animals (boolean mask over ``data.animals``)."""
    sel = animals[stats.entry_animal]
    p = stats.entry_pair[sel]
    G = np.zeros((n_pairs, M, M))
    XQ = np.zeros((n_pairs, M, stats.XQ.shape[2]))
    Xy = np.zeros((n_pairs, M))
    np.add.at(G, p, stats.G[sel])
    np.add.at(XQ, p, stats.XQ[sel])
    np.add.at(Xy, p, stats.Xy[sel])
    present = np.zeros(n_pairs, dtype=bool)
    present[p] = True
    return PairStats(
        G,
        XQ,
        Xy,
        stats.QQ[animals].sum(0),
        stats.Qy[animals].sum(0),
        float(stats.yy[animals].sum()),
        float(stats.n[animals].sum()),
        present,
    )


def predict(
    data: TaskData, designs: FloatArray, c: FloatArray, beta: FloatArray, rows: IntArray, chunk: int = 20_000
) -> FloatArray:
    """mu for the given traces: (len(rows), T)."""
    out = np.empty((rows.size, T))
    for start in range(0, rows.size, chunk):
        r = rows[start : start + chunk]
        x = designs[data.trace_trial[r]]
        q = data.q_columns(r)
        stim = np.einsum("ntm,nm->nt", x, c[data.trace_pair[r]])
        out[start : start + r.size] = stim + np.einsum("nth,h->nt", q, beta)
    return out
