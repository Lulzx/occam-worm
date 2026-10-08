"""Synthetic atlas-like data with known kernels (§4.5 step 4, §16 synthetic truth; OW-006).

Each animal has every target stimulated ``reps`` times; every responder of the target is observed with a random
mask. In T2a the input is an indicator-filtered pulse of random size (the autoresponse); responses are the input
convolved with the true pair kernel, plus Gaussian AR(1) noise with per-trace scale ``a * s_n``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from occamworm.analysis.dataset import POST, PRE
from occamworm.baselines.data import TaskData
from occamworm.baselines.indicator import Indicator, difference_of_exponentials

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticSpec:
    truth: str  # "shared" (one kernel per target) or "independent" (one per pair)
    n_animals: int = 16
    n_targets: int = 4
    n_responders: int = 6
    reps: int = 2
    dt: float = 0.5
    noise_a: float = 1.0
    noise_phi: float = 0.5
    s_range: tuple[float, float] = (0.01, 0.03)
    missing: float = 0.2
    indicator: Indicator = Indicator(0.4, 2.5)
    task: str = "T2a"
    seed: int = 0


def true_kernels(spec: SyntheticSpec, rng: np.random.Generator) -> FloatArray:
    """(targets, responders, POST) neural kernels sampled on lags 0..POST-1 (per volume of input)."""
    lags = np.arange(POST) * spec.dt
    k = np.zeros((spec.n_targets, spec.n_responders, POST))
    for j in range(spec.n_targets):
        tau_r, tau_d = rng.uniform(0.3, 1.0), rng.uniform(2.0, 8.0)
        shared = difference_of_exponentials(lags, tau_r, tau_r + tau_d)
        for i in range(spec.n_responders):
            amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.15)
            if spec.truth == "shared":
                k[j, i] = amp * shared
            elif spec.truth == "independent":
                r, d = rng.uniform(0.2, 3.0), rng.uniform(0.5, 12.0)
                k[j, i] = amp * difference_of_exponentials(lags, r, r + d)
            else:
                raise ValueError(spec.truth)
    return k


def _ar1(rng: np.random.Generator, n: int, phi: float) -> FloatArray:
    e = np.empty((n, POST))
    e[:, 0] = rng.standard_normal(n)
    for t in range(1, POST):
        e[:, t] = phi * e[:, t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(n)
    return e


def generate(spec: SyntheticSpec, kernels: FloatArray | None = None) -> tuple[TaskData, FloatArray]:
    rng = np.random.default_rng(spec.seed)
    k = true_kernels(spec, rng) if kernels is None else kernels
    animals = [f"syn:a{a:03d}" for a in range(spec.n_animals)]
    targets = [f"T{j}" for j in range(spec.n_targets)]
    pairs = [(t, f"R{j}_{i}") for j, t in enumerate(targets) for i in range(spec.n_responders)]
    ind = spec.indicator.kernel(PRE + POST, spec.dt)

    trial_animal, trial_target, inputs, z = [], [], [], []
    for a in range(spec.n_animals):
        for r in range(spec.reps):
            for j in range(spec.n_targets):
                trial_animal.append(a)
                trial_target.append(j)
                u = np.zeros(PRE + POST)
                if spec.task == "T2a":
                    u[PRE:] = rng.uniform(0.5, 1.5) * ind[:POST]
                else:
                    u[PRE] = 1.0
                inputs.append(u)
                z.append((r * spec.n_targets + j) / 50.0)
    u_all = np.array(inputs)
    n_trials = u_all.shape[0]

    tt, tp, ys = [], [], []
    for n in range(n_trials):
        j = trial_target[n]
        src = u_all[n] if spec.task == "T2a" else np.convolve(u_all[n], ind)[: PRE + POST]
        for i in range(spec.n_responders):
            resp = np.convolve(src, k[j, i])[PRE : PRE + POST]
            tt.append(n)
            tp.append(j * spec.n_responders + i)
            ys.append(resp)
    y_clean = np.array(ys)
    n_tr = y_clean.shape[0]
    s: FloatArray = np.asarray(rng.uniform(spec.s_range[0], spec.s_range[1], size=n_tr), dtype=np.float64)
    y = y_clean + spec.noise_a * s[:, None] * _ar1(rng, n_tr, spec.noise_phi)
    m = rng.random((n_tr, POST)) >= spec.missing
    data = TaskData(
        task=spec.task,
        history=False,
        dt=spec.dt,
        animals=animals,
        targets=targets,
        pairs=pairs,
        pair_target=np.repeat(np.arange(spec.n_targets), spec.n_responders).astype(np.int64),
        trial_ids=[f"syn:trial{n}" for n in range(n_trials)],
        trial_animal=np.array(trial_animal, dtype=np.int64),
        trial_target=np.array(trial_target, dtype=np.int64),
        trial_input=u_all if spec.task == "T2a" else np.zeros_like(u_all),
        trial_auto=u_all,
        trial_auto_ok=np.full(n_trials, spec.task == "T2a"),
        trial_z=np.array(z),
        trace_trial=np.array(tt, dtype=np.int64),
        trace_pair=np.array(tp, dtype=np.int64),
        y=y.astype(np.float32),
        m=m,
        s=s,
        pre_slope=np.zeros(n_tr),
        meta={"synthetic": spec.truth},
    )
    return data, k
