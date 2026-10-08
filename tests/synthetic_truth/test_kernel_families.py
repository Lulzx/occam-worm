"""OW-006 definition of done: synthetic data favour the correctly specified family on held-out animals."""

import numpy as np
import pytest

from occamworm.baselines.data import aggregate, compute_stats, conv_design, trial_designs
from occamworm.baselines.evaluate import evaluate_fold, grouped_inner
from occamworm.baselines.kernels import fit_independent, fit_lowrank, fit_shared, loss
from occamworm.baselines.synthetic import SyntheticSpec, generate


def heldout_nll(spec: SyntheticSpec, families: list[str]) -> dict[str, float]:
    data, _ = generate(spec)
    designs = trial_designs(data)
    stats = compute_stats(data, designs)
    n = len(data.animals)
    totals = {f: 0.0 for f in families}
    for fold in range(4):
        test = np.zeros(n, dtype=bool)
        test[fold::4] = True
        inner = grouped_inner(~test, 3, lambda a: data.animals[a])
        for f in families:
            r = evaluate_fold(data, stats, designs, ~test, test, inner, f, lam_grid=(0.01, 0.1, 1.0))
            totals[f] += sum(r.animal_nll.values())
    return totals


def test_shared_truth_favours_shared_kernel() -> None:
    s = heldout_nll(SyntheticSpec("shared", seed=1), ["B0", "B1", "B3"])
    assert s["B1"] < s["B3"] < s["B0"]


def test_independent_truth_favours_independent_kernels() -> None:
    s = heldout_nll(SyntheticSpec("independent", seed=2, s_range=(0.005, 0.01)), ["B1", "B3"])
    assert s["B3"] < s["B1"]


def test_conv_design_matches_numpy_convolution() -> None:
    rng = np.random.default_rng(0)
    u = rng.standard_normal((1, 70))
    c = rng.standard_normal(12)
    from occamworm.baselines.data import KERNEL_BASIS

    kernel = KERNEL_BASIS @ c
    expected = np.convolve(u[0], kernel)[10:70]
    np.testing.assert_allclose(conv_design(u)[0] @ c, expected, atol=1e-12)


def test_als_families_do_not_beat_unconstrained_training_loss() -> None:
    data, _ = generate(SyntheticSpec("independent", seed=3, n_animals=6))
    designs = trial_designs(data)
    ps = aggregate(compute_stats(data, designs), np.ones(len(data.animals), bool), len(data.pairs))
    b3 = fit_independent(ps, 0.1)
    b1 = fit_shared(ps, data.pair_target, len(data.targets), 0.1)
    b1d = fit_shared(ps, data.pair_target, len(data.targets), 0.1, delays=True)
    k8 = fit_lowrank(ps, 8, 0.1)
    lam = b3.extra["lam"]
    assert loss(ps, b3.c, b3.beta, lam) <= b1d.loss + 1e-9 <= b1.loss + 1e-9
    assert loss(ps, b3.c, b3.beta, lam) <= k8.loss + 1e-9
    assert b1.loss == pytest.approx(loss(ps, b1.c, b1.beta, lam))
