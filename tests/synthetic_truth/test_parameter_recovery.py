"""OW-011 definition of done: controlled synthetic parameters recover within known statistical uncertainty.

A known G1 rule on a small graph is simulated at known parameters, noise is added (independent Gaussian, or AR(1)),
some samples are masked, and the parameters are fitted by multi-start L-BFGS-B on the JAX simulator. The standard
error comes from the inverse Hessian of the NLL (cross-checked against a parametric bootstrap), and recovery is
``|estimate - truth| < 3 SE`` for every fitted parameter. The optimisation budget must be logged in the manifest.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from recovery_scenarios import ADAPT_RULE, ADAPT_TRUTH, GAP_RULE, GAP_TRUTH, Scenario, build_scenario

from occamworm.fit.losses import AR1Loss, GaussianLoss, ObservationLoss
from occamworm.fit.manifest import build_manifest, write_manifest
from occamworm.fit.multistart import FitBudget, FitResult, fit_multistart
from occamworm.fit.problem import FitProblem
from occamworm.fit.uncertainty import HessianSE, hessian_standard_errors, parametric_bootstrap
from occamworm.sim.ir import Program

Compile = Callable[[str], Program]
SIGMA = 0.05
BUDGET = FitBudget(maxiter=300, maxfun=1500)


def recover(problem: FitProblem, truth: dict[str, float], n_starts: int = 5) -> tuple[FitResult, HessianSE, np.ndarray]:
    fit = fit_multistart(problem, n_starts=n_starts, seed=1, budget=BUDGET)
    se = hessian_standard_errors(problem, fit.theta_free)
    z = np.asarray([(est - truth[n]) / s for n, est, s in zip(problem.free_names, fit.theta_free, se.se, strict=True)])
    return fit, se, z


def assert_recovered(
    problem: FitProblem, truth: dict[str, float], fit: FitResult, se: HessianSE, z: np.ndarray
) -> None:
    assert se.positive_definite and not any(se.at_bound), se
    assert np.all(np.isfinite(se.se))
    for name, est, s, zi in zip(problem.free_names, fit.theta_free, se.se, z, strict=True):
        assert abs(est - truth[name]) < 3.0 * s, (
            f"{name}: estimate {est:.5f}, truth {truth[name]}, SE {s:.5f}, z {zi:+.2f}"
        )
        assert s < 0.1 * truth[name], f"{name}: SE {s:.4f} is not informative against {truth[name]}"


@pytest.fixture(scope="module")
def adapt_scenario(compile_wrl: Compile) -> Scenario:
    return build_scenario(compile_wrl(ADAPT_RULE), ADAPT_TRUTH)


@pytest.fixture(scope="module")
def gap_scenario(compile_wrl: Compile) -> Scenario:
    return build_scenario(compile_wrl(GAP_RULE), GAP_TRUTH, seed=3)


def test_adaptation_circuit_recovery_gaussian(adapt_scenario: Scenario, tmp_path: Path) -> None:
    problem = adapt_scenario.problem(GaussianLoss(SIGMA), noise_seed=7)
    fit, se, z = recover(problem, ADAPT_TRUTH)
    assert_recovered(problem, ADAPT_TRUTH, fit, se, z)

    boot = parametric_bootstrap(problem, fit, n_rep=16, seed=2)
    ratio = boot.se / se.se
    assert np.all((ratio > 0.5) & (ratio < 2.0)), (boot.se, se.se)  # two independent estimates of the same SE

    manifest = build_manifest(fit, hessian=se, bootstrap=boot, label="adaptation circuit, Gaussian noise")
    path = write_manifest(tmp_path / "adapt.json", manifest)
    logged = json.loads(path.read_text())
    used = logged["budget"]["used"]
    assert used["fun_grad_evals"] > 0 and used["iterations"] > 0 and used["wall_seconds"] > 0
    assert used["simulated_trial_steps"] == used["fun_grad_evals"] * 6 * 100
    assert logged["budget"]["requested"]["n_starts"] == 5 and logged["seed"] == 1
    assert logged["program_hash"] == problem.program.program_hash
    assert logged["description_length"]["l_params_bits"] == 32  # 12 + 10 + 10 declared bits
    for entry in logged["parameters"]:
        if entry["fitted"]:
            assert abs(entry["estimate"] - ADAPT_TRUTH[entry["name"]]) < 3.0 * entry["std_error_hessian"]


def test_gap_circuit_recovery_ar1_noise(gap_scenario: Scenario) -> None:
    loss = AR1Loss(SIGMA, 0.6)
    problem = gap_scenario.problem(loss, noise_seed=11)
    fit, se, z = recover(problem, GAP_TRUTH)
    assert_recovered(problem, GAP_TRUTH, fit, se, z)
    boot = parametric_bootstrap(problem, fit, n_rep=16, seed=5)
    ratio = boot.se / se.se
    assert np.all((ratio > 0.5) & (ratio < 2.0)), (boot.se, se.se)
    # positively correlated noise carries less information per sample: at the same estimate and data, the white-noise
    # Hessian gives smaller standard errors than the AR(1) Hessian
    white = FitProblem(
        gap_scenario.program,
        gap_scenario.simulator,
        gap_scenario.stimulus,
        problem.target,
        GaussianLoss(SIGMA),
        mask=gap_scenario.mask,
    )
    assert np.all(se.se > hessian_standard_errors(white, fit.theta_free).se)


def test_profiled_noise_level_gives_the_same_recovery(adapt_scenario: Scenario) -> None:
    noisy = adapt_scenario.problem(GaussianLoss(SIGMA), noise_seed=7)
    profiled = FitProblem(
        adapt_scenario.program,
        adapt_scenario.simulator,
        adapt_scenario.stimulus,
        noisy.target,
        GaussianLoss(None),
        mask=adapt_scenario.mask,
    )
    fit, se, z = recover(profiled, ADAPT_TRUTH, n_starts=3)
    assert_recovered(profiled, ADAPT_TRUTH, fit, se, z)


def test_standard_errors_are_calibrated_across_replicate_datasets(adapt_scenario: Scenario) -> None:
    """Over replicate noise draws the z-scores behave like standard normals (SE neither too small nor too large)."""
    loss: ObservationLoss = GaussianLoss(SIGMA)
    base = adapt_scenario.problem(loss, noise_seed=100)
    zs = []
    for rep in range(16):
        rng = np.random.default_rng(1000 + rep)
        problem = base.with_target(adapt_scenario.clean + loss.sample_noise(rng, adapt_scenario.mask))
        fit = fit_multistart(problem, n_starts=2, seed=rep, budget=BUDGET)
        se = hessian_standard_errors(problem, fit.theta_free)
        zs.append([(e - ADAPT_TRUTH[n]) / s for n, e, s in zip(problem.free_names, fit.theta_free, se.se, strict=True)])
    z = np.asarray(zs)
    assert np.mean(np.abs(z) < 3.0) >= 0.95
    assert 0.4 < np.mean(z**2) < 2.0, np.mean(z**2)  # chi-square(1) mean is 1


def test_multistart_reaches_the_best_optimum_from_scattered_starts(gap_scenario: Scenario) -> None:
    problem = gap_scenario.problem(GaussianLoss(SIGMA), noise_seed=13)
    fit = fit_multistart(problem, n_starts=8, seed=21, budget=BUDGET)
    nlls = np.asarray([s.nll for s in fit.starts])
    assert fit.best_start.nll == nlls.min()
    # the declared start and most scattered starts reach the same optimum; the others are logged, not hidden
    assert np.sum(nlls < nlls.min() + 1e-6) >= 5
    assert abs(fit.best_start.nll - problem.nll_at(fit.theta_free)) < 1e-6
    # the optimum is a stationary point of the objective in the transformed coordinates
    raw = problem.transforms.inverse(fit.theta_free)
    _, grad = problem.value_and_grad_raw(raw)
    assert np.max(np.abs(grad)) < 1e-2


def test_budget_cap_is_respected_and_reported(adapt_scenario: Scenario) -> None:
    problem = adapt_scenario.problem(GaussianLoss(SIGMA), noise_seed=7)
    fit = fit_multistart(problem, n_starts=4, seed=3, budget=FitBudget(maxiter=4, maxfun=6))
    report = build_manifest(fit)["budget"]
    assert report["requested"]["maxiter_per_start"] == 4 and report["requested"]["maxfun_per_start"] == 6
    assert all(s.n_iterations <= 4 for s in fit.starts)
    assert report["used"]["fun_grad_evals"] == sum(s.n_fun_grad_evals for s in fit.starts)
    assert not any(s.success for s in fit.starts) or fit.best_start.n_iterations <= 4
