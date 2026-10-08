"""OW-011: bounded transforms, losses, accounting, gradients, multi-start determinism and the manifest."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from occamworm.analysis.scoring import ar1_nll
from occamworm.fit.accounting import (
    check_against_ir,
    description_length,
    grid_spacing,
    l_params_bits,
    quantise,
    quantise_vector,
)
from occamworm.fit.losses import AR1Loss, GaussianLoss
from occamworm.fit.manifest import MANIFEST_SCHEMA, build_manifest, write_manifest
from occamworm.fit.multistart import FitBudget, fit_multistart, start_points
from occamworm.fit.problem import FitProblem
from occamworm.fit.transforms import ParamTransform, ParamTransforms
from occamworm.fit.uncertainty import hessian_standard_errors
from occamworm.sim.ir import Program

sys.path.insert(0, str(Path(__file__).parents[2] / "synthetic_truth"))
from recovery_scenarios import ADAPT_RULE, ADAPT_TRUTH, build_scenario  # noqa: E402

Compile = Callable[[str], Program]


# -- transforms ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sigmoid", "softplus"])
def test_transform_roundtrip_monotone_and_inside_bounds(kind: str) -> None:
    t = ParamTransform("x", 0.05, 4.0, kind)  # type: ignore[arg-type]
    raw = jnp.linspace(-12.0, 12.0, 49)
    theta = np.asarray(t.forward(raw))
    assert np.all(np.diff(theta) > 0)
    assert theta.min() >= 0.05
    if kind == "sigmoid":
        assert theta.max() <= 4.0
    inside = np.linspace(0.06, 3.9, 40)
    back = np.asarray([float(t.forward(jnp.asarray(t.inverse(float(v))))) for v in inside])
    np.testing.assert_allclose(back, inside, rtol=1e-12)
    assert np.isfinite(t.inverse(0.05)) and np.isfinite(t.inverse(4.0))  # bounds map to finite raw values


def test_softplus_box_enforces_a_finite_upper_bound() -> None:
    t = ParamTransform("x", 1.0, 3.0, "softplus")
    lo, hi = t.raw_bounds()
    assert float(t.forward(jnp.asarray(hi))) == pytest.approx(3.0, rel=1e-12)
    assert float(t.forward(jnp.asarray(lo))) == pytest.approx(1.0, abs=1e-6)
    open_ended = ParamTransform("y", 0.0, float("inf"), "softplus")
    assert np.isfinite(open_ended.inverse(50.0)) and open_ended.span() > 0
    with pytest.raises(ValueError):
        ParamTransform("z", 0.0, float("inf"), "sigmoid")
    with pytest.raises(ValueError):
        ParamTransform("w", 2.0, 1.0, "sigmoid")


def test_transform_gradient_is_the_logistic_density() -> None:
    t = ParamTransform("x", 0.5, 2.5, "sigmoid")
    g = jax.grad(lambda r: t.forward(r))(0.3)
    s = 1.0 / (1.0 + np.exp(-0.3))
    assert float(g) == pytest.approx(2.0 * s * (1 - s), rel=1e-12)


def test_transforms_from_program_use_ir_bounds(compile_wrl: Compile) -> None:
    program = compile_wrl(ADAPT_RULE)
    tr = ParamTransforms.from_program(program)
    assert set(tr.names) == {"tau_v", "gain", "adapt"}  # fixed parameters are never fitted
    by_name = {t.name: t for t in tr.transforms}
    assert (by_name["tau_v"].lower, by_name["tau_v"].upper) == (0.05, 2.0)
    with pytest.raises(ValueError, match="fixed"):
        ParamTransforms.from_program(program, ["tau_h"])
    with pytest.raises(ValueError, match="unknown"):
        ParamTransforms.from_program(program, ["nope"])
    theta = np.asarray([program.parameters[program.parameter_index(n) or 0].value for n in tr.names])
    np.testing.assert_allclose(np.asarray(tr.forward(jnp.asarray(tr.inverse(theta)))), theta, rtol=1e-12)


# -- losses -------------------------------------------------------------------------------------------------------


def test_gaussian_loss_matches_closed_forms() -> None:
    rng = np.random.default_rng(0)
    pred, target = rng.normal(size=(2, 7, 3)), rng.normal(size=(2, 7, 3))
    mask = rng.random(pred.shape) > 0.3
    r = (target - pred)[mask]
    n = r.size
    known = GaussianLoss(0.7).bind(mask)(jnp.asarray(pred), jnp.asarray(target))
    expected = 0.5 * np.sum(r**2) / 0.49 + n * (np.log(0.7) + 0.5 * np.log(2 * np.pi))
    assert float(known) == pytest.approx(expected, rel=1e-12)
    profile = GaussianLoss(None).bind(mask)(jnp.asarray(pred), jnp.asarray(target))
    sigma_hat = np.sqrt(np.mean(r**2))
    assert float(profile) == pytest.approx(
        GaussianLoss(float(sigma_hat)).bind(mask)(jnp.asarray(pred), jnp.asarray(target)), rel=1e-12
    )
    # masked entries never matter, even if non-finite
    bad = target.copy()
    bad[~mask] = np.nan
    assert float(
        GaussianLoss(0.7).bind(mask)(jnp.asarray(pred), jnp.asarray(np.where(mask, bad, 0.0)))
    ) == pytest.approx(float(known), rel=1e-12)


def test_ar1_loss_matches_numpy_scoring_and_has_exact_gradient() -> None:
    rng = np.random.default_rng(1)
    b, s, k = 3, 25, 2
    pred, target = rng.normal(size=(b, s, k)), rng.normal(size=(b, s, k))
    mask = rng.random(pred.shape) > 0.35
    mask[0, :, 0] = False  # an entirely unobserved trace
    sigma = rng.uniform(0.5, 1.5, size=(b, k))
    phi = 0.65
    loss = AR1Loss(sigma, phi).bind(mask)
    got = float(loss(jnp.asarray(pred), jnp.asarray(target)))
    resid = (target - pred).transpose(0, 2, 1).reshape(b * k, s)
    valid = mask.transpose(0, 2, 1).reshape(b * k, s)
    expected = float(np.sum(ar1_nll(resid, valid, sigma.reshape(b * k), phi)))
    assert got == pytest.approx(expected, rel=1e-12)
    g = np.asarray(jax.grad(loss)(jnp.asarray(pred), jnp.asarray(target)))
    fd = np.zeros_like(pred)
    for idx in [(0, 3, 1), (1, 10, 0), (2, 24, 1), (1, 0, 1)]:
        d = np.zeros_like(pred)
        d[idx] = 1e-6
        fd[idx] = (
            float(loss(jnp.asarray(pred + d), jnp.asarray(target)))
            - float(loss(jnp.asarray(pred - d), jnp.asarray(target)))
        ) / 2e-6
        assert g[idx] == pytest.approx(fd[idx], rel=1e-6, abs=1e-8)
    assert np.all(g[0, :, 0] == 0.0)


def test_ar1_noise_has_the_requested_marginal_sd_and_correlation() -> None:
    mask = np.ones((200, 60, 2), dtype=bool)
    noise = AR1Loss(0.3, 0.7).sample_noise(np.random.default_rng(0), mask)
    assert noise.std() == pytest.approx(0.3, rel=0.03)
    lag1 = np.mean(noise[:, 1:, :] * noise[:, :-1, :]) / noise.var()
    assert lag1 == pytest.approx(0.7, abs=0.03)
    with pytest.raises(ValueError):
        AR1Loss(1.0, 1.0)


# -- accounting ---------------------------------------------------------------------------------------------------


def test_l_params_reproduces_the_ir_and_the_bit_code(compile_wrl: Compile) -> None:
    program = compile_wrl(ADAPT_RULE)
    assert check_against_ir(program)
    assert l_params_bits(program) == 12 + 10 + 10 == program.l_params_bits
    assert l_params_bits(program, ["gain"]) == 10
    d = description_length(program)
    assert d["l_struct_bits"] == program.l_struct_bits and d["l_total_bits"] == program.l_struct_bits + 32
    assert d["bit_code_version"] == 1
    assert set(description_length(program, ["gain"])["unfitted_trainable"]) == {"tau_v", "adapt"}
    with pytest.raises(ValueError, match="fixed"):
        l_params_bits(program, ["tau_h"])


def test_quantisation_grid(compile_wrl: Compile) -> None:
    program = compile_wrl(ADAPT_RULE)
    gain = program.parameters[program.parameter_index("gain") or 0]
    assert gain.bits == 10 and grid_spacing(gain) == pytest.approx(4.0 / 1023)
    q = quantise(gain, 1.6)
    assert abs(q - 1.6) <= grid_spacing(gain) / 2 + 1e-15
    assert quantise(gain, -5.0) == gain.lower and quantise(gain, 99.0) == gain.upper
    assert quantise(gain, q) == pytest.approx(q, abs=1e-14)
    theta = program.default_theta()
    snapped = quantise_vector(program, theta)
    fixed = [p.index for p in program.parameters if not p.trainable]
    assert all(snapped[i] == theta[i] for i in fixed)


# -- objective gradients, multi-start, manifest ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def problem(compile_wrl: Compile) -> FitProblem:
    sc = build_scenario(compile_wrl(ADAPT_RULE), ADAPT_TRUTH, n_trials=3, n_steps=60)
    return sc.problem(GaussianLoss(0.05), noise_seed=5)


def test_transformed_gradient_matches_finite_differences(problem: FitProblem) -> None:
    for raw in (None, np.asarray([-0.4, 0.9, 1.1])):
        err, g, fd = problem.check_gradient(raw)
        assert np.all(np.isfinite(g)) and np.abs(g).max() > 1.0
        assert err < 1e-6, (err, g, fd)


def test_start_points_are_deterministic_and_inside_bounds(problem: FitProblem) -> None:
    a, b, c = start_points(problem, 6, 11), start_points(problem, 6, 11), start_points(problem, 6, 12)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[1:], c[1:])
    np.testing.assert_array_equal(a[0], problem.theta_free(problem.theta_base))  # row 0 is the declared start
    assert np.all(a >= problem.transforms.lower()) and np.all(a <= problem.transforms.upper())


def test_multistart_is_reproducible_and_logs_its_budget(problem: FitProblem, tmp_path: Path) -> None:
    budget = FitBudget(maxiter=100, maxfun=60)
    r1 = fit_multistart(problem, n_starts=3, seed=4, budget=budget)
    r2 = fit_multistart(problem, n_starts=3, seed=4, budget=budget)
    np.testing.assert_array_equal(r1.theta_free, r2.theta_free)
    assert [s.nll for s in r1.starts] == [s.nll for s in r2.starts]
    assert all(s.n_fun_grad_evals <= 60 + 20 for s in r1.starts)  # maxfun, plus the line search that exceeds it
    assert r1.n_fun_grad_evals == sum(s.n_fun_grad_evals for s in r1.starts) > 0
    assert r1.best_start.nll == min(s.nll for s in r1.starts)

    capped = fit_multistart(problem, n_starts=6, seed=4, budget=FitBudget(maxiter=5, max_wall_seconds=1e-9))
    assert len(capped.starts) == 1 and capped.stopped_by_wall_cap  # the wall cap stops further starts

    se = hessian_standard_errors(problem, r1.theta_free)
    manifest = build_manifest(r1, hessian=se, label="unit")
    path = write_manifest(tmp_path / "fit.json", manifest)
    loaded = json.loads(path.read_text())  # strict JSON
    assert loaded["schema"] == MANIFEST_SCHEMA
    prog = problem.program
    assert loaded["program_hash"] == prog.program_hash
    assert loaded["grammar_version"] == prog.grammar_version and loaded["compiler_build"] == prog.compiler_build
    assert loaded["seed"] == 4
    assert loaded["loss_per_start"] == [s.nll for s in r1.starts]
    used = loaded["budget"]["used"]
    assert used["fun_grad_evals"] == r1.n_fun_grad_evals and used["wall_seconds"] > 0
    assert used["iterations"] == r1.n_iterations and used["simulated_trial_steps"] == r1.n_simulated_steps
    assert loaded["budget"]["requested"]["maxfun_per_start"] == 60
    dl = loaded["description_length"]
    assert dl["l_params_bits"] == 32 and dl["l_total_bits"] == dl["l_struct_bits"] + 32
    est = {p["name"]: p for p in loaded["parameters"]}
    assert est["tau_h"]["fitted"] is False and est["tau_v"]["fitted"] is True
    assert "std_error_hessian" in est["gain"] and "quantised_estimate" in est["gain"]
    assert set(loaded["estimates"]) == {p.source_name for p in prog.parameters}
