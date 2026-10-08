"""Fit result manifest (JSON): what was fitted, how, at what cost, with which description length (OW-011)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jax
import numpy as np
import scipy

from occamworm.fit.accounting import description_length, grid_spacing, quantise_vector
from occamworm.fit.multistart import FitResult, StartResult
from occamworm.fit.uncertainty import BootstrapResult, HessianSE

MANIFEST_SCHEMA = "occamworm.fit.manifest/0.1"


def _start_json(s: StartResult, names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "start": s.index,
        "initial": {n: float(v) for n, v in zip(names, s.theta0, strict=True)},
        "final": {n: float(v) for n, v in zip(names, s.theta, strict=True)},
        "nll": s.nll,
        "iterations": s.n_iterations,
        "fun_grad_evals": s.n_fun_grad_evals,
        "wall_seconds": s.wall_seconds,
        "converged": s.success,
        "message": s.message,
        "max_abs_gradient_raw": s.grad_norm,
    }


def build_manifest(
    result: FitResult,
    hessian: HessianSE | None = None,
    bootstrap: BootstrapResult | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """The manifest: program identity, estimates, per-start losses, budget used and the description length."""
    problem = result.problem
    program = problem.program
    names = problem.free_names
    theta = result.theta_full
    quantised = quantise_vector(program, theta)
    quantised_nll = problem.nll_at(problem.theta_free(quantised))
    se_h = {} if hessian is None else hessian.to_dict()
    se_b = {} if bootstrap is None else {n: float(v) for n, v in zip(bootstrap.names, bootstrap.se, strict=True)}
    parameters = []
    for p in program.parameters:
        entry: dict[str, Any] = {
            "name": p.source_name,
            "canonical_name": p.name,
            "unit": p.unit,
            "lower": p.lower,
            "upper": p.upper,
            "trainable": p.trainable,
            "bits": p.bits,
            "fitted": p.source_name in names,
            "estimate": float(theta[p.index]),
        }
        if p.source_name in names:
            entry["quantised_estimate"] = float(quantised[p.index])
            entry["grid_spacing"] = grid_spacing(p)
            if p.source_name in se_h:
                entry["std_error_hessian"] = se_h[p.source_name]
            if p.source_name in se_b:
                entry["std_error_bootstrap"] = se_b[p.source_name]
        parameters.append(entry)
    wall = result.wall_seconds
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "label": label,
        "program_hash": program.program_hash,
        "grammar_version": program.grammar_version,
        "compiler_build": program.compiler_build,
        "rule": program.rule,
        "tier": program.tier,
        "seed": result.seed,
        "fitted_parameters": list(names),
        "transform": {t.name: t.kind for t in problem.transforms.transforms},
        "loss": problem.loss.to_json(),
        "data": {
            "n_trials": problem.n_trials,
            "n_samples": int(problem.target.shape[1]),
            "n_neurons": int(problem.target.shape[2]),
            "n_observed_values": int(problem.mask.sum()),
            "dt": problem.simulator.dt,
            "n_steps": problem.simulator.n_steps,
        },
        "parameters": parameters,
        "estimates": result.estimates,
        "nll": result.best_start.nll,
        "nll_at_quantised_estimates": quantised_nll,
        "best_start": result.best,
        "starts": [_start_json(s, names) for s in result.starts],
        "loss_per_start": [s.nll for s in result.starts],
        "iterations": {"best_start": result.best_start.n_iterations, "total": result.n_iterations},
        "budget": {
            "requested": {**result.budget.to_json(), "n_starts": result.n_starts_requested},
            "used": {
                "n_starts": len(result.starts),
                "fun_grad_evals": result.n_fun_grad_evals,
                "iterations": result.n_iterations,
                "simulated_trial_steps": result.n_simulated_steps,
                "wall_seconds": wall,
                "stopped_by_wall_cap": result.stopped_by_wall_cap,
            },
        },
        "description_length": description_length(program, names),
        "software": {"jax": jax.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
    }
    if hessian is not None:
        manifest["uncertainty_hessian"] = {
            "positive_definite": hessian.positive_definite,
            "at_bound": dict(zip(hessian.names, hessian.at_bound, strict=True)),
            "condition_number": hessian.condition_number,
        }
    if bootstrap is not None:
        manifest["uncertainty_bootstrap"] = {
            "n_replicates": int(bootstrap.estimates.shape[0]),
            "bias": {n: float(v) for n, v in zip(bootstrap.names, bootstrap.bias, strict=True)},
            "fun_grad_evals": bootstrap.n_fun_grad_evals,
            "wall_seconds": bootstrap.wall_seconds,
        }
    return manifest


def _finite(value: Any) -> Any:
    """Strict JSON: non-finite floats (for example the standard error of a boundary estimate) become ``null``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def write_manifest(path: Path | str, manifest: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_finite(manifest), indent=2, allow_nan=False) + "\n")
    return p
