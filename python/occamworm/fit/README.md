# `occamworm.fit`

Parameter fitting against the differentiable simulator: bounded transforms, gradient checks against finite differences, multi-start L-BFGS-B, `L_params` accounting, uncertainty and a result manifest.

**Invariant:** training-only calibration. Nothing in this package looks at held-out animals; it fits the parameters of one fixed program on the data it is handed.

Tickets: [OW-011](../../../docs/planning/tickets/OW-011.md) · Spec: [§5.7](../../../docs/language/GRAMMAR.md), [§7.5](../../../docs/inference/SEARCH_AND_INFERENCE.md), [§7.6](../../../docs/inference/SEARCH_AND_INFERENCE.md), [WRL_SYNTAX.md §9](../../../docs/language/WRL_SYNTAX.md)

Status: implemented (OW-011). Known-truth G1 circuits recover their parameters within 3 standard errors (`tests/synthetic_truth/test_parameter_recovery.py`).

| Module | Role |
|---|---|
| `transforms.py` | Bounded maps taken from the IR bounds: `theta = lower + (upper - lower) sigmoid(raw)` (default for finite bounds) or `theta = lower + softplus(raw)` (one-sided; a finite upper bound is then enforced by the optimiser box on `raw`). Strictly increasing, exact inverses, `ParamTransforms.from_program` selects the fitted parameters (fixed parameters are rejected). |
| `losses.py` | Masked observation NLLs on `(trials, samples, neurons)`: `GaussianLoss(sigma)` (known noise SD, or `None` to profile it out, i.e. least squares) and `AR1Loss(sigma, phi)`, the exact masked Gaussian AR(1) likelihood of `occamworm.analysis.scoring.ar1_nll` reimplemented in JAX (tested equal to the numpy scorer and its gradient against finite differences). Each loss can also draw noise from its own model. |
| `problem.py` | `FitProblem(program, simulator, stimulus, target, loss, mask=..., free=...)`: the objective `raw -> theta -> simulation -> loss`, its value and gradient (`jax.value_and_grad`), Hessian in natural parameters, `check_gradient` (analytic against central finite differences), and `with_target` to reuse one compilation on replicate data. |
| `multistart.py` | `fit_multistart(problem, n_starts, seed, FitBudget(...))`: scipy L-BFGS-B on `raw` with exact JAX gradients. Start 0 is the declared (midpoint) parameter vector, the rest a seeded Latin hypercube over the bounded box, so starts are deterministic in `seed`. Objective is scaled to a per-observation NLL (otherwise L-BFGS-B's first step jumps into the saturated corner of the box). Counts function-and-gradient evaluations, iterations, simulated trial-steps and wall time per start and in total; an optional wall cap stops launching new starts. |
| `uncertainty.py` | `hessian_standard_errors` (inverse Hessian of the NLL at the optimum, flags non-positive-definite and at-bound estimates) and `parametric_bootstrap` (refits replicate datasets simulated from the fit and the loss's noise model). |
| `accounting.py` | `L_params` from declared precision (see below), quantisation to the declared grid, `description_length` (`L_struct` from the IR, `L_params`, `L_total`). |
| `manifest.py` | `build_manifest` / `write_manifest`: strict-JSON result record. |

```python
from occamworm.fit.losses import GaussianLoss
from occamworm.fit.manifest import build_manifest, write_manifest
from occamworm.fit.multistart import FitBudget, fit_multistart
from occamworm.fit.problem import FitProblem
from occamworm.fit.uncertainty import hessian_standard_errors

problem = FitProblem(program, sim, stimulus, target, GaussianLoss(0.05), mask=observed)  # sim: JaxSimulator
problem.check_gradient()  # (max relative error, analytic, finite differences)
fit = fit_multistart(problem, n_starts=8, seed=0, budget=FitBudget(maxiter=300, maxfun=1500))
se = hessian_standard_errors(problem, fit.theta_free)
write_manifest("artifacts/fit.json", build_manifest(fit, hessian=se))
```

**`L_params`.** Following WRL_SYNTAX.md §9 (bit code version 1), a trainable parameter with `b` declared bits is quantised to the `2^b`-point grid `lower + k (upper - lower) / (2^b - 1)` and costs `b` bits whatever its fitted value; `L_params` is the sum over the fitted parameters and equals the IR's `l_params.total_bits` when all trainable parameters are fitted. `L_struct` is read from the IR, `L_total = L_struct + L_params`. The manifest also reports each estimate snapped to its grid, the grid spacing and the NLL at the quantised estimates, so the declared precision is checkable. Value-dependent codes (GRAMMAR.md §5.7 allows them) are a later refinement that would bump `bit_code_version`.

**Manifest** (`occamworm.fit.manifest/0.1`): `program_hash`, `grammar_version`, `compiler_build`, `rule`, `seed`, `loss`, `data` summary, `parameters` (bounds, declared bits, estimate, quantised estimate, standard errors), `estimates`, `nll`, `starts` and `loss_per_start`, `iterations`, `budget.requested` and `budget.used` (function-and-gradient evaluations, iterations, simulated trial-steps, wall seconds, wall-cap flag), `description_length` (`l_struct_bits`, `l_params_bits`, `l_total_bits`), optional Hessian and bootstrap diagnostics, library versions.

**Notes.** Standard errors are asymptotic and assume the fitted program is the data-generating one with the stated noise model; they are for recovery tests and reporting, not a substitute for the held-out evaluation of the nested search. Fitting a G0 program is out of scope (no continuous parameters). Local optima exist (the tests log starts that stop in poorer minima): use several starts and report `loss_per_start`.

Tests: `tests/unit/fit/` (transforms, losses against the numpy scorer, accounting, gradients, determinism, manifest) and `tests/synthetic_truth/test_parameter_recovery.py`.
