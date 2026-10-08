"""``L_params`` and description-length accounting (GRAMMAR.md §5.7, WRL_SYNTAX.md §9, bit code version 1).

``L_struct`` is taken from the IR (the compiler computes it); ``L_params`` is the sum of the declared precision
``bits`` over the fitted trainable parameters: a trainable parameter with ``b`` bits is quantised to the
``2^b``-point grid ``lower + k (upper - lower) / (2^b - 1)``, ``k = 0 .. 2^b - 1``, and costs ``b`` bits whatever
its fitted value (value-dependent codes would bump ``bit_code_version``). With every trainable parameter fitted,
``L_params`` equals the IR's ``l_params.total_bits``. ``L_total = L_struct + L_params``. Fixed parameters are
program constants already counted inside ``L_struct``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from occamworm.sim.ir import Parameter, Program

LN2 = 0.6931471805599453


def fitted_parameters(program: Program, fitted: Iterable[str] | None = None) -> list[Parameter]:
    """Trainable parameters that are fitted (``fitted`` source names; default all trainable), in IR order."""
    wanted = None if fitted is None else set(fitted)
    out = []
    for par in program.parameters:
        if not par.trainable:
            if wanted is not None and par.source_name in wanted:
                raise ValueError(f"parameter '{par.source_name}' is fixed and carries no L_params")
            continue
        if wanted is None or par.source_name in wanted:
            out.append(par)
    return out


def l_params_bits(program: Program, fitted: Iterable[str] | None = None) -> int:
    """Declared-precision cost of the fitted parameters, in bits."""
    return sum(p.bits for p in fitted_parameters(program, fitted))


def grid_spacing(par: Parameter) -> float:
    """Spacing of the ``2^bits``-point quantisation grid over ``[lower, upper]``."""
    return float((par.upper - par.lower) / (2**par.bits - 1))


def quantise(par: Parameter, value: float) -> float:
    """Nearest point of the parameter's declared-precision grid (clipped into the bounds)."""
    spacing = grid_spacing(par)
    k = round((min(max(value, par.lower), par.upper) - par.lower) / spacing)
    return float(min(par.lower + k * spacing, par.upper))


def quantise_vector(program: Program, theta: Sequence[float] | npt.NDArray[np.float64]) -> list[float]:
    """``theta`` with every trainable parameter snapped to its grid (fixed parameters untouched)."""
    return [quantise(p, theta[p.index]) if p.trainable else float(theta[p.index]) for p in program.parameters]


def description_length(program: Program, fitted: Iterable[str] | None = None) -> dict[str, Any]:
    """``L_struct`` (from the IR), ``L_params`` (declared bits of the fitted parameters) and ``L_total``."""
    pars = fitted_parameters(program, fitted)
    l_struct = program.l_struct_bits
    l_params = sum(p.bits for p in pars)
    return {
        "bit_code_version": int(program.raw.get("bit_code_version", 1)),
        "l_struct_bits": l_struct,
        "l_params_bits": l_params,
        "l_total_bits": l_struct + l_params,
        "per_parameter_bits": {p.source_name: p.bits for p in pars},
        "unfitted_trainable": [p.source_name for p in program.parameters if p.trainable and p not in pars],
    }


def check_against_ir(program: Program) -> bool:
    """True when the all-trainable accounting reproduces the compiler's own ``l_params``."""
    declared: Mapping[str, Any] = program.raw["l_params"]
    return l_params_bits(program) == int(declared["total_bits"])
