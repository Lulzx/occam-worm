"""Bounded parameter transforms (SEARCH_AND_INFERENCE.md §7.5).

Optimisation runs on unconstrained ``raw`` values. A parameter with finite bounds ``[lower, upper]`` (every
trainable WRL parameter has them, WRL_SYNTAX.md §1) uses a sigmoid into the interval,
``theta = lower + (upper - lower) * sigmoid(raw)``. The one-sided alternative of §7.5,
``theta = lower + softplus(raw)``, is available per parameter (``kind="softplus"``) and is chosen automatically when
``upper`` is infinite; for a finite ``upper`` the optimiser box on ``raw`` then enforces the upper bound. Both maps
are strictly increasing, so a bounded optimum in ``theta`` is an optimum in ``raw`` and the transform adds no
local minima.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from occamworm.sim import jaxsim  # noqa: F401  (enables float64 in JAX before any array is created)
from occamworm.sim.ir import Parameter, Program

FloatArray = npt.NDArray[np.float64]
Kind = Literal["sigmoid", "softplus"]

RAW_LIMIT = 15.0  # |raw| beyond this is numerically saturated for a sigmoid (theta within 3e-7 of a bound)
_EDGE = 1e-12  # keeps the inverse maps finite when theta sits on a bound


@dataclass(frozen=True)
class ParamTransform:
    name: str
    lower: float
    upper: float
    kind: Kind

    def __post_init__(self) -> None:
        if not self.lower < self.upper:
            raise ValueError(f"{self.name}: bounds must satisfy lower < upper")
        if self.kind == "sigmoid" and not math.isfinite(self.upper - self.lower):
            raise ValueError(f"{self.name}: a sigmoid transform needs finite bounds")
        if not math.isfinite(self.lower):
            raise ValueError(f"{self.name}: the lower bound must be finite")

    def forward(self, raw: jax.Array) -> jax.Array:
        if self.kind == "sigmoid":
            return self.lower + (self.upper - self.lower) * jax.nn.sigmoid(raw)
        return self.lower + jax.nn.softplus(raw)

    def inverse(self, theta: float) -> float:
        """``raw`` with ``forward(raw) = theta`` (``theta`` is clipped into the open interval first)."""
        if self.kind == "sigmoid":
            u = (theta - self.lower) / (self.upper - self.lower)
            u = min(max(u, _EDGE), 1.0 - _EDGE)
            return math.log(u / (1.0 - u))
        y = max(theta - self.lower, _EDGE)
        return y + math.log(-math.expm1(-y)) if y < 30.0 else y  # inverse softplus, stable for large y

    def raw_bounds(self) -> tuple[float, float]:
        """Box on ``raw`` handed to L-BFGS-B: saturation limits, or the image of a finite upper bound."""
        if self.kind == "sigmoid":
            return -RAW_LIMIT, RAW_LIMIT
        hi = self.inverse(self.upper) if math.isfinite(self.upper) else RAW_LIMIT
        return -RAW_LIMIT, hi

    def span(self) -> float:
        """Width used to place random starting points."""
        return self.upper - self.lower if math.isfinite(self.upper - self.lower) else 10.0 * max(1.0, abs(self.lower))


def transform_for(par: Parameter, kind: Kind | None = None) -> ParamTransform:
    """Transform from the declared IR bounds of a parameter (``sigmoid`` unless ``upper`` is infinite)."""
    chosen: Kind = kind if kind is not None else ("sigmoid" if math.isfinite(par.upper) else "softplus")
    return ParamTransform(par.source_name, par.lower, par.upper, chosen)


class ParamTransforms:
    """Transforms for an ordered set of fitted parameters, acting on vectors."""

    def __init__(self, transforms: Sequence[ParamTransform]) -> None:
        self.transforms = tuple(transforms)

    @classmethod
    def from_program(
        cls, program: Program, names: Iterable[str] | None = None, kind: Kind | None = None
    ) -> ParamTransforms:
        """Transforms for ``names`` (source names; default: every trainable parameter), in IR order."""
        wanted = None if names is None else set(names)
        chosen = []
        for par in program.parameters:
            if wanted is None:
                if par.trainable:
                    chosen.append(transform_for(par, kind))
            elif par.source_name in wanted:
                if not par.trainable:
                    raise ValueError(f"parameter '{par.source_name}' is fixed and cannot be fitted")
                chosen.append(transform_for(par, kind))
        if wanted is not None:
            missing = wanted - {t.name for t in chosen}
            if missing:
                raise ValueError(f"unknown parameters {sorted(missing)}")
        return cls(chosen)

    def __len__(self) -> int:
        return len(self.transforms)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.transforms)

    def forward(self, raw: jax.Array) -> jax.Array:
        """``theta`` (natural units) from an unconstrained ``raw`` vector."""
        return jnp.stack([t.forward(raw[k]) for k, t in enumerate(self.transforms)])

    def inverse(self, theta: Sequence[float] | FloatArray) -> FloatArray:
        return np.asarray([t.inverse(float(theta[k])) for k, t in enumerate(self.transforms)], dtype=np.float64)

    def raw_bounds(self) -> list[tuple[float, float]]:
        return [t.raw_bounds() for t in self.transforms]

    def lower(self) -> FloatArray:
        return np.asarray([t.lower for t in self.transforms], dtype=np.float64)

    def upper(self) -> FloatArray:
        return np.asarray([t.upper for t in self.transforms], dtype=np.float64)
