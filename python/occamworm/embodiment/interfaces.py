"""Strict embodiment interfaces (§13.3, OW-016).

Each boundary has one typed record and one protocol. Records validate shape, finiteness and declared units on
construction, so a mismatched connector fails loudly instead of silently broadcasting. The neural side is any
``NeuralStepper``: the loop only calls ``step``, so a frozen program is connected without touching simulator
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class InterfaceError(ValueError):
    """Raised when a record crossing an embodiment boundary is malformed."""


def _check(name: str, x: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape != shape:
        raise InterfaceError(f"{name}: shape {a.shape}, expected {shape}")
    if not np.all(np.isfinite(a)):
        raise InterfaceError(f"{name}: non-finite values")
    return a


@dataclass(frozen=True)
class BodyState:
    """Planar body: head position (m), head-segment heading (rad), joint curvatures (1/m)."""

    head: FloatArray  # (2,)
    heading: float
    curvature: FloatArray  # (n_segments - 1,)
    time: float = 0.0

    def __post_init__(self) -> None:
        _check("BodyState.head", self.head, (2,))
        if not np.isfinite(self.heading):
            raise InterfaceError("BodyState.heading: non-finite")
        _check("BodyState.curvature", self.curvature, (np.asarray(self.curvature).size,))


@dataclass(frozen=True)
class SensoryOutput:
    """``SensoryInput`` result: stimulus per neuron (dimensionless drive) and its uncertainty."""

    neuron_ids: tuple[str, ...]
    stimulus: FloatArray
    uncertainty: FloatArray

    def __post_init__(self) -> None:
        n = len(self.neuron_ids)
        _check("SensoryOutput.stimulus", self.stimulus, (n,))
        u = _check("SensoryOutput.uncertainty", self.uncertainty, (n,))
        if np.any(u < 0):
            raise InterfaceError("SensoryOutput.uncertainty: negative")


@dataclass(frozen=True)
class MuscleActivation:
    """``MotorOutput`` result: activation in [0, 1] per muscle, with provenance and saturation flags."""

    muscle_ids: tuple[str, ...]
    activation: FloatArray
    saturated: BoolArray
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        a = _check("MuscleActivation.activation", self.activation, (len(self.muscle_ids),))
        if np.any(a < 0) or np.any(a > 1):
            raise InterfaceError("MuscleActivation.activation: outside [0, 1]")
        if np.asarray(self.saturated).shape != a.shape:
            raise InterfaceError("MuscleActivation.saturated: shape mismatch")


@dataclass(frozen=True)
class BodyStepResult:
    state: BodyState
    velocity: FloatArray  # (3,): head velocity x, y (m/s) and heading rate (rad/s)
    forces: FloatArray  # (n_segments, 2) drag force per segment (N)
    sensory_fields: dict[str, float]


class SensoryInput(Protocol):
    neuron_ids: tuple[str, ...]

    def __call__(self, time: float, environment: Any, body: BodyState) -> SensoryOutput: ...


class MotorOutput(Protocol):
    muscle_ids: tuple[str, ...]

    def __call__(self, neuron_activity: FloatArray, neuron_ids: tuple[str, ...], time: float) -> MuscleActivation: ...


class BodyStep(Protocol):
    def __call__(self, body: BodyState, muscles: MuscleActivation, environment: Any, dt: float) -> BodyStepResult: ...


class NeuralStepper(Protocol):
    """A frozen neural program: ``step`` advances one tick given external stimulus per neuron."""

    neuron_ids: tuple[str, ...]
    dt: float

    def init(self) -> Any: ...

    def step(self, state: Any, stimulus: FloatArray) -> tuple[Any, FloatArray]: ...
