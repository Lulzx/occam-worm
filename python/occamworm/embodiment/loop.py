"""Closed- and open-loop runners, controls and behaviour endpoints (§13.2-§13.6, OW-016).

``run_loop`` wires SensoryInput -> NeuralStepper -> MotorOutput -> BodyStep and back. The neural program is
called only through ``NeuralStepper.step``; open-loop replay feeds a recorded stimulus history instead of the
sensors. ``RandomizedNeural`` replaces the neural program by AR(1) noise with matched scale (§13.5 control).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from occamworm.embodiment.body import RFTBody
from occamworm.embodiment.interfaces import (
    BodyState,
    BodyStep,
    InterfaceError,
    MotorOutput,
    NeuralStepper,
    SensoryInput,
    SensoryOutput,
)

FloatArray = npt.NDArray[np.float64]


@dataclass
class LoopTrace:
    time: list[float] = field(default_factory=list)
    head: list[FloatArray] = field(default_factory=list)
    heading: list[float] = field(default_factory=list)
    curvature: list[FloatArray] = field(default_factory=list)
    centroid: list[FloatArray] = field(default_factory=list)
    axis: list[FloatArray] = field(default_factory=list)
    saturated_fraction: list[float] = field(default_factory=list)
    neural: list[FloatArray] = field(default_factory=list)


class LinearStepper:
    """A fixed linear-rate network as a NeuralStepper (exact exponential step): x' = A x + stimulus."""

    def __init__(self, neuron_ids: tuple[str, ...], a: FloatArray, dt: float) -> None:
        from scipy.linalg import expm

        self.neuron_ids = neuron_ids
        self.dt = dt
        self._e = expm(a * dt)
        self._b = np.linalg.solve(a, self._e - np.eye(a.shape[0]))  # zero-order hold input

    def init(self) -> FloatArray:
        return np.zeros(len(self.neuron_ids))

    def step(self, state: FloatArray, stimulus: FloatArray) -> tuple[FloatArray, FloatArray]:
        x = self._e @ state + self._b @ stimulus
        return x, x


class RandomizedNeural:
    """§13.5 control: AR(1) activity with a declared scale replaces the neural program."""

    def __init__(self, neuron_ids: tuple[str, ...], dt: float, scale: float, tau_s: float = 1.0, seed: int = 0):
        self.neuron_ids = neuron_ids
        self.dt = dt
        self.scale = scale
        self.phi = float(np.exp(-dt / tau_s))
        self.rng = np.random.default_rng(seed)

    def init(self) -> FloatArray:
        return np.zeros(len(self.neuron_ids))

    def step(self, state: FloatArray, stimulus: FloatArray) -> tuple[FloatArray, FloatArray]:
        noise = self.rng.standard_normal(state.size) * self.scale * np.sqrt(1 - self.phi**2)
        x = self.phi * state + noise
        return x, x


class ProprioceptiveSensors:
    """Declared sensor map: neuron -> (joint, gain); drive = gain * curvature * body length. A hypothesis input."""

    def __init__(
        self,
        neuron_ids: tuple[str, ...],
        sensor_map: dict[str, tuple[int, float]],
        body_length: float,
        uncertainty: float = 0.0,
    ) -> None:
        self.neuron_ids = neuron_ids
        self.map = sensor_map
        self.length = body_length
        self.unc = uncertainty

    def __call__(self, time: float, environment: Any, body: BodyState) -> SensoryOutput:
        s = np.zeros(len(self.neuron_ids))
        for k, n in enumerate(self.neuron_ids):
            if n in self.map:
                joint, gain = self.map[n]
                s[k] = gain * body.curvature[joint] * self.length
        return SensoryOutput(self.neuron_ids, s, np.full(s.size, self.unc))


def run_loop(
    neural: NeuralStepper,
    motor: MotorOutput,
    body_step: BodyStep,
    body: RFTBody,
    duration: float,
    sensors: SensoryInput | None = None,
    replay: FloatArray | None = None,
    environment: Any = None,
) -> LoopTrace:
    """Closed loop with ``sensors``; open loop with ``replay`` (steps x neurons); neither = no sensory input."""
    if sensors is not None and replay is not None:
        raise InterfaceError("choose closed loop (sensors) or open-loop replay, not both")
    if sensors is not None and sensors.neuron_ids != neural.neuron_ids:
        raise InterfaceError("sensor neuron order differs from the neural program's")
    n_steps = int(round(duration / neural.dt))
    if replay is not None and replay.shape != (n_steps, len(neural.neuron_ids)):
        raise InterfaceError(f"replay must be ({n_steps}, {len(neural.neuron_ids)})")
    state = neural.init()
    pose = body.rest()
    trace = LoopTrace()
    for k in range(n_steps):
        t = k * neural.dt
        if sensors is not None:
            stim = sensors(t, environment, pose).stimulus
        elif replay is not None:
            stim = replay[k]
        else:
            stim = np.zeros(len(neural.neuron_ids))
        state, activity = neural.step(state, stim)
        muscles = motor(activity, neural.neuron_ids, t)
        res = body_step(pose, muscles, environment, neural.dt)
        pose = res.state
        mids = body.midpoints(pose.head, pose.heading, pose.curvature)
        axis = mids[0] - mids[-1]
        trace.time.append(pose.time)
        trace.head.append(pose.head)
        trace.heading.append(pose.heading)
        trace.curvature.append(pose.curvature)
        trace.centroid.append(mids.mean(axis=0))
        trace.axis.append(axis / max(float(np.linalg.norm(axis)), 1e-15))
        trace.saturated_fraction.append(float(np.mean(muscles.saturated)))
        trace.neural.append(np.asarray(activity, dtype=np.float64))
    return trace


def behaviour(trace: LoopTrace, dt: float, body_length: float) -> dict[str, float | None]:
    """§13.6 endpoints from one run: speed, forward fraction, curvature-wave frequency and wavelength."""
    if len(trace.centroid) < 3:
        return {"speed_mm_s": None}
    c = np.asarray(trace.centroid)
    ax = np.asarray(trace.axis)
    v = np.einsum("ni,ni->n", np.diff(c, axis=0) / dt, ax[:-1])
    kappa = np.asarray(trace.curvature)
    mid = kappa[:, kappa.shape[1] // 2] - kappa[:, kappa.shape[1] // 2].mean()
    spec = np.abs(np.fft.rfft(mid)) ** 2
    freqs = np.fft.rfftfreq(mid.size, dt)
    f_peak = float(freqs[1:][np.argmax(spec[1:])]) if mid.size > 4 and np.any(spec[1:] > 0) else None
    # spatial wavelength from the phase gradient of the dominant temporal frequency along the body
    wavelength = None
    if f_peak:
        ph = np.angle(np.fft.rfft(kappa - kappa.mean(axis=0), axis=0)[int(np.argmax(spec[1:])) + 1])
        slope = np.polyfit(np.linspace(0, 1, ph.size), np.unwrap(ph), 1)[0]
        wavelength = float(2 * np.pi / abs(slope)) if slope != 0 else None
    return {
        "speed_mm_s": float(np.mean(v) * 1e3),
        "forward_fraction": float(np.mean(v > 0)),
        "curvature_frequency_hz": f_peak,
        "wavelength_body_lengths": wavelength,
        "mean_saturated_fraction": float(np.mean(trace.saturated_fraction)),
        "body_length_mm": body_length * 1e3,
    }
