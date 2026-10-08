"""Motor output: neuron activity -> body-wall muscle activation (§13.3, §13.5; OW-016).

The map is fixed from the NMJ edges of one declared reconstruction; nothing in it is fitted. Sign follows the
presynaptic neuron's annotated transmitter when every source agrees on GABA-only (inhibitory) or ACh without GABA
(excitatory); any other neuron is excluded and listed in the provenance rather than given a guessed sign (§3.2).

Muscles named M{D,V}{L,R}NN are reduced to the body's joints: left and right quadrants are averaged and muscle
number NN (1-24, head to tail) is placed at joint round((NN - 1) * (joints - 1) / 23).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from occamworm.embodiment.interfaces import InterfaceError, MuscleActivation

FloatArray = npt.NDArray[np.float64]
MUSCLE = re.compile(r"^M([DV])([LR])(\d{2})$")


def joint_ids(n_joints: int) -> tuple[str, ...]:
    return tuple(f"D{j:02d}" for j in range(n_joints)) + tuple(f"V{j:02d}" for j in range(n_joints))


class NMJMotorOutput:
    """Linear NMJ map with a saturating clip to [0, 1] and per-joint saturation flags."""

    def __init__(
        self,
        neuron_ids: Sequence[str],
        nmj_edges: Sequence[dict[str, Any]],
        transmitters: dict[str, list[str]],
        n_joints: int,
        reconstruction: str,
        gain: float = 1.0,
        baseline: float = 0.5,
    ) -> None:
        self.neuron_ids = tuple(neuron_ids)
        self.muscle_ids = joint_ids(n_joints)
        index = {n: k for k, n in enumerate(self.neuron_ids)}
        sign: dict[str, float] = {}
        excluded: list[str] = []
        for n in self.neuron_ids:
            tx = set(transmitters.get(n, []))
            if tx == {"GABA"}:
                sign[n] = -1.0
            elif "ACh" in tx and "GABA" not in tx:
                sign[n] = 1.0
        w = np.zeros((2 * n_joints, len(self.neuron_ids)))
        used = 0
        for e in nmj_edges:
            if e["edge_kind"] != "nmj" or e["source_reconstruction"] != reconstruction:
                continue
            m = MUSCLE.match(str(e["target_neuron_id"]))
            src = str(e["source_neuron_id"])
            if m is None or src not in index:
                continue
            if src not in sign:
                excluded.append(src)
                continue
            side = 0 if m.group(1) == "D" else 1
            joint = round((int(m.group(3)) - 1) * (n_joints - 1) / 23)
            w[side * n_joints + joint, index[src]] += 0.5 * sign[src] * float(e["synapse_count"] or 0.0)
            used += 1
        scale = np.abs(w).sum(axis=1).max()
        self.weights = w / scale if scale > 0 else w
        self.gain = gain
        self.baseline = baseline
        self.provenance = {
            "reconstruction": reconstruction,
            "nmj_edges_used": used,
            "neurons_excluded_unknown_sign": sorted(set(excluded)),
            "sign_rule": "GABA-only -> inhibitory; ACh without GABA -> excitatory; otherwise excluded",
            "fitted_parameters": [],
            "gain": gain,
            "baseline": baseline,
        }

    def __call__(self, neuron_activity: FloatArray, neuron_ids: tuple[str, ...], time: float) -> MuscleActivation:
        if neuron_ids != self.neuron_ids:
            raise InterfaceError("NMJMotorOutput: neuron order differs from the map it was built for")
        raw = self.baseline + self.gain * (self.weights @ np.asarray(neuron_activity, dtype=np.float64))
        act = np.clip(raw, 0.0, 1.0)
        return MuscleActivation(self.muscle_ids, act, (raw < 0.0) | (raw > 1.0), self.provenance)


class DecoderOnlyControl:
    """§13.5 decoder-only control: a fixed travelling wave with no neural input at all."""

    def __init__(self, n_joints: int, frequency_hz: float = 0.5, wavelength_bodies: float = 1.0) -> None:
        self.muscle_ids = joint_ids(n_joints)
        self.n = n_joints
        self.f = frequency_hz
        self.k = 1.0 / wavelength_bodies

    def __call__(self, neuron_activity: FloatArray, neuron_ids: tuple[str, ...], time: float) -> MuscleActivation:
        s = np.arange(self.n) / max(self.n - 1, 1)
        w = 0.5 * np.sin(2 * np.pi * (self.k * s - self.f * time))
        act = np.concatenate([0.5 + w, 0.5 - w])
        return MuscleActivation(
            self.muscle_ids, act, np.zeros(act.size, dtype=bool), {"control": "decoder_only", "frequency_hz": self.f}
        )
