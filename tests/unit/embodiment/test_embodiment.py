"""OW-016: strict interfaces, a reference body, a fixed neural program in the loop, and decoder-only controls."""

import numpy as np
import pytest

from occamworm.embodiment.body import RFTBody
from occamworm.embodiment.interfaces import BodyState, InterfaceError, MuscleActivation, SensoryOutput
from occamworm.embodiment.loop import (
    LinearStepper,
    ProprioceptiveSensors,
    RandomizedNeural,
    behaviour,
    run_loop,
)
from occamworm.embodiment.motor import DecoderOnlyControl, NMJMotorOutput, joint_ids

BODY = RFTBody(n_segments=12)
J = BODY.n_segments - 1
DT = 0.01


def progress(sign: int, seconds: float = 2.0) -> float:
    st = BODY.rest()
    s = np.arange(J) / (J - 1)
    total = 0.0
    for k in range(int(seconds / DT)):
        t = k * DT
        w = 0.5 * np.sin(2 * np.pi * (s - sign * 0.5 * t)) if sign else 0.5 * np.sin(2 * np.pi * s) * np.sin(np.pi * t)
        a = np.concatenate([0.5 + w, 0.5 - w])
        m0 = BODY.midpoints(st.head, st.heading, st.curvature)
        res = BODY(st, MuscleActivation(joint_ids(J), a, np.zeros(a.size, bool)), None, DT)
        m1 = BODY.midpoints(res.state.head, res.state.heading, res.state.curvature)
        axis = (m0[0] - m0[-1]) / np.linalg.norm(m0[0] - m0[-1])
        total += float((m1.mean(0) - m0.mean(0)) @ axis)
        st = res.state
        assert np.abs(res.forces.sum(axis=0)).max() < 1e-15  # force-free swimmer
    return total


def test_body_moves_forward_backward_and_not_for_standing_waves() -> None:
    fwd, back, stand = progress(1), progress(-1), progress(0)
    assert fwd > 0 and back < 0
    assert abs(stand) < 0.1 * fwd


def test_zero_drive_stays_at_rest() -> None:
    a = np.full(2 * J, 0.5)
    res = BODY(BODY.rest(), MuscleActivation(joint_ids(J), a, np.zeros(a.size, bool)), None, DT)
    np.testing.assert_allclose(res.velocity, 0.0, atol=1e-18)


def test_interfaces_reject_malformed_records() -> None:
    with pytest.raises(InterfaceError):
        MuscleActivation(("D00",), np.array([1.5]), np.array([False]))
    with pytest.raises(InterfaceError):
        SensoryOutput(("A", "B"), np.zeros(3), np.zeros(2))
    with pytest.raises(InterfaceError):
        BodyState(np.array([np.nan, 0.0]), 0.0, np.zeros(3))
    with pytest.raises(InterfaceError):
        BODY(BODY.rest(), MuscleActivation(("D00",), np.array([0.5]), np.array([False])), None, DT)


NEURONS = ("DB1", "VB1", "DD1", "AVBL")
TX = {"DB1": ["ACh"], "VB1": ["ACh"], "DD1": ["GABA"], "AVBL": ["ACh", "GABA"]}
EDGES = [
    {
        "source_neuron_id": "DB1",
        "target_neuron_id": "MDL01",
        "edge_kind": "nmj",
        "synapse_count": 4.0,
        "source_reconstruction": "toy",
    },
    {
        "source_neuron_id": "VB1",
        "target_neuron_id": "MVR12",
        "edge_kind": "nmj",
        "synapse_count": 2.0,
        "source_reconstruction": "toy",
    },
    {
        "source_neuron_id": "DD1",
        "target_neuron_id": "MDR24",
        "edge_kind": "nmj",
        "synapse_count": 2.0,
        "source_reconstruction": "toy",
    },
    {
        "source_neuron_id": "AVBL",
        "target_neuron_id": "MDL02",
        "edge_kind": "nmj",
        "synapse_count": 9.0,
        "source_reconstruction": "toy",
    },
]


def test_nmj_map_signs_and_saturation() -> None:
    motor = NMJMotorOutput(NEURONS, EDGES, TX, J, "toy", gain=1.0)
    assert motor.provenance["neurons_excluded_unknown_sign"] == ["AVBL"]
    assert motor.provenance["fitted_parameters"] == []
    out = motor(np.array([10.0, 0.0, 10.0, 0.0]), NEURONS, 0.0)
    ids = list(out.muscle_ids)
    assert out.activation[ids.index("D00")] == 1.0 and out.saturated[ids.index("D00")]
    assert out.activation[ids.index(f"D{J - 1:02d}")] == 0.0  # GABA drives below baseline
    with pytest.raises(InterfaceError):
        motor(np.zeros(4), ("X", "Y", "Z", "W"), 0.0)


def oscillator() -> LinearStepper:
    """A fixed, predeclared 4-neuron program: a damped rotation coupling the B and D motor neurons."""
    w, damp = 2 * np.pi * 0.5, 0.05
    a = np.array([[-damp, -w, 0, 0], [w, -damp, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, -1.0]])
    return LinearStepper(NEURONS, a, DT)


def test_fixed_program_runs_closed_and_open_loop() -> None:
    motor = NMJMotorOutput(NEURONS, EDGES, TX, J, "toy", gain=0.5)
    sensors = ProprioceptiveSensors(NEURONS, {"DB1": (0, 1.0), "VB1": (5, -1.0)}, BODY.length)
    closed = run_loop(oscillator(), motor, BODY, BODY, 2.0, sensors=sensors)
    assert len(closed.time) == 200 and np.all(np.isfinite(np.asarray(closed.curvature)))
    replay = np.zeros((200, 4))
    replay[:5, 0] = 1.0
    a = run_loop(oscillator(), motor, BODY, BODY, 2.0, replay=replay)
    b = run_loop(oscillator(), motor, BODY, BODY, 2.0, replay=replay)
    np.testing.assert_array_equal(np.asarray(a.head), np.asarray(b.head))  # deterministic replay
    with pytest.raises(InterfaceError):
        run_loop(oscillator(), motor, BODY, BODY, 2.0, sensors=sensors, replay=replay)


def test_decoder_only_and_randomized_controls_run() -> None:
    zero = LinearStepper(NEURONS, -np.eye(4), DT)
    dec = run_loop(zero, DecoderOnlyControl(J), BODY, BODY, 4.0)
    m = behaviour(dec, DT, BODY.length)
    assert m["speed_mm_s"] is not None and m["speed_mm_s"] > 0  # the decoder alone swims: §13.5 caveat
    assert m["curvature_frequency_hz"] == pytest.approx(0.5, abs=0.15)
    motor = NMJMotorOutput(NEURONS, EDGES, TX, J, "toy", gain=0.5)
    rnd = run_loop(RandomizedNeural(NEURONS, DT, scale=1.0, seed=1), motor, BODY, BODY, 2.0)
    assert np.all(np.isfinite(np.asarray(rnd.head)))
    assert behaviour(rnd, DT, BODY.length)["speed_mm_s"] is not None
