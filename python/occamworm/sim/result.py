"""Simulation result container shared by the Python simulators and the ``ow sim run`` output (§6.8)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Matrix = list[list[float]]  # [sample][reported neuron]


@dataclass
class SimResult:
    sample_ticks: list[int]
    neurons: list[str]
    registers: dict[str, Matrix]  # keyed by register source name
    observation: Matrix
    observation_operator: str = ""
    observation_register: str = ""


def result_from_json(data: Mapping[str, Any]) -> SimResult:
    """Parse the ``occamworm.sim.result/0.1`` JSON printed by ``ow sim run``."""
    obs = data["observation"]
    return SimResult(
        sample_ticks=[int(t) for t in data["sample_ticks"]],
        neurons=[str(n) for n in data["neurons"]],
        registers={str(k): [[float(x) for x in row] for row in v] for k, v in data["registers"].items()},
        observation=[[float(x) for x in row] for row in obs["values"]],
        observation_operator=str(obs["operator"]),
        observation_register=str(obs["register"]),
    )


def max_abs_deviation(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Largest ``|a - b|`` over two equally shaped matrices (``inf`` on a shape mismatch)."""
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b, strict=True)):
        return float("inf")
    return max((abs(x - y) for ra, rb in zip(a, b, strict=True) for x, y in zip(ra, rb, strict=True)), default=0.0)


def result_deviation(a: SimResult, b: SimResult) -> float:
    """Largest absolute difference over all shared registers and the observation (``inf`` if shapes differ)."""
    if set(a.registers) != set(b.registers) or a.sample_ticks != b.sample_ticks or a.neurons != b.neurons:
        return float("inf")
    parts = [max_abs_deviation(a.observation, b.observation)]
    parts += [max_abs_deviation(a.registers[k], b.registers[k]) for k in a.registers]
    return max(parts)
