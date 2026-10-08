"""OW-009: scalar reference, JAX simulator and C++ interpreter agree on random graphs, rules and parameters."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from occamworm.sim import jaxsim, reference
from occamworm.sim.cpp import simulate_cpp
from occamworm.sim.graph import sim_input_from_json
from occamworm.sim.ir import Program, find_ow
from occamworm.sim.result import result_deviation

RULES = ["leak-adapt.wrl", "leak-adapt-euler.wrl", "gap-leak.wrl", "rule90.wrl"]
TYPES = ["sensory", "inter", "motor"]


def random_input(program: Program, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = rng.randint(6, 14)
    ids = [f"n{k}" for k in range(n)]
    g0 = program.tier == "G0"
    chemical = [
        {
            "pre": rng.choice(ids),
            "post": rng.choice(ids),
            "weight": 1.0 if g0 else round(rng.uniform(0.0, 1.5), 3),
            "sign": 1 if g0 else rng.choice([1, -1]),
            "delay": rng.randint(0, 4),
        }
        for _ in range(rng.randint(n, 4 * n))
    ]
    gap = []
    if program.gap is not None:
        for _ in range(rng.randint(n // 2, 2 * n)):
            a, b = rng.sample(ids, 2)
            gap.append({"a": a, "b": b, "g": round(rng.uniform(0.0, 3.0), 3)})
    steps = rng.randint(10, 40)
    ticks = sorted(rng.sample(range(steps + 1), rng.randint(2, steps + 1)))
    params = {
        p.source_name: rng.uniform(p.lower, p.upper) for p in program.parameters if p.trainable and rng.random() < 0.8
    }
    first = program.registers[0].source_name
    if g0:
        initial = {first: {i: float(rng.randint(0, 2)) for i in rng.sample(ids, n // 2)}}
    else:
        initial = {first: {i: rng.uniform(-0.5, 0.5) for i in ids}}
    return {
        "dt": 0.05 if program.dt_max is not None else rng.choice([0.05, 0.1, 0.25]),
        "n_steps": steps,
        "sample_ticks": ticks,
        "graph": {"neurons": [{"id": i, "type": rng.choice(TYPES)} for i in ids], "chemical": chemical, "gap": gap},
        "stimulus": [
            {
                "neuron": rng.choice(ids),
                "start": rng.randint(0, steps),
                "end": rng.randint(0, steps + 5),
                "amplitude": 1.0 if g0 else round(rng.uniform(-1.5, 1.5), 3),
            }
            for _ in range(rng.randint(1, 5))
        ],
        "params": params,
        "initial_state": initial,
        "observed": rng.sample(ids, rng.randint(1, n)),
    }


def fix_stimulus(raw: dict[str, Any]) -> dict[str, Any]:
    for event in raw["stimulus"]:
        event["end"] = max(event["end"], event["start"])
    return raw


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("rule", RULES)
def test_random_graphs_agree(compile_wrl: Callable[[str], Program], rule: str, seed: int) -> None:
    source = (Path(__file__).parents[3] / "configs" / "rules" / rule).read_text()
    program = compile_wrl(source)
    raw = fix_stimulus(random_input(program, seed))
    sim_input = sim_input_from_json(raw)
    ref = reference.simulate(program, sim_input)
    jx = jaxsim.simulate(program, sim_input)
    assert result_deviation(jx, ref) <= 1e-12
    ow = find_ow(required=False)
    if ow is not None:
        cpp = simulate_cpp(source, raw)
        assert result_deviation(ref, cpp) <= 1e-12
        assert result_deviation(jx, cpp) <= 1e-12
