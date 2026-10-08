"""Helpers for the Python conformance tests (OW-009): suite, IR cache, ``ow`` access and agreement statistics."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from occamworm.sim import jaxsim, reference
from occamworm.sim.cases import Case, load_suite
from occamworm.sim.graph import SimInput, sim_input_from_json
from occamworm.sim.ir import Program, compile_source, find_ow
from occamworm.sim.result import SimResult, result_from_json

SUITE_DIR = Path(__file__).parent
IR_CACHE = SUITE_DIR / "ir_cache"
OW = find_ow(required=False)
NEEDS_OW = pytest.mark.skipif(
    OW is None,
    reason="the ow binary was not found: build it (cmake --preset clang && cmake --build --preset clang) or set OW_CLI",
)

Simulate = Callable[[Program, SimInput], SimResult]
IMPLEMENTATIONS: dict[str, Simulate] = {"reference": reference.simulate, "jax": jaxsim.simulate}
DEVIATIONS: dict[str, float] = {}


def record_deviation(label: str, value: float) -> None:
    DEVIATIONS[label] = max(DEVIATIONS.get(label, 0.0), value)


ALL_CASES = load_suite(SUITE_DIR)


def cases_of(kind: str) -> list[Case]:
    return [c for c in ALL_CASES if c.kind == kind]


def case_ids(cases: list[Case]) -> list[str]:
    return [c.path.stem for c in cases]


def program_for(case: Case) -> Program:
    """Compile through ``ow`` when it is available, else from the committed IR cache."""
    return compile_source(case.source, cache_dir=IR_CACHE)


def update_cache_requested() -> bool:
    return os.environ.get("OW_UPDATE_IR_CACHE") == "1"


def cpp_run(source: str, raw_input: Mapping[str, Any]) -> SimResult:
    """``ow sim run`` on a program source and a simulation input, parsed into a :class:`SimResult`."""
    assert OW is not None
    with tempfile.TemporaryDirectory() as tmp:
        wrl = Path(tmp) / "p.wrl"
        wrl.write_text(source)
        case = Path(tmp) / "case.json"
        case.write_text(json.dumps({"input": raw_input}))
        out = Path(tmp) / "out.json"
        proc = subprocess.run(
            [str(OW), "sim", "run", "--program", str(wrl), "--case", str(case), "--out", str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        return result_from_json(json.loads(out.read_text()))


def case_runs(case: Case) -> list[tuple[str, dict[str, Any]]]:
    """The raw simulation inputs a case exercises (its own, permuted, or one per convergence ``dt``)."""
    raw: dict[str, Any] = json.loads(json.dumps(case.raw_input))
    if case.kind == "traces":
        return [("base", raw)]
    if case.kind == "permutation":
        base = dict(raw)
        base.pop("observed", None)
        permuted = json.loads(json.dumps(base))
        permuted["graph"]["neurons"] = [base["graph"]["neurons"][k] for k in case.check["permutation"]]
        return [("base", base), ("permuted", permuted)]
    if case.kind == "convergence":
        runs = []
        for dt in case.check["dts"]:
            steps = round(case.check["time"] / dt)
            run = dict(raw)
            run.update(dt=dt, n_steps=steps, sample_ticks=[steps], observed=[case.check["neuron"]])
            runs.append((f"dt={dt}", run))
        return runs
    return []


def parsed(raw_input: Mapping[str, Any]) -> SimInput:
    return sim_input_from_json(raw_input)
