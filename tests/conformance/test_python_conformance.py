"""OW-009 done-when, Python half: both Python implementations pass the 52-case suite and agree with C++ (§6.8).

Every non-error case runs through the scalar reference interpreter and the JAX simulator against the analytic
expected values with the case's own tolerance. Where the ``ow`` binary is present, the same inputs also run through
``ow sim run`` and the three implementations must agree (JAX vs scalar reference to 1e-12). Programs come from
``ow rule inspect`` or, without the binary, from the committed IR cache ``ir_cache/``; compile-error cases assert
that the C++ compiler rejects the program (Python only consumes IR, so it cannot accept what ``ow`` rejects).
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from conformance_support import (
    ALL_CASES,
    IMPLEMENTATIONS,
    IR_CACHE,
    NEEDS_OW,
    case_ids,
    case_runs,
    cases_of,
    cpp_run,
    parsed,
    program_for,
    record_deviation,
    update_cache_requested,
)

from occamworm.sim.cases import Case
from occamworm.sim.ir import CompilerError, compile_source, inspect_source, source_digest
from occamworm.sim.result import SimResult, result_deviation

AGREEMENT_TOLERANCE = 1e-12
TRACE_CASES = cases_of("traces")
PERMUTATION_CASES = cases_of("permutation")
CONVERGENCE_CASES = cases_of("convergence")
ERROR_CASES = cases_of("compile_error")
RUNNABLE_CASES = TRACE_CASES + PERMUTATION_CASES + CONVERGENCE_CASES


def test_suite_has_52_cases() -> None:
    assert len(ALL_CASES) == 52
    assert len(TRACE_CASES) + len(PERMUTATION_CASES) + len(CONVERGENCE_CASES) + len(ERROR_CASES) == 52


@pytest.mark.parametrize("impl", sorted(IMPLEMENTATIONS))
@pytest.mark.parametrize("case", TRACE_CASES, ids=case_ids(TRACE_CASES))
def test_traces_match_expected(case: Case, impl: str) -> None:
    program = program_for(case)
    if case.program_hash is not None:
        assert program.program_hash == case.program_hash
    result = IMPLEMENTATIONS[impl](program, case.sim_input())
    expected = case.check["expected"]
    tol = case.tolerance
    checked = 0
    for name, matrix in expected.get("registers", {}).items():
        assert name in result.registers, f"expected register '{name}' is not in the program"
        checked += _compare("register " + name, result.registers[name], matrix, tol.close)
    if "observation" in expected:
        checked += _compare("observation", result.observation, expected["observation"], tol.close)
    assert checked > 0


def _compare(
    label: str, actual: list[list[float]], expected: list[list[float]], close: Callable[[float, float], bool]
) -> int:
    assert len(actual) == len(expected), f"{label}: {len(actual)} samples, expected {len(expected)}"
    count = 0
    for s, (row, ref) in enumerate(zip(actual, expected, strict=True)):
        assert len(row) == len(ref), f"{label}: sample {s} has {len(row)} neurons, expected {len(ref)}"
        for i, (x, y) in enumerate(zip(row, ref, strict=True)):
            assert close(x, y), f"{label}[sample {s}][neuron {i}]: got {x!r}, expected {y!r}"
            count += 1
    return count


@pytest.mark.parametrize("impl", sorted(IMPLEMENTATIONS))
@pytest.mark.parametrize("case", PERMUTATION_CASES, ids=case_ids(PERMUTATION_CASES))
def test_permutation_equivariance(case: Case, impl: str) -> None:
    program = program_for(case)
    (_, base), (_, permuted) = case_runs(case)
    original = IMPLEMENTATIONS[impl](program, parsed(base))
    relabelled = IMPLEMENTATIONS[impl](program, parsed(permuted))
    tol = case.tolerance
    permutation = case.check["permutation"]
    assert sorted(permutation) == list(range(len(permutation)))
    for name in original.registers:
        for s, row in enumerate(relabelled.registers[name]):
            for j, value in enumerate(row):
                assert tol.close(value, original.registers[name][s][permutation[j]]), (name, s, j)
    for s, row in enumerate(relabelled.observation):
        for j, value in enumerate(row):
            assert tol.close(value, original.observation[s][permutation[j]]), ("observation", s, j)


@pytest.mark.parametrize("impl", sorted(IMPLEMENTATIONS))
@pytest.mark.parametrize("case", CONVERGENCE_CASES, ids=case_ids(CONVERGENCE_CASES))
def test_convergence_order(case: Case, impl: str) -> None:
    program = program_for(case)
    check = case.check
    errors = []
    for _, raw in case_runs(case):
        result = IMPLEMENTATIONS[impl](program, parsed(raw))
        errors.append(abs(result.registers[check["register"]][0][0] - float(check["expected"])))
    for coarse, fine in zip(errors, errors[1:], strict=False):
        assert fine <= float(check["max_ratio"]) * coarse, errors
    assert errors[-1] <= float(check["max_error_finest"]), errors


@pytest.mark.parametrize("case", RUNNABLE_CASES, ids=case_ids(RUNNABLE_CASES))
def test_jax_matches_scalar_reference(case: Case) -> None:
    program = program_for(case)
    for label, raw in case_runs(case):
        ref = IMPLEMENTATIONS["reference"](program, parsed(raw))
        jx = IMPLEMENTATIONS["jax"](program, parsed(raw))
        deviation = result_deviation(jx, ref)
        record_deviation("jax vs scalar reference", deviation)
        assert deviation <= AGREEMENT_TOLERANCE, f"{label}: {deviation:.3e}"


@NEEDS_OW
@pytest.mark.parametrize("impl", sorted(IMPLEMENTATIONS))
@pytest.mark.parametrize("case", RUNNABLE_CASES, ids=case_ids(RUNNABLE_CASES))
def test_agrees_with_cpp_interpreter(case: Case, impl: str) -> None:
    program = program_for(case)
    for label, raw in case_runs(case):
        cpp = cpp_run(case.source, raw)
        py = IMPLEMENTATIONS[impl](program, parsed(raw))
        _assert_same_shape(py, cpp)
        deviation = result_deviation(py, cpp)
        record_deviation(f"{impl} vs C++ (ow sim run)", deviation)
        assert deviation <= AGREEMENT_TOLERANCE, f"{label}: {deviation:.3e}"


def _assert_same_shape(a: SimResult, b: SimResult) -> None:
    assert a.sample_ticks == b.sample_ticks
    assert a.neurons == b.neurons
    assert set(a.registers) == set(b.registers)
    assert a.observation_operator == b.observation_operator
    assert a.observation_register == b.observation_register


@NEEDS_OW
@pytest.mark.parametrize("case", ERROR_CASES, ids=case_ids(ERROR_CASES))
def test_ow_rejects_compile_error_cases(case: Case) -> None:
    """Python consumes IR, so it can only run what ``ow`` accepts: the compiler must reject these programs."""
    with pytest.raises(CompilerError) as raised:
        compile_source(case.source)
    assert raised.value.code == case.check["code"]
    assert not (IR_CACHE / f"{source_digest(case.source)}.json").exists(), "a rejected program has a cached IR"


@NEEDS_OW
def test_ir_cache_is_fresh_and_complete() -> None:
    """The committed IR cache equals what the binary emits now (``OW_UPDATE_IR_CACHE=1`` rewrites it)."""
    stale = []
    for case in RUNNABLE_CASES:
        text = inspect_source(case.source)
        cached = IR_CACHE / f"{source_digest(case.source)}.json"
        if update_cache_requested():
            IR_CACHE.mkdir(exist_ok=True)
            cached.write_text(text)
        if not cached.is_file():
            stale.append(f"{case.path.name}: missing")
            continue
        fresh, old = json.loads(text), json.loads(cached.read_text())
        fresh.pop("compiler_build"), old.pop("compiler_build")
        if fresh != old:
            stale.append(f"{case.path.name}: differs")
    assert not stale, f"IR cache out of date, rerun with OW_UPDATE_IR_CACHE=1: {stale}"
