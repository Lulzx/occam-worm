"""OW-009: IR validation, graph building and simulation-input validation (no ow binary needed: IR cache)."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from occamworm.sim import jaxsim, reference
from occamworm.sim.cases import load_case
from occamworm.sim.graph import (
    ChemicalEdge,
    GapJunction,
    GraphSpec,
    InputError,
    SimulationError,
    build_graph,
    permute_neurons,
    sim_input_from_json,
)
from occamworm.sim.ir import (
    CompilerNotFoundError,
    IrError,
    Program,
    compile_source,
    find_ow,
    load_ir,
    source_digest,
)

CONFORMANCE = Path(__file__).parents[2] / "conformance"
CACHE = CONFORMANCE / "ir_cache"


def cached_program(case_file: str) -> tuple[Program, dict[str, Any], str]:
    case = load_case(CONFORMANCE / case_file)
    program = compile_source(case.source, cache_dir=CACHE)
    return program, dict(case.raw_input), case.source


def ir_json(case_file: str = "13-gap-multigraph.json") -> dict[str, Any]:
    case = load_case(CONFORMANCE / case_file)
    path = CACHE / f"{source_digest(case.source)}.json"
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def test_load_ir_exposes_the_program() -> None:
    data = ir_json("31-param-override.json")
    program = load_ir(data)
    assert program.tier == "G1"
    assert program.l_struct_bits == data["l_struct"]["total_bits"]
    assert program.l_params_bits == data["l_params"]["total_bits"]
    assert program.theta_from_dict({}) == program.default_theta()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(schema="occamworm.wrl.ir/9"),
        lambda d: d.update(program_hash="xyz"),
        lambda d: d.update(tier="G7"),
        lambda d: d["instructions"][-1].update(op="exp"),
        lambda d: d["instructions"][-1].update(args=[len(d["instructions"])]),
        lambda d: d["instructions"].pop(0),
        lambda d: d.update(writes=d["writes"][:-1]),
        lambda d: d["writes"][0].update(value=10_000),
        lambda d: d["observation"].update(register=7),
        lambda d: d["observation"].update(operator="fluorescence"),
        lambda d: d["registers"][0].update(index=3),
    ],
)
def test_load_ir_rejects_malformed_documents(mutate: Any) -> None:
    data = copy.deepcopy(ir_json())
    mutate(data)
    with pytest.raises(IrError):
        load_ir(data)


def test_load_ir_rejects_bad_attributes_and_calcium_tau() -> None:
    data = copy.deepcopy(ir_json("19-calcium-impulse.json"))
    data["observation"]["tau"] = {"param": 0, "const": 1.0}
    with pytest.raises(IrError):
        load_ir(data)
    data = copy.deepcopy(ir_json("04-chemical-excitation.json"))
    for ins in data["instructions"]:
        if ins["op"] == "sum_in":
            ins["attrs"]["select"] = "bogus"
    with pytest.raises(IrError):
        load_ir(data)


def test_missing_binary_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OW_CLI", "/nonexistent/ow")
    with pytest.raises(CompilerNotFoundError, match="OW_CLI"):
        find_ow()
    assert find_ow(required=False) is None
    case = load_case(CONFORMANCE / "01-leaky-const-drive.json")
    assert compile_source(case.source, cache_dir=CACHE).tier == "G1"  # served from the cache
    with pytest.raises(CompilerNotFoundError):
        compile_source(case.source)


def test_graph_rows_follow_the_reference_orientation() -> None:
    spec = GraphSpec(
        neurons=(("A", "x"), ("B", "y"), ("C", "x")),
        chemical=(
            ChemicalEdge("C", "A", 0.5, -1, 2),
            ChemicalEdge("B", "A", 0.7, 1, 0),
            ChemicalEdge("B", "A", 0.2, 1, 0),
            ChemicalEdge("A", "C", 1.0, 1, 1),
        ),
        gap=(GapJunction("A", "B", 2.0), GapJunction("B", "A", 1.0)),
    )
    g = build_graph(spec)
    assert g.in_offsets == (0, 3, 3, 4)
    assert g.pre == (1, 1, 2, 0) and g.weight == (0.2, 0.7, 0.5, 1.0)  # row A sorted by (pre, delay, sign, weight)
    assert g.gap_offsets == (0, 2, 4, 4)
    assert g.gap_neighbor == (1, 1, 0, 0) and g.gap_conductance == (1.0, 2.0, 1.0, 2.0)
    assert g.max_edge_delay == 2


@pytest.mark.parametrize(
    "spec",
    [
        GraphSpec((("A", "t"), ("A", "t"))),
        GraphSpec((("bad id", "t"),)),
        GraphSpec((("A", "t"),), chemical=(ChemicalEdge("A", "Z"),)),
        GraphSpec((("A", "t"),), chemical=(ChemicalEdge("A", "A", -1.0),)),
        GraphSpec((("A", "t"),), chemical=(ChemicalEdge("A", "A", 1.0, 0),)),
        GraphSpec((("A", "t"),), chemical=(ChemicalEdge("A", "A", 1.0, 1, -1),)),
        GraphSpec((("A", "t"),), chemical=(ChemicalEdge("A", "A", float("nan")),)),
        GraphSpec((("A", "t"), ("B", "t")), gap=(GapJunction("A", "A", 1.0),)),
        GraphSpec((("A", "t"), ("B", "t")), gap=(GapJunction("A", "B", -1.0),)),
    ],
)
def test_invalid_graphs_are_rejected(spec: GraphSpec) -> None:
    with pytest.raises(InputError):
        build_graph(spec)


def test_edits_and_dense_initial_state_are_applied() -> None:
    raw = {
        "dt": 0.1,
        "n_steps": 2,
        "graph": {
            "neurons": [{"id": "A"}, {"id": "B"}],
            "chemical": [{"pre": "A", "post": "B"}, {"pre": "A", "post": "B", "weight": 2}, {"pre": "B", "post": "A"}],
            "gap": [{"a": "A", "b": "B", "g": 1.0}],
        },
        "edits": {"delete_chemical": [["A", "B"]], "delete_gap": [["B", "A"]]},
        "initial_state": {"v": [0.5, 0.25]},
    }
    parsed = sim_input_from_json(raw)
    assert [(e.pre, e.post) for e in parsed.graph.chemical] == [("B", "A")]
    assert parsed.graph.gap == ()
    assert parsed.initial_state["v"] == {"A": 0.5, "B": 0.25}
    assert parsed.sample_ticks == (0, 1, 2)
    raw["edits"] = {"delete_gap": [["A", "Q"]]}
    with pytest.raises(InputError):
        sim_input_from_json(raw)
    raw["edits"] = {}
    raw["initial_state"] = {"v": [0.5]}
    with pytest.raises(InputError):
        sim_input_from_json(raw)


def test_permute_neurons_reorders_the_neuron_list() -> None:
    _, raw, _ = cached_program("34-permutation-equivariance.json")
    base = sim_input_from_json(raw)
    moved = permute_neurons(base, [2, 0, 1, 3, 4, 5])
    assert [n for n, _ in moved.graph.neurons][:3] == ["N2", "N0", "N1"]


@pytest.mark.parametrize("impl", [reference.simulate, jaxsim.simulate])
def test_run_validation_matches_the_cpp_runtime(impl: Any) -> None:
    program, raw, _ = cached_program("36-convergence-euler.json")  # dt_max 0.1
    bad: list[dict[str, Any]] = [
        {**raw, "dt": 0.2},
        {**raw, "dt": 0.0},
        {**raw, "n_steps": -1},
        {**raw, "sample_ticks": [0, 0]},
        {**raw, "sample_ticks": [0, 99]},
        {**raw, "stimulus": [{"neuron": "Q", "start": 0, "end": 1, "amplitude": 1.0}]},
        {**raw, "stimulus": [{"neuron": "A", "start": 3, "end": 1, "amplitude": 1.0}]},
        {**raw, "observed": ["Q"]},
        {**raw, "params": {"nope": 1.0}},
        {**raw, "initial_state": {"v": {"Q": 1.0}}},
        {**raw, "initial_state": {"nope": {"A": 1.0}}},
    ]
    for item in bad:
        with pytest.raises(InputError):
            impl(program, sim_input_from_json(item))


def test_parameter_bounds_and_dead_code_names() -> None:
    program, raw, _ = cached_program("32-dead-code-names-ignored.json")
    assert program.eliminated_parameters
    assert program.theta_from_dict({program.eliminated_parameters[0]: 1e9}) == program.default_theta()
    program, raw, _ = cached_program("31-param-override.json")
    trainable = next(p for p in program.parameters if p.trainable)
    with pytest.raises(ValueError):
        program.theta_from_dict({trainable.source_name: trainable.upper * 2 + 1})
    with pytest.raises(InputError):
        reference.simulate(
            program, sim_input_from_json({**raw, "params": {trainable.source_name: trainable.upper + 1}})
        )


@pytest.mark.parametrize("impl", [reference.simulate, jaxsim.simulate])
def test_non_finite_state_is_a_runtime_error(compile_wrl: Callable[[str], Program], impl: Any) -> None:
    program = compile_wrl("wrl 0.1\ntier G1\nstate v : 1 = 10\nnext v = v * v * v\nobserve identity_v1(v)\n")
    raw = {"dt": 0.1, "n_steps": 8, "graph": {"neurons": [{"id": "A"}]}}
    with pytest.raises(SimulationError):
        impl(program, sim_input_from_json(raw))
