"""Graph, stimulus and simulation-input loaders (WRL_SYNTAX.md §6.1, §6.2).

Pure Python, no numpy: the scalar reference interpreter uses these structures directly and the JAX simulator
converts them to arrays. Validation mirrors ``Graph::build`` and ``sim_input_from_json`` of the C++ runtime.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

MAX_STEPS = 10_000_000
MAX_EDGE_DELAY_TICKS = 100_000
_NAME = re.compile(r"[A-Za-z0-9_.:\-]{1,64}")


class InputError(ValueError):
    """Invalid simulation input or graph (``E_INPUT`` / ``E_GRAPH`` of the C++ runtime)."""


class SimulationError(RuntimeError):
    """Non-finite value during a run (``E_RUNTIME``)."""


@dataclass(frozen=True)
class ChemicalEdge:
    pre: str
    post: str
    weight: float = 1.0
    sign: int = 1
    delay: int = 0


@dataclass(frozen=True)
class GapJunction:
    a: str
    b: str
    g: float


@dataclass(frozen=True)
class GraphSpec:
    neurons: tuple[tuple[str, str], ...]  # (id, type)
    chemical: tuple[ChemicalEdge, ...] = ()
    gap: tuple[GapJunction, ...] = ()


@dataclass(frozen=True)
class StimulusEvent:
    neuron: str
    start: int
    end: int
    amplitude: float


@dataclass(frozen=True)
class SimInput:
    dt: float
    n_steps: int
    graph: GraphSpec
    sample_ticks: tuple[int, ...]
    stimulus: tuple[StimulusEvent, ...] = ()
    params: Mapping[str, float] = field(default_factory=dict)
    initial_state: Mapping[str, Mapping[str, float]] = field(default_factory=dict)  # register -> neuron id -> value
    observed: tuple[str, ...] = ()  # empty: all neurons in graph order


@dataclass(frozen=True)
class Graph:
    """Validated graph in CSR form, orientation version 1 (WRL_SYNTAX.md §6.2).

    Chemical edges are grouped by postsynaptic neuron (row ``i`` lists the inputs of neuron ``i``), sorted by
    ``(pre, delay, sign, weight)``; gap junctions appear in both endpoint rows, sorted by ``(neighbour, g)``.
    """

    ids: tuple[str, ...]
    types: tuple[str, ...]
    in_offsets: tuple[int, ...]
    pre: tuple[int, ...]
    weight: tuple[float, ...]
    sign: tuple[int, ...]
    delay: tuple[int, ...]
    gap_offsets: tuple[int, ...]
    gap_neighbor: tuple[int, ...]
    gap_conductance: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def max_edge_delay(self) -> int:
        return max(self.delay, default=0)

    def index_of(self, neuron: str) -> int | None:
        try:
            return self.ids.index(neuron)
        except ValueError:
            return None


def _valid_name(name: object) -> bool:
    return isinstance(name, str) and _NAME.fullmatch(name) is not None


def _num(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(f"{what} must be a number")
    return float(value)


def _int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{what} must be an integer")
    return value


def graph_spec_from_json(data: Mapping[str, Any]) -> GraphSpec:
    try:
        neurons = tuple((str(n["id"]), str(n.get("type", "generic"))) for n in data["neurons"])
        chemical = tuple(
            ChemicalEdge(
                str(e["pre"]),
                str(e["post"]),
                _num(e.get("weight", 1.0), "weight"),
                _int(e.get("sign", 1), "sign"),
                _int(e.get("delay", 0), "delay"),
            )
            for e in data.get("chemical", [])
        )
        gap = tuple(GapJunction(str(e["a"]), str(e["b"]), _num(e["g"], "g")) for e in data.get("gap", []))
    except (KeyError, TypeError) as error:
        raise InputError(f"malformed graph description: {error!r}") from error
    return GraphSpec(neurons, chemical, gap)


def delete_chemical(spec: GraphSpec, pre: str, post: str) -> GraphSpec:
    ids = {i for i, _ in spec.neurons}
    if pre not in ids or post not in ids:
        raise InputError("delete_chemical names an unknown neuron")
    return replace(spec, chemical=tuple(e for e in spec.chemical if not (e.pre == pre and e.post == post)))


def delete_gap(spec: GraphSpec, a: str, b: str) -> GraphSpec:
    ids = {i for i, _ in spec.neurons}
    if a not in ids or b not in ids:
        raise InputError("delete_gap names an unknown neuron")
    return replace(spec, gap=tuple(e for e in spec.gap if {e.a, e.b} != {a, b}))


def build_graph(spec: GraphSpec) -> Graph:
    index: dict[str, int] = {}
    ids: list[str] = []
    types: list[str] = []
    for nid, ntype in spec.neurons:
        if not _valid_name(nid):
            raise InputError(f"invalid neuron id '{nid}'")
        if not _valid_name(ntype):
            raise InputError(f"invalid neuron type '{ntype}'")
        if nid in index:
            raise InputError(f"duplicate neuron id '{nid}'")
        index[nid] = len(ids)
        ids.append(nid)
        types.append(ntype)
    n = len(ids)

    def lookup(nid: str) -> int:
        if nid not in index:
            raise InputError(f"edge references unknown neuron '{nid}'")
        return index[nid]

    chem: list[tuple[int, int, int, int, float]] = []  # post, pre, delay, sign, weight
    for e in spec.chemical:
        if not math.isfinite(e.weight) or e.weight < 0.0:
            raise InputError("chemical weight must be finite and non-negative")
        if e.sign not in (1, -1):
            raise InputError("chemical sign must be +1 or -1")
        if not 0 <= e.delay <= MAX_EDGE_DELAY_TICKS:
            raise InputError(f"chemical delay must be an integer in [0, {MAX_EDGE_DELAY_TICKS}]")
        chem.append((lookup(e.post), lookup(e.pre), e.delay, e.sign, e.weight))
    chem.sort()
    in_offsets = [0] * (n + 1)
    for post, *_ in chem:
        in_offsets[post + 1] += 1
    for i in range(n):
        in_offsets[i + 1] += in_offsets[i]

    gap: list[tuple[int, int, float]] = []  # row, neighbour, conductance
    for j in spec.gap:
        if not math.isfinite(j.g) or j.g < 0.0:
            raise InputError("gap conductance must be finite and non-negative")
        a, b = lookup(j.a), lookup(j.b)
        if a == b:
            raise InputError("gap junction cannot connect a neuron to itself")
        gap.append((a, b, j.g))
        gap.append((b, a, j.g))
    gap.sort()
    gap_offsets = [0] * (n + 1)
    for row, *_ in gap:
        gap_offsets[row + 1] += 1
    for i in range(n):
        gap_offsets[i + 1] += gap_offsets[i]

    return Graph(
        ids=tuple(ids),
        types=tuple(types),
        in_offsets=tuple(in_offsets),
        pre=tuple(c[1] for c in chem),
        weight=tuple(c[4] for c in chem),
        sign=tuple(c[3] for c in chem),
        delay=tuple(c[2] for c in chem),
        gap_offsets=tuple(gap_offsets),
        gap_neighbor=tuple(g[1] for g in gap),
        gap_conductance=tuple(g[2] for g in gap),
    )


def sim_input_from_json(data: Mapping[str, Any]) -> SimInput:
    """Parse the simulation input of §6.1 (edge ``edits`` are applied to the graph)."""
    try:
        spec = graph_spec_from_json(data["graph"])
        edits = data.get("edits", {})
        for pre, post in edits.get("delete_chemical", []):
            spec = delete_chemical(spec, str(pre), str(post))
        for a, b in edits.get("delete_gap", []):
            spec = delete_gap(spec, str(a), str(b))
        dt = _num(data["dt"], "dt")
        n_steps = _int(data["n_steps"], "n_steps")
        if "sample_ticks" in data:
            ticks = tuple(_int(t, "sample tick") for t in data["sample_ticks"])
        else:
            ticks = tuple(range(n_steps + 1)) if 0 <= n_steps <= MAX_STEPS else ()
        events = tuple(
            StimulusEvent(
                str(s["neuron"]), _int(s["start"], "start"), _int(s["end"], "end"), _num(s["amplitude"], "amplitude")
            )
            for s in data.get("stimulus", [])
        )
        params = {str(k): _num(v, f"parameter {k}") for k, v in data.get("params", {}).items()}
        initial: dict[str, dict[str, float]] = {}
        for reg, values in data.get("initial_state", {}).items():
            if isinstance(values, Mapping):
                initial[str(reg)] = {str(k): _num(v, "initial state") for k, v in values.items()}
            else:
                if len(values) != len(spec.neurons):
                    raise InputError(f"dense initial_state for '{reg}' must list every neuron")
                initial[str(reg)] = {
                    nid: _num(v, "initial state") for (nid, _), v in zip(spec.neurons, values, strict=True)
                }
        observed = tuple(str(o) for o in data.get("observed", []))
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, InputError):
            raise
        raise InputError(f"malformed simulation input: {error!r}") from error
    return SimInput(dt, n_steps, spec, ticks, events, params, initial, observed)


def validate_run(sim_input: SimInput, dt_max: float | None) -> None:
    """The checks of ``Simulator::validate`` (dt, step count and sample ticks)."""
    if not math.isfinite(sim_input.dt) or not sim_input.dt > 0.0:
        raise InputError("dt must be a positive finite number")
    if dt_max is not None and sim_input.dt > dt_max:
        raise InputError("dt exceeds the program's declared dt_max")
    if not 0 <= sim_input.n_steps <= MAX_STEPS:
        raise InputError(f"n_steps must be in [0, {MAX_STEPS}]")
    previous = -1
    for tick in sim_input.sample_ticks:
        if tick <= previous or tick > sim_input.n_steps:
            raise InputError("sample_ticks must be strictly increasing and within [0, n_steps]")
        previous = tick


def validate_stimulus(events: Sequence[StimulusEvent], graph: Graph) -> None:
    for e in events:
        if graph.index_of(e.neuron) is None:
            raise InputError(f"stimulus names unknown neuron '{e.neuron}'")
        if e.start < 0 or e.end < e.start or not math.isfinite(e.amplitude):
            raise InputError("stimulus needs 0 <= start <= end and a finite amplitude")


def permute_neurons(sim_input: SimInput, permutation: Sequence[int]) -> SimInput:
    """New position ``j`` holds old neuron ``permutation[j]`` (the ``permutation`` conformance check)."""
    neurons = tuple(sim_input.graph.neurons[old] for old in permutation)
    return replace(sim_input, graph=replace(sim_input.graph, neurons=neurons))
