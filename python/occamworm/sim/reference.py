"""Independent scalar interpreter of the canonical WRL IR (WRL_SYNTAX.md §6, runtime oracle of §6.8).

Deliberately plain: Python floats, ``math`` and explicit loops over neurons, edges and instructions, no numpy and
no JAX, so that it shares no code with ``jaxsim`` and is an honest second oracle next to the C++ interpreter.
It follows the reference summation order (rows sorted as in §6.2, sums accumulated sequentially from ``0.0``).
History is kept as the full list of committed ticks, ``H_r[j](s) = x_r[j](s)`` for ``s >= 0`` and ``x_r[j](0)``
for ``s < 0``, rather than a ring: it realises the §6.5 definition directly.
"""

from __future__ import annotations

import math

from occamworm.sim.graph import (
    Graph,
    InputError,
    SimInput,
    SimulationError,
    build_graph,
    validate_run,
    validate_stimulus,
)
from occamworm.sim.ir import Instruction, Program
from occamworm.sim.result import Matrix, SimResult


def _sigmoid(x: float) -> float:
    try:
        e = math.exp(-x)
    except OverflowError:
        e = math.inf
    return 1.0 / (1.0 + e)


def _lut(x: float, table: list[float]) -> float:
    floored = math.floor(x)
    n = len(table)
    if floored <= 0:
        return table[0]
    if floored >= n - 1:
        return table[n - 1]
    return table[floored]


def _leaky(x: float, target: float, tau: float, dt: float) -> float:
    return target + (x - target) * math.exp(-(dt / tau))


def _euler(x: float, target: float, tau: float, dt: float) -> float:
    return x + (dt / tau) * (target - x)


def _initial_registers(program: Program, graph: Graph, sim_input: SimInput) -> list[list[float]]:
    state = [[reg.init] * graph.n for reg in program.registers]
    for name, values in sim_input.initial_state.items():
        r = program.register_index(name)
        if r is None:
            if name in program.eliminated_registers:
                continue
            raise InputError(f"initial_state names unknown register '{name}'")
        for neuron, value in values.items():
            i = graph.index_of(neuron)
            if i is None:
                raise InputError(f"initial_state names unknown neuron '{neuron}'")
            if not math.isfinite(value):
                raise InputError("initial_state values must be finite")
            state[r][i] = value
    return state


def simulate(program: Program, sim_input: SimInput) -> SimResult:
    """Run ``sim_input`` through ``program`` with the scalar semantics of §6.3-§6.7."""
    validate_run(sim_input, program.dt_max)
    graph = build_graph(sim_input.graph)
    validate_stimulus(sim_input.stimulus, graph)
    try:
        theta = program.theta_from_dict(sim_input.params)
    except ValueError as error:
        raise InputError(str(error)) from error
    reported: list[int] = []
    if sim_input.observed:
        for neuron in sim_input.observed:
            idx = graph.index_of(neuron)
            if idx is None:
                raise InputError(f"observed names unknown neuron '{neuron}'")
            reported.append(idx)
    else:
        reported = list(range(graph.n))

    n = graph.n
    dt = sim_input.dt
    n_regs = len(program.registers)
    instrs = program.instructions
    state = _initial_registers(program, graph, sim_input)
    history: list[list[list[float]]] = [[list(state[r])] for r in range(n_regs)]  # history[r][tick][neuron]
    obs_reg = program.observation.register
    obs = list(state[obs_reg])
    calcium_tau = 0.0
    if program.observation.operator == "calcium_linear_v1":
        if program.observation.tau_param is not None:
            calcium_tau = theta[program.observation.tau_param]
        else:
            assert program.observation.tau_const is not None
            calcium_tau = program.observation.tau_const
    gap_scale = 1.0
    if program.gap is not None and program.gap.scale_param is not None:
        gap_scale = theta[program.gap.scale_param]

    def past(r: int, j: int, d: int, t: int) -> float:
        """``H_r[j](t - d)``."""
        s = t - d
        return history[r][0][j] if s < 0 else history[r][s][j]

    def evaluate(ins: Instruction, i: int, t: int, values: list[float], u: list[float]) -> float:
        op = ins.op
        args = ins.args
        if op == "const":
            return float(ins.attrs["value"])
        if op == "param":
            return theta[int(ins.attrs["param"])]
        if op == "state":
            return state[int(ins.attrs["register"])][i]
        if op == "stimulus":
            return u[i]
        if op == "type_mask":
            return 1.0 if graph.types[i] == ins.attrs["type"] else 0.0
        if op == "sum_in":
            r = int(ins.attrs["register"])
            select = ins.attrs["select"]
            acc = 0.0
            if select == "mod":
                for e in range(graph.mod_offsets[i], graph.mod_offsets[i + 1]):
                    acc += graph.mod_weight[e] * past(r, graph.mod_pre[e], 0, t)
                return acc
            for e in range(graph.in_offsets[i], graph.in_offsets[i + 1]):
                term = graph.weight[e] * past(r, graph.pre[e], graph.delay[e], t)
                excitatory = graph.sign[e] > 0
                if select == "exc":
                    if excitatory:
                        acc += term
                elif select == "inh":
                    if not excitatory:
                        acc += term
                else:
                    acc += term if excitatory else -term
            return acc
        if op == "count_in":
            r = int(ins.attrs["register"])
            k = float(ins.attrs["k"])
            count = 0.0
            for e in range(graph.in_offsets[i], graph.in_offsets[i + 1]):
                if past(r, graph.pre[e], graph.delay[e], t) == k:
                    count += 1.0
            return count
        if op == "delay":
            return past(int(ins.attrs["register"]), i, int(ins.attrs["ticks"]), t)
        if op == "add":
            acc = values[args[0]]
            for a in args[1:]:
                acc = acc + values[a]
            return acc
        if op == "mul":
            acc = values[args[0]]
            for a in args[1:]:
                acc = acc * values[a]
            return acc
        x = values[args[0]] if args else 0.0
        if op == "neg":
            return -x
        if op == "abs":
            return math.fabs(x)
        if op in ("min", "max"):
            acc = x
            for a in args[1:]:
                v = values[a]
                if (v < acc) if op == "min" else (v > acc):
                    acc = v
            return acc
        if op == "clamp":
            lo, hi = values[args[1]], values[args[2]]
            m = lo if lo > x else x
            return hi if hi < m else m
        if op == "relu":
            return x if x > 0.0 else 0.0
        if op == "tanh":
            return math.tanh(x)
        if op == "sigmoid":
            return _sigmoid(x)
        if op == "threshold":
            return 1.0 if x >= values[args[1]] else 0.0
        if op == "select":
            return values[args[1]] if x != 0.0 else values[args[2]]
        if op == "lut":
            return _lut(x, [float(v) for v in ins.attrs["table"]])
        if op == "leaky_integrate":
            return _leaky(x, values[args[1]], values[args[2]], dt)
        if op == "euler_leak":
            return _euler(x, values[args[1]], values[args[2]], dt)
        raise SimulationError(f"unhandled operator '{op}'")

    events: list[tuple[int, int, int, float]] = []
    for event in sim_input.stimulus:
        target = graph.index_of(event.neuron)
        assert target is not None  # validate_stimulus
        events.append((target, event.start, event.end, event.amplitude))
    samples = set(sim_input.sample_ticks)
    out_ticks: list[int] = []
    out_regs: list[Matrix] = [[] for _ in range(n_regs)]
    out_obs: Matrix = []

    def record(tick: int) -> None:
        if tick in samples:
            out_ticks.append(tick)
            for r in range(n_regs):
                out_regs[r].append([state[r][i] for i in reported])
            out_obs.append([obs[i] for i in reported])

    record(0)
    for t in range(sim_input.n_steps):
        u = [0.0] * n  # step 1: stimulus
        for target, start, end, amplitude in events:
            if start <= t < end:
                u[target] += amplitude
        nxt = [[0.0] * n for _ in range(n_regs)]
        values = [0.0] * len(instrs)
        for i in range(n):  # steps 2, 3, 5: every read sees tick-t state and history only
            for k, ins in enumerate(instrs):
                v = evaluate(ins, i, t, values, u)
                if not math.isfinite(v):
                    raise SimulationError(f"non-finite value at tick {t}, neuron '{graph.ids[i]}', instruction n{k}")
                values[k] = v
            for r in range(n_regs):
                nxt[r][i] = values[program.writes[r]]
        if program.gap is not None:  # step 4 + gap part of step 5
            g = program.gap.register
            old = state[g]
            for i in range(n):
                total = 0.0
                weighted = 0.0
                for e in range(graph.gap_offsets[i], graph.gap_offsets[i + 1]):
                    c = gap_scale * graph.gap_conductance[e]
                    total += c
                    weighted += c * old[graph.gap_neighbor[e]]
                nxt[g][i] = (nxt[g][i] + dt * weighted) / (1.0 + dt * total)
                if not math.isfinite(nxt[g][i]):
                    raise SimulationError(f"non-finite value after gap coupling at tick {t}")
        for i in range(n):  # step 6: observation from the post-update register
            if program.observation.operator == "identity_v1":
                obs[i] = nxt[obs_reg][i]
            else:
                obs[i] = _leaky(obs[i], nxt[obs_reg][i], calcium_tau, dt)
        for r in range(n_regs):  # step 7: commit
            state[r] = nxt[r]
            history[r].append(list(nxt[r]))
        record(t + 1)  # step 8

    return SimResult(
        sample_ticks=out_ticks,
        neurons=[graph.ids[i] for i in reported],
        registers={reg.source_name: out_regs[reg.index] for reg in program.registers},
        observation=out_obs,
        observation_operator=program.observation.operator,
        observation_register=program.registers[obs_reg].source_name,
    )
