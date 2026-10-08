"""Differentiable JAX simulator of the canonical WRL IR, float64, identical semantics to the scalar interpreters.

Time is a ``jax.lax.scan``; neurons are vectorised; chemical edges are an edge list (CSR order of §6.2) gathered
through a history ring of ``D + 1`` slots per register and reduced with ``segment_sum`` (a sequential scatter-add
on CPU, so sums keep the reference row order). Gap junctions use the node-local semi-implicit step of §6.6.
Trials, stimuli, initial states and parameter vectors can be batched with ``jax.vmap``; everything is
differentiable with respect to the parameter vector ``theta`` (IR parameter order), the dense stimulus array and
the initial state. Operators with a discrete choice (``threshold``, ``select``, ``lut``, ``min``/``max`` ties,
``relu`` at 0) have the almost-everywhere gradient of the selected branch, as usual for ``jnp.where``.

Dense inputs: ``theta`` ``(P,)``, ``stimulus`` ``(T, N)`` with ``T = n_steps`` (see :func:`stimulus_array`) and
``init`` ``(R, N)`` (see :func:`initial_array`). Outputs: ``registers`` ``(R, S, K)`` and ``observation``
``(S, K)`` for ``S`` sample ticks and ``K`` reported neurons; a leading batch axis is added when any input is
batched (``theta`` ``(B, P)``, ``stimulus`` ``(B, T, N)``, ``init`` ``(B, R, N)``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from occamworm.sim.graph import (
    Graph,
    InputError,
    SimInput,
    SimulationError,
    StimulusEvent,
    build_graph,
    validate_run,
    validate_run_settings,
    validate_stimulus,
)
from occamworm.sim.ir import Instruction, Program
from occamworm.sim.result import SimResult

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

Array = jax.Array
FloatArray = npt.NDArray[np.float64]


class SimOutput(NamedTuple):
    registers: Array  # (R, S, K), keyed by IR register index
    observation: Array  # (S, K)


def stimulus_array(events: Sequence[StimulusEvent], graph: Graph, n_steps: int) -> FloatArray:
    """Dense ``(n_steps, N)`` stimulus: events add their amplitude on ``start <= t < end`` (§6.3 step 1)."""
    validate_stimulus(events, graph)
    u = np.zeros((n_steps, graph.n), dtype=np.float64)
    for e in events:
        idx = graph.index_of(e.neuron)
        assert idx is not None
        u[e.start : min(e.end, n_steps), idx] += e.amplitude
    return u


def initial_array(
    program: Program, graph: Graph, initial_state: Mapping[str, Mapping[str, float]] | None = None
) -> FloatArray:
    """Initial registers ``(R, N)``: declared ``init`` values with the sparse per-neuron overrides applied."""
    x = np.empty((len(program.registers), graph.n), dtype=np.float64)
    for reg in program.registers:
        x[reg.index, :] = reg.init
    for name, values in (initial_state or {}).items():
        r = program.register_index(name)
        if r is None:
            if name in program.eliminated_registers:
                continue
            raise InputError(f"initial_state names unknown register '{name}'")
        for neuron, value in values.items():
            i = graph.index_of(neuron)
            if i is None:
                raise InputError(f"initial_state names unknown neuron '{neuron}'")
            if not np.isfinite(value):
                raise InputError("initial_state values must be finite")
            x[r, i] = value
    return x


def _edge_rows(offsets: Sequence[int]) -> npt.NDArray[np.int32]:
    return np.repeat(np.arange(len(offsets) - 1, dtype=np.int32), np.diff(np.asarray(offsets)))


class JaxSimulator:
    """A jitted simulator for one program and one graph; ``dt``, the step count and sampling are static."""

    def __init__(
        self,
        program: Program,
        graph: Graph,
        dt: float,
        n_steps: int,
        observed: Sequence[int] | None = None,
        sample_ticks: Sequence[int] | None = None,
    ) -> None:
        self.program = program
        self.graph = graph
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.n = graph.n
        self.n_registers = len(program.registers)
        self.reported = np.asarray(range(graph.n) if observed is None else observed, dtype=np.int32)
        ticks = range(self.n_steps + 1) if sample_ticks is None else sample_ticks
        self.sample_ticks = np.asarray(list(ticks), dtype=np.int32)
        validate_run_settings(self.dt, self.n_steps, [int(t) for t in self.sample_ticks], program.dt_max)
        self.ring_size = max(graph.max_edge_delay, program.max_delay_ticks) + 1

        post = _edge_rows(graph.in_offsets)
        pre = np.asarray(graph.pre, dtype=np.int32)
        weight = np.asarray(graph.weight, dtype=np.float64)
        sign = np.asarray(graph.sign, dtype=np.int32)
        delay = np.asarray(graph.delay, dtype=np.int32)
        self._edges: dict[str, tuple[Array, Array, Array, Array]] = {}
        for select, keep in (("exc", sign > 0), ("inh", sign < 0), ("all", np.ones_like(sign, dtype=bool))):
            w = weight[keep] * (np.where(sign[keep] > 0, 1.0, -1.0) if select == "all" else 1.0)
            self._edges[select] = (
                jnp.asarray(pre[keep]),
                jnp.asarray(post[keep]),
                jnp.asarray(w),
                jnp.asarray(delay[keep]),
            )
        mod_pre = np.asarray(graph.mod_pre, dtype=np.int32)
        self._edges["mod"] = (
            jnp.asarray(mod_pre),
            jnp.asarray(_edge_rows(graph.mod_offsets) if graph.mod_offsets else np.zeros(0, dtype=np.int32)),
            jnp.asarray(np.asarray(graph.mod_weight, dtype=np.float64)),
            jnp.zeros(mod_pre.shape[0], dtype=jnp.int32),
        )
        self._all_edges = (jnp.asarray(pre), jnp.asarray(post), jnp.asarray(delay))
        self._gap_row = jnp.asarray(_edge_rows(graph.gap_offsets))
        self._gap_nbr = jnp.asarray(np.asarray(graph.gap_neighbor, dtype=np.int32))
        self._gap_g = jnp.asarray(np.asarray(graph.gap_conductance, dtype=np.float64))
        self._type_masks: dict[str, Array] = {}
        for ins in program.instructions:
            if ins.op == "type_mask":
                name = str(ins.attrs["type"])
                self._type_masks[name] = jnp.asarray(np.asarray([t == name for t in graph.types], dtype=np.float64))
        self._jit_cache: dict[tuple[Any, ...], Callable[..., Any]] = {}

    # -- one step ------------------------------------------------------------------------------------------------

    def _eval(self, ins: Instruction, values: list[Array], theta: Array, ring: Array, t: Array, u: Array) -> Array:
        op, args = ins.op, ins.args
        n, size = self.n, self.ring_size
        if op == "const":
            return jnp.asarray(float(ins.attrs["value"]), dtype=jnp.float64)
        if op == "param":
            return theta[int(ins.attrs["param"])]
        if op == "state":
            return ring[int(ins.attrs["register"]), t % size]
        if op == "stimulus":
            return u
        if op == "type_mask":
            return self._type_masks[str(ins.attrs["type"])]
        if op == "sum_in":
            r = int(ins.attrs["register"])
            pre, post, w, delay = self._edges[str(ins.attrs["select"])]
            if pre.shape[0] == 0:
                return jnp.zeros(n)
            hist = ring[r, (t - delay) % size, pre]
            return jax.ops.segment_sum(w * hist, post, num_segments=n, indices_are_sorted=True)
        if op == "count_in":
            r = int(ins.attrs["register"])
            pre, post, delay = self._all_edges
            if pre.shape[0] == 0:
                return jnp.zeros(n)
            hist = ring[r, (t - delay) % size, pre]
            hit = jnp.where(hist == float(ins.attrs["k"]), 1.0, 0.0)
            return jax.ops.segment_sum(hit, post, num_segments=n, indices_are_sorted=True)
        if op == "delay":
            return ring[int(ins.attrs["register"]), (t - int(ins.attrs["ticks"])) % size]
        a = [values[k] for k in args]
        if op == "add":
            acc = a[0]
            for v in a[1:]:
                acc = acc + v
            return acc
        if op == "mul":
            acc = a[0]
            for v in a[1:]:
                acc = acc * v
            return acc
        if op == "neg":
            return -a[0]
        if op == "abs":
            return jnp.abs(a[0])
        if op == "min":
            acc = a[0]
            for v in a[1:]:
                acc = jnp.where(v < acc, v, acc)
            return acc
        if op == "max":
            acc = a[0]
            for v in a[1:]:
                acc = jnp.where(v > acc, v, acc)
            return acc
        if op == "clamp":
            m = jnp.where(a[1] > a[0], a[1], a[0])
            return jnp.where(a[2] < m, a[2], m)
        if op == "relu":
            return jnp.where(a[0] > 0.0, a[0], 0.0)
        if op == "tanh":
            return jnp.tanh(a[0])
        if op == "sigmoid":
            return 1.0 / (1.0 + jnp.exp(-a[0]))
        if op == "threshold":
            return jnp.where(a[0] >= a[1], 1.0, 0.0)
        if op == "select":
            return jnp.where(a[0] != 0.0, a[1], a[2])
        if op == "lut":
            table = jnp.asarray([float(v) for v in ins.attrs["table"]], dtype=jnp.float64)
            index = jnp.clip(jnp.floor(a[0]), 0, table.shape[0] - 1).astype(jnp.int32)
            return table[index]
        if op == "leaky_integrate":
            return a[1] + (a[0] - a[1]) * jnp.exp(-(self.dt / a[2]))
        if op == "euler_leak":
            return a[0] + (self.dt / a[2]) * (a[1] - a[0])
        raise SimulationError(f"unhandled operator '{op}'")

    def _step(self, theta: Array, record: bool) -> Callable[[tuple[Array, Array], tuple[Array, Array]], Any]:
        prog = self.program
        n, size, dt = self.n, self.ring_size, self.dt
        reported = jnp.asarray(self.reported)
        calcium = prog.observation.operator == "calcium_linear_v1"
        if calcium:
            tau_obs = (
                theta[prog.observation.tau_param]
                if prog.observation.tau_param is not None
                else jnp.asarray(float(prog.observation.tau_const or 0.0), dtype=jnp.float64)
            )
        gap_scale = theta[prog.gap.scale_param] if prog.gap is not None and prog.gap.scale_param is not None else 1.0

        def step(carry: tuple[Array, Array], xs: tuple[Array, Array]) -> tuple[tuple[Array, Array], Any]:
            ring, obs = carry
            t, u = xs
            values: list[Array] = []
            for ins in prog.instructions:
                values.append(self._eval(ins, values, theta, ring, t, u))
            new = jnp.stack([jnp.broadcast_to(values[w], (n,)) for w in prog.writes])
            if prog.gap is not None:
                g = prog.gap.register
                old = ring[g, t % size]
                c = gap_scale * self._gap_g
                total = jax.ops.segment_sum(c, self._gap_row, num_segments=n, indices_are_sorted=True)
                weighted = jax.ops.segment_sum(
                    c * old[self._gap_nbr], self._gap_row, num_segments=n, indices_are_sorted=True
                )
                new = new.at[g].set((new[g] + dt * weighted) / (1.0 + dt * total))
            x_obs = new[prog.observation.register]
            obs = x_obs if not calcium else x_obs + (obs - x_obs) * jnp.exp(-(dt / tau_obs))
            ring = ring.at[:, (t + 1) % size].set(new)
            out = (new[:, reported], obs[reported]) if record else obs[reported]
            return (ring, obs), out

        return step

    def _run_single(self, theta: Array, stimulus: Array, init: Array, record: bool) -> Any:
        prog = self.program
        ring0 = jnp.broadcast_to(init[:, None, :], (self.n_registers, self.ring_size, self.n))
        obs0 = init[prog.observation.register]
        ticks = jnp.arange(self.n_steps, dtype=jnp.int32)
        _, ys = jax.lax.scan(self._step(theta, record), (ring0, obs0), (ticks, stimulus))
        keep = jnp.asarray(self.sample_ticks)
        reported = jnp.asarray(self.reported)
        if record:
            regs, obs = ys
            regs = jnp.concatenate([init[:, reported][None], regs], axis=0)[keep]  # (S, R, K)
            obs = jnp.concatenate([obs0[reported][None], obs], axis=0)[keep]
            return SimOutput(jnp.transpose(regs, (1, 0, 2)), obs)
        return jnp.concatenate([obs0[reported][None], ys], axis=0)[keep]

    # -- public API ----------------------------------------------------------------------------------------------

    def _compiled(self, theta: Array, stimulus: Array, init: Array, record: bool) -> Callable[..., Any]:
        axes = (0 if theta.ndim == 2 else None, 0 if stimulus.ndim == 3 else None, 0 if init.ndim == 3 else None)
        key = (axes, record)
        if key not in self._jit_cache:

            def single(th: Array, st: Array, i0: Array) -> Any:
                return self._run_single(th, st, i0, record)

            fn: Callable[..., Any] = single if axes == (None, None, None) else jax.vmap(single, in_axes=axes)
            self._jit_cache[key] = jax.jit(fn)
        return self._jit_cache[key]

    def _prepare(self, theta: Any, stimulus: Any, init: Any) -> tuple[Array, Array, Array]:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        stimulus = jnp.asarray(stimulus, dtype=jnp.float64)
        if init is None:
            init = np.asarray([[reg.init] * self.n for reg in self.program.registers], dtype=np.float64)
        init = jnp.asarray(init, dtype=jnp.float64)
        if stimulus.shape[-2:] != (self.n_steps, self.n):
            raise ValueError(f"stimulus must have trailing shape ({self.n_steps}, {self.n}), got {stimulus.shape}")
        if init.shape[-2:] != (self.n_registers, self.n):
            raise ValueError(f"init must have trailing shape ({self.n_registers}, {self.n}), got {init.shape}")
        if theta.shape[-1:] != (len(self.program.parameters),):
            raise ValueError(f"theta must have {len(self.program.parameters)} entries, got {theta.shape}")
        return theta, stimulus, init

    def run(self, theta: Any, stimulus: Any, init: Any = None) -> SimOutput:
        """Registers and observation; any of ``theta``, ``stimulus`` and ``init`` may carry a leading batch axis."""
        theta, stimulus, init = self._prepare(theta, stimulus, init)
        out: SimOutput = self._compiled(theta, stimulus, init, True)(theta, stimulus, init)
        return out

    def observe(self, theta: Any, stimulus: Any, init: Any = None) -> Array:
        """Observation only, ``(S, K)`` (``(B, S, K)`` batched). The cheap entry point for fitting."""
        theta, stimulus, init = self._prepare(theta, stimulus, init)
        out: Array = self._compiled(theta, stimulus, init, False)(theta, stimulus, init)
        return out


def make_simulator(program: Program, sim_input: SimInput) -> tuple[JaxSimulator, FloatArray, FloatArray, FloatArray]:
    """Build a simulator from a §6.1 input; returns ``(sim, theta, stimulus, init)`` ready for ``sim.run``."""
    validate_run(sim_input, program.dt_max)
    graph = build_graph(sim_input.graph)
    reported = []
    for neuron in sim_input.observed:
        idx = graph.index_of(neuron)
        if idx is None:
            raise InputError(f"observed names unknown neuron '{neuron}'")
        reported.append(idx)
    sim = JaxSimulator(program, graph, sim_input.dt, sim_input.n_steps, reported or None, list(sim_input.sample_ticks))
    try:
        theta = np.asarray(program.theta_from_dict(sim_input.params), dtype=np.float64)
    except ValueError as error:
        raise InputError(str(error)) from error
    stimulus = stimulus_array(sim_input.stimulus, graph, sim_input.n_steps)
    init = initial_array(program, graph, sim_input.initial_state)
    return sim, theta, stimulus, init


def simulate(program: Program, sim_input: SimInput) -> SimResult:
    """Drop-in counterpart of ``reference.simulate`` (same input, same :class:`SimResult`)."""
    sim, theta, stimulus, init = make_simulator(program, sim_input)
    out = sim.run(theta, stimulus, init)
    regs = np.asarray(out.registers)
    obs = np.asarray(out.observation)
    if not (np.all(np.isfinite(regs)) and np.all(np.isfinite(obs))):
        raise SimulationError("non-finite value in the simulated trajectory")
    return SimResult(
        sample_ticks=[int(t) for t in sim.sample_ticks],
        neurons=[sim.graph.ids[i] for i in sim.reported],
        registers={reg.source_name: regs[reg.index].tolist() for reg in program.registers},
        observation=obs.tolist(),
        observation_operator=program.observation.operator,
        observation_register=program.registers[program.observation.register].source_name,
    )
