"""WRL programs as nested-search candidates (§7.3, §7.4, §7.8; OW-012).

A program runs on one declared connectome (``cook2019-herm``; chemical signs from ``fenyves2020-sign-prediction``,
edges with an unknown or missing sign taken as excitatory and counted in the metadata) through the differentiable
JAX simulator (OW-009). Its linear response to a small pulse into each target gives the pair kernels on the shared
tent basis, so a program is fitted and scored by exactly the machinery of the B0-B4 baselines:

    c_ij[m] = (r_ij((knot_m + 1) dt; +a) - r_ij((knot_m + 1) dt; 0)) / a

where ``r_ij(t; a)`` is the observed output of responder ``i`` after a pulse of amplitude ``a`` held over the first
volume into target ``j``. A measured sample is read as the end of its one-volume hold, hence the ``+1``. This is
the declared linear-response approximation of a possibly nonlinear program. Trainable parameters are fitted by
L-BFGS-B through the bounded transforms of ``occamworm.fit`` on the profiled pair-statistics loss (the drift and
history coefficients are solved in closed form for each parameter vector), with a few seeded starts.

A program is gated before any fit: it must read the stimulus, couple neurons (``sum_in`` or a gap term) and stay
finite and bounded at its declared parameters. Gated programs are counted in the search log. Because kernels come
from the graph, ``score`` predicts every mappable held-out pair, including pairs never seen in training.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
from scipy import optimize

from occamworm.baselines.data import KERNEL_KNOTS, M, PairStats, TaskData, aggregate, compute_stats, trial_designs
from occamworm.baselines.evaluate import noise_for, rows_of, score_rows
from occamworm.baselines.kernels import EPS, Fit, loss
from occamworm.baselines.linear_network import RECONSTRUCTION, label_map
from occamworm.fit.accounting import l_params_bits
from occamworm.fit.transforms import ParamTransforms
from occamworm.search.candidates import CandidateRejected, InnerResult, _indicator, score_kernels
from occamworm.search.views import TrainingView
from occamworm.sim.graph import ChemicalEdge, GapJunction, Graph, GraphSpec, build_graph
from occamworm.sim.ir import Program, compile_source, run_ow
from occamworm.sim.jaxsim import JaxSimulator

FloatArray = npt.NDArray[np.float64]
SIGN_SOURCE = "fenyves2020-sign-prediction"
PULSE = 0.05  # amplitude of the probing pulse, in stimulus units
RESPONSE_BOUND = 1e3  # |response| above this at declared parameters marks a program unstable
MAX_ITER = 100


class GateError(CandidateRejected):
    """The program is rejected before fitting (inert, uncoupled or unstable)."""


@dataclass
class Connectome:
    spec: GraphSpec
    graph: Graph
    meta: dict[str, Any]


def load_connectome(root: Path, reconstruction: str = RECONSTRUCTION) -> Connectome:
    """Chemical and gap edges of one reconstruction, weights scaled to unit max row sum (as in B4)."""
    ann = root / "data" / "normalized" / "annotations-v1"
    neurons = sorted(
        str(x) for x in pq.read_table(ann / "neurons.parquet", columns=["neuron_id"]).column(0).to_pylist()
    )
    known = set(neurons)
    edges = pq.read_table(ann / "edges.parquet").to_pylist()
    signs = {
        (e["source_neuron_id"], e["target_neuron_id"]): e["sign"]
        for e in edges
        if e["source_reconstruction"] == SIGN_SOURCE
    }
    chem: dict[tuple[str, str], float] = {}
    gap: dict[tuple[str, str], float] = {}
    for e in edges:
        s, t = e["source_neuron_id"], e["target_neuron_id"]
        if e["source_reconstruction"] != reconstruction or s == t or s not in known or t not in known:
            continue
        w = float(e["synapse_count"] or 0.0)
        if w <= 0:
            continue
        if e["edge_kind"] == "chem":
            chem[(s, t)] = chem.get((s, t), 0.0) + w
        elif e["edge_kind"] == "gap":
            key = (min(s, t), max(s, t))
            gap[key] = max(gap.get(key, 0.0), w)
    row: dict[str, float] = {}
    for (_, t), w in chem.items():
        row[t] = row.get(t, 0.0) + w
    chem_scale = max(row.values(), default=1.0)
    grow: dict[str, float] = {}
    for (a, b), w in gap.items():
        grow[a] = grow.get(a, 0.0) + w
        grow[b] = grow.get(b, 0.0) + w
    gap_scale = max(grow.values(), default=1.0)
    counts = {"excitatory": 0, "inhibitory": 0, "unknown_as_excitatory": 0}
    chemical = []
    for (s, t), w in sorted(chem.items()):
        sign = signs.get((s, t))
        counts[
            "inhibitory" if sign == "inhibitory" else "excitatory" if sign == "excitatory" else "unknown_as_excitatory"
        ] += 1
        chemical.append(ChemicalEdge(s, t, w / chem_scale, -1 if sign == "inhibitory" else 1))
    gaps = tuple(GapJunction(a, b, w / gap_scale) for (a, b), w in sorted(gap.items()))
    spec = GraphSpec(tuple((n, "neuron") for n in neurons), tuple(chemical), gaps)
    meta = {
        "reconstruction": reconstruction,
        "sign_source": SIGN_SOURCE,
        "chemical_edges": counts,
        "gap_junctions": len(gaps),
    }
    return Connectome(spec, build_graph(spec), meta)


def gate_structure(program: Program) -> None:
    ops = {ins.op for ins in program.instructions}
    if "stimulus" not in ops:
        raise GateError("inert: the program never reads the stimulus")
    if not ({"sum_in"} & ops) and program.gap is None:
        raise GateError("uncoupled: no sum_in and no gap term, so every off-target kernel is zero")


class ResponseModel:
    """Pair kernels of a program on a connectome, for the (target, responder) labels of one task."""

    def __init__(
        self, program: Program, conn: Connectome, pairs: Sequence[tuple[str, str]], labels: dict[str, str], dt: float
    ) -> None:
        self.program = program
        index = {n: k for k, n in enumerate(conn.graph.ids)}
        mapped = [
            (p, labels[t], labels[r])
            for p, (t, r) in enumerate(pairs)
            if t in labels and r in labels and labels[t] in index and labels[r] in index
        ]
        self.n_pairs = len(pairs)
        self.pair_rows = np.asarray([m[0] for m in mapped], dtype=np.int64)
        targets = sorted({m[1] for m in mapped})
        responders = sorted({m[2] for m in mapped})
        t_index = {t: k for k, t in enumerate(targets)}
        r_index = {r: k for k, r in enumerate(responders)}
        self.pair_t = jnp.asarray([t_index[m[1]] for m in mapped], dtype=jnp.int32)
        self.pair_r = jnp.asarray([r_index[m[2]] for m in mapped], dtype=jnp.int32)
        n_sub = 1 if program.dt_max is None or program.dt_max >= dt else math.ceil(dt / program.dt_max - 1e-9)
        self.n_sub = n_sub
        n_steps = (int(KERNEL_KNOTS[-1]) + 2) * n_sub
        ticks = [(int(k) + 1) * n_sub for k in KERNEL_KNOTS]
        self.empty = not mapped
        if self.empty:
            return
        self.sim = JaxSimulator(program, conn.graph, dt / n_sub, n_steps, [index[r] for r in responders], ticks)
        stim = np.zeros((len(targets) + 1, n_steps, conn.graph.n))
        for k, t in enumerate(targets):
            stim[k + 1, :n_sub, index[t]] = PULSE
        self.stimulus = jnp.asarray(stim)

    def coefficients(self, theta: jax.Array) -> jax.Array:
        """(mapped pairs, M) tent coefficients for a full IR-order parameter vector."""
        obs = self.sim.observe(theta, self.stimulus)  # (J + 1, M, K)
        d = (obs[1:] - obs[0]) / PULSE
        return d[self.pair_t, :, self.pair_r]

    def dense(self, theta: Sequence[float]) -> FloatArray:
        c = np.zeros((self.n_pairs, M))
        if not self.empty:
            c[self.pair_rows] = np.asarray(self.coefficients(jnp.asarray(theta, dtype=jnp.float64)))
        return c


def gate_dynamics(model: ResponseModel, theta: Sequence[float]) -> None:
    if model.empty:
        raise GateError("no pair of the task maps onto the connectome")
    obs = np.asarray(model.sim.observe(jnp.asarray(theta, dtype=jnp.float64), model.stimulus))
    if not np.all(np.isfinite(obs)) or float(np.max(np.abs(obs))) > RESPONSE_BOUND:
        raise GateError("unstable: non-finite or unbounded response at the declared parameters")


def _profiled(ps: PairStats, rows: npt.NDArray[np.int64]) -> Callable[[jax.Array], jax.Array]:
    g, xq, xy = jnp.asarray(ps.G[rows]), jnp.asarray(ps.XQ[rows]), jnp.asarray(ps.Xy[rows])
    qq = jnp.asarray(ps.QQ)
    qq_inv = jnp.asarray(np.linalg.inv(ps.QQ + EPS * np.eye(ps.QQ.shape[0])))
    qy = jnp.asarray(ps.Qy)

    def value(c: jax.Array) -> jax.Array:
        beta = qq_inv @ (qy - jnp.einsum("pmh,pm->h", xq, c))
        quad = jnp.einsum("pm,pmk,pk->", c, g, c)
        cross = 2.0 * jnp.einsum("pm,pmh,h->", c, xq, beta)
        lin = -2.0 * jnp.einsum("pm,pm->", c, xy)
        return quad + cross + lin + beta @ qq @ beta - 2.0 * beta @ qy + ps.yy

    return value


def fit_program(
    model: ResponseModel, ps: PairStats, starts: int, seed: int, init: Sequence[float] | None = None
) -> Fit:
    program = model.program
    base = np.asarray(program.default_theta() if init is None else init, dtype=np.float64)
    tf = ParamTransforms.from_program(program) if any(p.trainable for p in program.parameters) else None
    rows = model.pair_rows
    keep = ps.present[rows]
    theta = base
    info: dict[str, Any] = {"converged": True, "starts": 0, "function_evaluations": 0}
    if tf is not None and keep.any():
        free = np.asarray([program.parameter_index(n) for n in tf.names], dtype=np.int64)
        value = _profiled(ps, rows[keep])
        keep_j = jnp.asarray(np.nonzero(keep)[0])

        def objective(raw: jax.Array) -> jax.Array:
            th = jnp.asarray(base).at[jnp.asarray(free)].set(tf.forward(raw))
            return value(model.coefficients(th)[keep_j])

        vg = jax.jit(jax.value_and_grad(objective))
        scale = max(abs(ps.yy), 1.0)

        def fun(x: FloatArray) -> tuple[float, FloatArray]:
            v, gr = vg(jnp.asarray(x))
            v, gr = float(v) / scale, np.asarray(gr, dtype=np.float64) / scale
            if not (math.isfinite(v) and np.all(np.isfinite(gr))):
                return 1e30, np.zeros_like(x)
            return v, gr

        rng = np.random.default_rng(seed)
        bounds = tf.raw_bounds()
        x0s = [tf.inverse(base[free])] + [rng.uniform(-3.0, 3.0, len(tf)) for _ in range(starts - 1)]
        best = None
        for x0 in x0s:
            res = optimize.minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": MAX_ITER})
            info["function_evaluations"] += int(res.nfev)
            if best is None or res.fun < best.fun:
                best = res
        assert best is not None
        theta = base.copy()
        theta[free] = np.asarray(tf.forward(jnp.asarray(best.x)))
        info.update(converged=bool(best.success), starts=len(x0s), iterations=int(best.nit))
    c = model.dense(theta.tolist())
    c[~ps.present] = 0.0
    beta = np.linalg.solve(ps.QQ + EPS * np.eye(ps.QQ.shape[0]), ps.Qy - np.einsum("pmh,pm->h", ps.XQ, c))
    n_params = l_params_count(program) + beta.size
    return Fit(
        "wrl",
        c,
        beta,
        loss(ps, c, beta, 0.0),
        n_params,
        int(info.get("iterations", 0)),
        {"theta": theta.tolist(), **info},
    )


def l_params_count(program: Program) -> int:
    return sum(1 for p in program.parameters if p.trainable)


@dataclass
class WrlCandidate:
    """One WRL program (source text) on a connectome; by default the declared one under ``root``."""

    source: str
    root: Path
    name: str = ""
    starts: int = 2
    seed: int = 0
    kind: str = "wrl"
    connectome: Connectome | None = field(default=None, repr=False)
    labels: dict[str, str] | None = field(default=None, repr=False)
    _program: Program | None = field(default=None, repr=False)

    @property
    def program(self) -> Program:
        if self._program is None:
            self._program = compile_source(self.source)
        return self._program

    @property
    def key(self) -> str:
        return f"wrl:{self.name or self.program.program_hash[:12]}"

    @property
    def l_struct_bits(self) -> float:
        return float(self.program.l_struct_bits)

    def _model(self, data: TaskData) -> ResponseModel:
        if self.connectome is None:
            self.connectome = load_connectome(self.root)
        if self.labels is None:
            self.labels = label_map(self.root)
        return ResponseModel(self.program, self.connectome, data.pairs, self.labels, data.dt)

    def inner_select(self, view: TrainingView) -> InnerResult:
        t0 = time.time()
        gate_structure(self.program)
        data = view.data
        model = self._model(data)
        gate_dynamics(model, self.program.default_theta())
        designs = trial_designs(data, _indicator(data))
        stats = compute_stats(data, designs)
        total, per_animal = 0.0, dict[str, float]()
        thetas: list[list[float]] = []
        for itrain, ival in view.inner:
            ps = aggregate(stats, itrain, len(data.pairs))
            fit = fit_program(model, ps, self.starts, self.seed)
            thetas.append(fit.extra["theta"])
            noise = noise_for(data, designs, fit, rows_of(data, itrain))
            vrows = rows_of(data, ival)
            nll, _, _ = score_rows(data, designs, fit, noise, vrows)
            total += float(nll.sum())
            for a, x in zip(np.asarray(data.animals, dtype=object)[data.trace_animal[vrows]], nll, strict=True):
                per_animal[a] = per_animal.get(a, 0.0) + float(x)
        return InnerResult(
            self.key,
            0.0,
            total,
            per_animal,
            self.l_struct_bits,
            float(l_params_bits(self.program)),
            l_params_count(self.program),
            time.time() - t0,
            {"inner_theta": thetas, "program_hash": self.program.program_hash},
        )

    def fit_final(self, view: TrainingView, ridge: float) -> dict[str, Any]:
        data = view.data
        ind = _indicator(data)
        designs = trial_designs(data, ind)
        ps = aggregate(compute_stats(data, designs), np.ones(len(data.animals), dtype=bool), len(data.pairs))
        fit = fit_program(self._model(data), ps, self.starts, self.seed)
        noise = noise_for(data, designs, fit, np.arange(data.n_traces))
        return {
            "program_hash": self.program.program_hash,
            "source": self.source,
            "theta": fit.extra["theta"],
            "connectome": (self.connectome.meta if self.connectome else {}),
            "task": data.task,
            "history": data.history,
            "beta": fit.beta.tolist(),
            "noise": noise.to_json(),
            "indicator": ind.to_json() if ind else None,
            "n_params": fit.n_params,
            "converged": fit.extra["converged"],
        }

    def score(self, fitted: dict[str, Any], test: TaskData) -> dict[str, float]:
        c = self._model(test).dense(fitted["theta"])
        kernels = {f"{t}|{r}": c[p].tolist() for p, (t, r) in enumerate(test.pairs) if np.any(c[p])}
        return score_kernels({**fitted, "kernels": kernels}, test)


def enumerate_programs(config: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Programs of a search config in canonical enumeration order (``ow search enumerate``), with L_struct."""
    out = run_ow(["search", "enumerate", "--config", str(config)])
    rows = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    progs = [r for r in rows if r.get("type") == "program"]
    return progs if limit is None else progs[:limit]


def search_volume(
    programs: Sequence[dict[str, Any]], root: Path, **kw: Any
) -> tuple[list[WrlCandidate], dict[str, Any]]:
    """Structurally gated candidates and a log of how many programs were enumerated, gated and kept (§7.8)."""
    kept: list[WrlCandidate] = []
    gated: dict[str, int] = {}
    for p in programs:
        cand = WrlCandidate(p["source"], root, name=p["hash"][:12], **kw)
        try:
            gate_structure(cand.program)
        except GateError as e:
            reason = str(e).split(":", 1)[0]
            gated[reason] = gated.get(reason, 0) + 1
            continue
        kept.append(cand)
    return kept, {"enumerated": len(programs), "gated": gated, "kept": len(kept)}


def budget_curve(inner: Sequence[InnerResult], orders: int = 20, seed: int = 0) -> dict[str, Any]:
    """Best inner NLL after k candidates, in enumeration order and over random orders (the §7.8 control)."""
    nll = np.asarray([r.inner_nll for r in inner], dtype=np.float64)
    if nll.size == 0:
        return {"k": [], "enumeration": [], "random_mean": [], "random_sd": []}
    rng = np.random.default_rng(seed)
    rand = np.stack([np.minimum.accumulate(nll[rng.permutation(nll.size)]) for _ in range(orders)])
    return {
        "k": list(range(1, nll.size + 1)),
        "enumeration": np.minimum.accumulate(nll).tolist(),
        "random_mean": rand.mean(axis=0).tolist(),
        "random_sd": rand.std(axis=0).tolist(),
    }
