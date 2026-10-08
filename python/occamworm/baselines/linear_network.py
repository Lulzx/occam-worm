"""B4: linear network dynamics on a fixed anatomical topology (§4.1, OW-007).

    dx/dt = A x + b_j u(t),    A = -I / tau + W + g_gap L,    b_j = W[:, j] + g_gap Gn[:, j]

``C[i, j]`` counts chemical synapses j -> i and ``Gn`` is the symmetric gap-junction count matrix, both from one
declared reconstruction (``RECONSTRUCTION``) and scaled by their largest row sum. ``L = Gn - diag(Gn 1)`` is the
gap-junction Laplacian (negative semidefinite, so it preserves constant equilibria). The stimulated neuron j feeds
its measured activity (T2a) or a nominal impulse (T2b; the indicator is applied by the design) into its
postsynaptic and gap partners; its own state is not clamped, which is the declared approximation.

- B4-fixed:   W = (0.99 / tau) g C,          g = tanh(theta) in (-1, 1)      (3 parameters with tau and g_gap)
- B4-learned: W = (0.99 / tau) C diag(s),    s_j = tanh(theta_j) per presynaptic neuron (sign and strength)

Because C is scaled to unit max row sum, every Gershgorin disc of A lies in the open left half-plane for all
parameter values: the network is stable by construction. The kernel from target j to responder i is
``K_ij(t) = [exp(A t) b_j]_i``; on the shared tent basis its coefficients are ``dt * K_ij(knot * dt)``, so B4 is
scored by exactly the same machinery as B0-B3. Pairs whose labels do not resolve to a canonical neuron (OW-013)
get a zero kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
from scipy import optimize

from occamworm.baselines.data import KERNEL_KNOTS, M, PairStats, TaskData
from occamworm.baselines.kernels import EPS, Fit

Fitter = Callable[[PairStats, float, Fit | None], Fit]

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
RECONSTRUCTION = "cook2019-herm"
STABILITY_MARGIN = 0.99
TAU_MIN = 0.1  # s


@dataclass
class Network:
    neurons: list[str]
    chem: FloatArray  # (N, N), C[i, j] = synapses j -> i, unit max row sum
    gap: FloatArray  # (N, N) symmetric, unit max row sum
    reconstruction: str


def load_network(root: Path, reconstruction: str = RECONSTRUCTION) -> Network:
    ann = root / "data" / "normalized" / "annotations-v1"
    neurons = sorted(
        str(x) for x in pq.read_table(ann / "neurons.parquet", columns=["neuron_id"]).column(0).to_pylist()
    )
    edges = pq.read_table(ann / "edges.parquet").to_pylist()
    return network_from_edges(neurons, edges, reconstruction)


def network_from_edges(neurons: list[str], edges: list[dict[str, Any]], reconstruction: str) -> Network:
    index = {n: k for k, n in enumerate(neurons)}
    n = len(neurons)
    chem = np.zeros((n, n))
    gap = np.zeros((n, n))
    for e in edges:
        if e["source_reconstruction"] != reconstruction:
            continue
        s, t = index.get(e["source_neuron_id"]), index.get(e["target_neuron_id"])
        if s is None or t is None or s == t:
            continue
        w = float(e["synapse_count"] or 0.0)
        if e["edge_kind"] == "chem":
            chem[t, s] += w
        elif e["edge_kind"] == "gap":
            gap[t, s] = max(gap[t, s], w)
    gap = np.maximum(gap, gap.T)  # one row per direction upstream; symmetric by definition
    chem /= max(float(np.max(np.abs(chem).sum(axis=1))), 1e-12)
    gap /= max(float(np.max(gap.sum(axis=1))), 1e-12)
    return Network(neurons, chem, gap, reconstruction)


def label_map(root: Path) -> dict[str, str]:
    """Atlas label -> canonical neuron id, exact or unambiguous alias only (OW-013)."""
    rows = pq.read_table(root / "data" / "normalized" / "annotations-v1" / "unresolved.parquet").to_pylist()
    return {r["label"]: r["neuron_id"] for r in rows if r["resolved"] and r["neuron_id"]}


def _tau(theta: jax.Array) -> jax.Array:
    return TAU_MIN + jax.nn.softplus(theta)


def system(theta: jax.Array, net_chem: jax.Array, net_gap: jax.Array, learned: bool) -> tuple[jax.Array, jax.Array]:
    """(A, B) with B[:, j] the input vector of target j."""
    tau = _tau(theta[0])
    g_gap = jax.nn.softplus(theta[1])
    n = net_chem.shape[0]
    if learned:
        w = (STABILITY_MARGIN / tau) * net_chem * jnp.tanh(theta[2 : 2 + n])[None, :]
    else:
        w = (STABILITY_MARGIN / tau) * jnp.tanh(theta[2]) * net_chem
    lap = net_gap - jnp.diag(net_gap.sum(axis=1))
    a = -jnp.eye(n) / tau + w + g_gap * lap
    return a, w + g_gap * net_gap


def kernels_at_knots(a: jax.Array, b: jax.Array, dt: float) -> jax.Array:
    """exp(A * knot * dt) @ B for every knot: (M, N, J)."""
    step = jax.scipy.linalg.expm(a * dt)

    def body(x: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        return step @ x, x

    _, xs = jax.lax.scan(body, b, None, length=int(KERNEL_KNOTS[-1]) + 1)
    return xs[jnp.asarray(KERNEL_KNOTS)]


class B4Model:
    """Maps parameters to pair kernel coefficients and fits them on pair statistics."""

    def __init__(self, data: TaskData, net: Network, labels: dict[str, str], learned: bool) -> None:
        index = {n: k for k, n in enumerate(net.neurons)}
        self.learned = learned
        self.dt = data.dt
        self.chem = jnp.asarray(net.chem)
        self.gap = jnp.asarray(net.gap)
        targets = sorted({labels[t] for t, _ in data.pairs if t in labels and labels[t] in index})
        t_index = {t: k for k, t in enumerate(targets)}
        self.target_cols = jnp.asarray([index[t] for t in targets], dtype=jnp.int32)
        mapped = [
            (p, t_index[labels[t]], index[labels[r]])
            for p, (t, r) in enumerate(data.pairs)
            if t in labels and r in labels and labels[t] in index and labels[r] in index
        ]
        self.pair_rows = np.asarray([m[0] for m in mapped], dtype=np.int64)
        self.pair_target = jnp.asarray([m[1] for m in mapped], dtype=jnp.int32)
        self.pair_resp = jnp.asarray([m[2] for m in mapped], dtype=jnp.int32)
        self.n_pairs = len(data.pairs)
        self.n_theta = 2 + (len(net.neurons) if learned else 1)
        # Compiled once per model: every fit (inner folds x ridge values, outer refit) reuses it.
        self._value_and_grad = jax.jit(jax.value_and_grad(self._objective))

    def _objective(
        self,
        theta: jax.Array,
        g: jax.Array,
        xq: jax.Array,
        xy: jax.Array,
        qq: jax.Array,
        qq_inv: jax.Array,
        qy: jax.Array,
        yy: float,
        lam: float,
        theta_l2: float,
    ) -> jax.Array:
        c = self.coefficients(theta)
        beta = qq_inv @ (qy - jnp.einsum("pmh,pm->h", xq, c))
        quad = jnp.einsum("pm,pmk,pk->", c, g, c)
        cross = 2.0 * jnp.einsum("pm,pmh,h->", c, xq, beta)
        lin = -2.0 * jnp.einsum("pm,pm->", c, xy)
        q = beta @ qq @ beta - 2.0 * beta @ qy
        out: jax.Array = quad + cross + lin + lam * jnp.sum(c**2) + q + yy + theta_l2 * jnp.sum(theta[2:] ** 2)
        return out

    def coefficients(self, theta: jax.Array) -> jax.Array:
        """(mapped pairs, M) tent coefficients."""
        a, b = system(theta, self.chem, self.gap, self.learned)
        k = kernels_at_knots(a, b[:, self.target_cols], self.dt)  # (M, N, J)
        return self.dt * k[:, self.pair_resp, self.pair_target].T

    def initial(self) -> FloatArray:
        th = np.zeros(self.n_theta)
        th[0] = np.log(np.expm1(2.0 - TAU_MIN))  # tau = 2 s
        th[1] = -3.0
        return th

    def fit(self, ps: PairStats, lam_rel: float, lam: float, init: Fit | None) -> Fit:
        rows = self.pair_rows
        g = jnp.asarray(ps.G[rows])
        xq = jnp.asarray(ps.XQ[rows])
        xy = jnp.asarray(ps.Xy[rows])
        qq_inv = jnp.asarray(np.linalg.inv(ps.QQ + EPS * np.eye(ps.QQ.shape[0])))
        qy = jnp.asarray(ps.Qy)
        theta_l2 = 1e-3 * lam_rel
        args = (g, xq, xy, jnp.asarray(ps.QQ), qq_inv, qy, ps.yy, lam, theta_l2)
        scale = max(abs(ps.yy), 1.0)

        def fun(x: FloatArray) -> tuple[float, FloatArray]:
            v, gr = self._value_and_grad(jnp.asarray(x), *args)
            return float(v) / scale, np.asarray(gr, dtype=np.float64) / scale

        x0 = np.asarray(init.extra["theta"]) if init is not None and "theta" in init.extra else self.initial()
        res = optimize.minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": 300, "gtol": 1e-9})
        theta = jnp.asarray(res.x)
        c_mapped = np.asarray(self.coefficients(theta))
        c = np.zeros((self.n_pairs, M))
        c[rows] = c_mapped
        c[~ps.present] = 0.0
        beta = np.linalg.solve(ps.QQ + EPS * np.eye(ps.QQ.shape[0]), ps.Qy - np.einsum("pmh,pm->h", ps.XQ, c))
        from occamworm.baselines.kernels import loss

        tau = float(_tau(theta[0]))
        return Fit(
            "B4-learned" if self.learned else "B4-fixed",
            c,
            beta,
            loss(ps, c, beta, lam),
            self.n_theta + beta.size,
            int(res.nit),
            {
                "lam": lam,
                "theta": np.asarray(theta),
                "tau_s": tau,
                "g_gap": float(jax.nn.softplus(theta[1])),
                "converged": bool(res.success),
                "mapped_pairs": int(rows.size),
                "function_evaluations": int(res.nfev),
            },
        )


def make_b4_fitter(family: str, data: TaskData, root: Path) -> Fitter:
    from occamworm.baselines.kernels import lam_scale

    model = B4Model(data, load_network(root), label_map(root), learned=family == "B4-learned")

    def fitter(ps: PairStats, lam_rel: float, init: Fit | None) -> Fit:
        return model.fit(ps, lam_rel, lam_rel * lam_scale(ps), init)

    return fitter
