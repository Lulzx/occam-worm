"""Reference body: a planar chain under resistive-force theory (§13.4 kinematic test, OW-016).

The body is ``n`` rigid segments of length ``L / n``. Muscles set a target curvature per joint,

    kappa_target_j = kappa_max * (a_dorsal_j - a_ventral_j),     d kappa / dt = (kappa_target - kappa) / tau_m,

integrated exactly over each step. At low Reynolds number the body is force- and torque-free: each segment feels
drag ``f = -ds (c_t (v.t) t + c_n (v.n) n)``; given the shape and its rate of change, the head velocity and heading
rate solve the 3x3 linear force/torque balance exactly. Head is node 0; a curvature wave travelling from head to
tail drives the body forward when ``c_n > c_t``.

This is a deliberately small, deterministic reference engine for interface and control checks, not a validated
worm body (BAAIWorm, M7, is the candidate for that).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from occamworm.embodiment.interfaces import BodyState, BodyStepResult, InterfaceError, MuscleActivation

FloatArray = npt.NDArray[np.float64]


def _e(phi: FloatArray) -> FloatArray:
    return np.stack([np.cos(phi), np.sin(phi)], axis=-1)


def _eperp(phi: FloatArray) -> FloatArray:
    return np.stack([-np.sin(phi), np.cos(phi)], axis=-1)


@dataclass(frozen=True)
class RFTBody:
    n_segments: int = 24
    length: float = 1e-3  # m
    c_t: float = 1.0  # tangential drag per length (N s / m^2), relative units
    c_n: float = 20.0  # normal drag; c_n / c_t ~ 20-40 on agar
    kappa_max: float = 10e3  # 1/m at full differential activation (~10/mm)
    tau_muscle: float = 0.1  # s

    @property
    def ds(self) -> float:
        return self.length / self.n_segments

    def rest(self) -> BodyState:
        return BodyState(np.zeros(2), 0.0, np.zeros(self.n_segments - 1))

    def angles(self, heading: float, kappa: FloatArray) -> FloatArray:
        return heading + self.ds * np.concatenate([[0.0], np.cumsum(kappa)])

    def midpoints(self, head: FloatArray, heading: float, kappa: FloatArray) -> FloatArray:
        phi = self.angles(heading, kappa)
        steps = self.ds * _e(phi)
        nodes = head + np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])
        return np.asarray(0.5 * (nodes[:-1] + nodes[1:]))

    def _jacobians(self, heading: float, kappa: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Midpoint velocity = A u + B kappa_dot with u = (head_x', head_y', heading'). Returns (A, B, phi)."""
        n, ds = self.n_segments, self.ds
        phi = self.angles(heading, kappa)
        ep = _eperp(phi)  # (n, 2)
        a = np.zeros((n, 2, 3))
        a[:, 0, 0] = 1.0
        a[:, 1, 1] = 1.0
        # d m_i / d heading = sum_{k<i} ds eperp_k + ds/2 eperp_i
        cum = np.vstack([np.zeros(2), np.cumsum(ds * ep, axis=0)[:-1]])
        a[:, :, 2] = cum + 0.5 * ds * ep
        # d m_i / d kappa_j = ds * (sum_{j<k<i} ds eperp_k + [i > j] ds/2 eperp_i)
        b = np.zeros((n, 2, n - 1))
        for j in range(n - 1):
            for i in range(j + 1, n):
                b[i, :, j] = ds * (ds * ep[j + 1 : i].sum(axis=0) + 0.5 * ds * ep[i])
        return a, b, phi

    def drag_matrices(self, phi: FloatArray) -> FloatArray:
        t, nn = _e(phi), _eperp(phi)
        return np.asarray(
            self.ds * (self.c_t * t[:, :, None] * t[:, None, :] + self.c_n * nn[:, :, None] * nn[:, None, :])
        )

    def rigid_velocity(self, head: FloatArray, heading: float, kappa: FloatArray, kappa_dot: FloatArray) -> FloatArray:
        """Solve zero net force and zero net torque (about the head) for u = (head velocity, heading rate)."""
        a, b, phi = self._jacobians(heading, kappa)
        d = self.drag_matrices(phi)
        r = self.midpoints(head, heading, kappa) - head
        shape_v = np.einsum("nij,j->ni", b, kappa_dot)
        da = np.einsum("nij,njk->nik", d, a)  # force per unit u
        db = np.einsum("nij,nj->ni", d, shape_v)

        def cross(rr: FloatArray, ff: FloatArray) -> FloatArray:
            return np.asarray(rr[..., 0] * ff[..., 1] - rr[..., 1] * ff[..., 0])

        lhs = np.zeros((3, 3))
        rhs = np.zeros(3)
        lhs[:2] = da.sum(axis=0)
        rhs[:2] = -db.sum(axis=0)
        lhs[2] = np.array([cross(r, da[:, :, k]).sum() for k in range(3)])
        rhs[2] = -cross(r, db).sum()
        return np.asarray(np.linalg.solve(lhs, rhs))

    def __call__(self, body: BodyState, muscles: MuscleActivation, environment: object, dt: float) -> BodyStepResult:
        n = self.n_segments
        act = muscles.activation
        if act.size != 2 * (n - 1):
            raise InterfaceError(f"RFTBody expects {2 * (n - 1)} joint activations (dorsal, ventral), got {act.size}")
        target = self.kappa_max * (act[: n - 1] - act[n - 1 :])
        decay = np.exp(-dt / self.tau_muscle)
        kappa_new = target + (body.curvature - target) * decay
        kappa_dot = (kappa_new - body.curvature) / dt
        mid_kappa = 0.5 * (body.curvature + kappa_new)
        u = self.rigid_velocity(body.head, body.heading, mid_kappa, kappa_dot)
        heading = body.heading + dt * u[2]
        head = body.head + dt * u[:2]
        a, b, phi = self._jacobians(body.heading, mid_kappa)
        v = np.einsum("nij,j->ni", a, u) + np.einsum("nij,j->ni", b, kappa_dot)
        forces = -np.einsum("nij,nj->ni", self.drag_matrices(phi), v)
        state = BodyState(head, float(heading), kappa_new, body.time + dt)
        return BodyStepResult(state, u, forces, {"head_x": float(head[0]), "head_y": float(head[1])})
