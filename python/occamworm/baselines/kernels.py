"""Kernel baselines B0-B3 on pair sufficient statistics (§4.1, OW-006).

All families share one quadratic loss and one ridge penalty on the pair kernel, so their scores differ only by
the constraint on ``c_p`` (§4.1: "equally legitimate regularization"):

    L = sum_p [c_p' G_p c_p + 2 c_p' XQ_p beta - 2 c_p' Xy_p + lam ||c_p||^2] + beta' QQ beta - 2 beta' Qy + yy

- B0 null:            c_p = 0
- B1 shared per target: c_p = a_p h_j            (h_j shared by the responders of target j)
- B1d with delays:     c_p = a_p S_{d_p} h_j       (per-pair delay d_p in DELAYS volumes; S_d shifts the kernel)
- B2 low-rank bank:    c_p = V a_p, V (M x K)    (K global kernels; K = 1 is the stringent global B1)
- B3 independent:      c_p free                   (closed form)

``lam`` is ``lam_rel`` times the median per-pair ``trace(G_p) / M`` of the training data, so one grid of relative
values serves every family. Bilinear families are fitted by alternating least squares from a deterministic
initialization (the leading singular vectors of the B3 solution).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from occamworm.baselines.data import M, PairStats

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
EPS = 1e-9
MAX_ITER = 200
MAX_ITER_DELAYS = 60  # B1d improves by < 1e-7 per sweep after ~50 sweeps; recorded as not converged
TOL = 1e-7


@dataclass
class Fit:
    family: str
    c: FloatArray  # (P, M)
    beta: FloatArray  # (H,)
    loss: float
    n_params: int
    iterations: int
    extra: dict[str, Any] = field(default_factory=dict)


def lam_scale(ps: PairStats) -> float:
    tr = np.trace(ps.G, axis1=1, axis2=2)[ps.present] / M
    return float(np.median(tr)) if tr.size else 1.0


DELAYS = (-2, -1, 0, 1, 2)  # volumes, relative to the target's shared kernel


def shift_matrices() -> FloatArray:
    """S_d (len(DELAYS), M, M): tent coefficients of kernel(t - d), least-squares projected back onto the basis."""
    from occamworm.baselines.data import KERNEL_BASIS

    pinv = np.linalg.pinv(KERNEL_BASIS)
    n = KERNEL_BASIS.shape[0]
    out = np.zeros((len(DELAYS), M, M))
    for i, dd in enumerate(DELAYS):
        shifted = np.zeros_like(KERNEL_BASIS)
        if dd >= 0:
            shifted[dd:] = KERNEL_BASIS[: n - dd]
        else:
            shifted[: n + dd] = KERNEL_BASIS[-dd:]
        out[i] = pinv @ shifted
    return out


def loss(ps: PairStats, c: FloatArray, beta: FloatArray, lam: float) -> float:
    quad = np.einsum("pm,pmk,pk->", c, ps.G, c, optimize=True)
    cross = 2.0 * np.einsum("pm,pmh,h->", c, ps.XQ, beta, optimize=True)
    lin = -2.0 * np.einsum("pm,pm->", c, ps.Xy)
    return float(quad + cross + lin + lam * np.sum(c**2) + beta @ ps.QQ @ beta - 2.0 * beta @ ps.Qy + ps.yy)


def _beta(ps: PairStats, c: FloatArray) -> FloatArray:
    rhs = ps.Qy - np.einsum("pmh,pm->h", ps.XQ, c)
    return np.asarray(np.linalg.solve(ps.QQ + EPS * np.eye(ps.QQ.shape[0]), rhs))


def _resid_rhs(ps: PairStats, beta: FloatArray) -> FloatArray:
    return np.asarray(ps.Xy - np.einsum("pmh,h->pm", ps.XQ, beta))


def fit_null(ps: PairStats) -> Fit:
    c = np.zeros_like(ps.Xy)
    beta = _beta(ps, c)
    return Fit("B0", c, beta, loss(ps, c, beta, 0.0), beta.size, 1)


def fit_independent(ps: PairStats, lam_rel: float) -> Fit:
    lam = lam_rel * lam_scale(ps)
    eye = np.eye(M)
    a = ps.G + (lam + EPS) * eye
    ainv_xq = np.linalg.solve(a, ps.XQ)  # (P, M, H)
    ainv_xy = np.linalg.solve(a, ps.Xy[..., None])[..., 0]
    lhs = ps.QQ - np.einsum("pmh,pmk->hk", ps.XQ, ainv_xq)
    rhs = ps.Qy - np.einsum("pmh,pm->h", ps.XQ, ainv_xy)
    beta = np.linalg.solve(lhs + EPS * np.eye(lhs.shape[0]), rhs)
    c = ainv_xy - np.einsum("pmh,h->pm", ainv_xq, beta)
    c[~ps.present] = 0.0
    return Fit("B3", c, beta, loss(ps, c, beta, lam), int(ps.present.sum()) * M + beta.size, 1, {"lam": lam})


def _top_vectors(c: FloatArray, groups: IntArray | None, n_groups: int, k: int) -> FloatArray:
    """Leading right singular vectors of the rows of ``c`` (per group if given); deterministic sign."""

    def lead(rows: FloatArray, kk: int) -> FloatArray:
        if rows.shape[0] == 0 or not np.any(rows):
            v = np.zeros((M, kk))
            v[: min(kk, M), : min(kk, M)] = np.eye(min(kk, M))
            return v
        _, _, vt = np.linalg.svd(rows, full_matrices=False)
        v = np.zeros((M, kk))
        v[:, : min(kk, vt.shape[0])] = vt[:kk].T
        for j in range(kk):
            if v[np.argmax(np.abs(v[:, j])), j] < 0:
                v[:, j] *= -1
        return v

    if groups is None:
        return lead(c, k)
    return np.stack([lead(c[groups == g], 1)[:, 0] for g in range(n_groups)])


def fit_shared(
    ps: PairStats,
    pair_target: IntArray,
    n_targets: int,
    lam_rel: float,
    delays: bool = False,
    init: Fit | None = None,
) -> Fit:
    """B1 (and B1d with ``delays``): one kernel per stimulated target, per-pair amplitude (and discrete delay).

    ``init`` warm-starts the kernels from an earlier fit on a subset or superset of the same training animals;
    it never comes from held-out data.
    """
    lam = lam_rel * lam_scale(ps)
    if init is not None and "kernels" in init.extra:
        h = np.array(init.extra["kernels"], dtype=np.float64)
        beta = init.beta.copy()
    else:
        start = fit_independent(ps, lam_rel)
        h = _top_vectors(start.c, pair_target, n_targets, 1)  # (J, M)
        beta = start.beta
    shifts = shift_matrices() if delays else np.eye(M)[None]
    zero = DELAYS.index(0) if delays else 0
    gl = ps.G + lam * np.eye(M)
    live = ps.present
    n_p = ps.Xy.shape[0]
    amp = np.zeros(n_p)
    which = np.full(n_p, zero, dtype=np.int64)
    prev = np.inf
    cur = np.inf
    c = np.zeros_like(ps.Xy)
    it = 0
    converged = False
    for it in range(1, (MAX_ITER_DELAYS if delays else MAX_ITER) + 1):  # noqa: B007
        r = _resid_rhs(ps, beta)
        # pair step: best delay and amplitude given the target kernel
        v = np.einsum("dmk,pk->pdm", shifts, h[pair_target])  # (P, D, M)
        num = np.einsum("pdm,pm->pd", v, r)
        den = np.sum(v * np.transpose(gl @ np.transpose(v, (0, 2, 1)), (0, 2, 1)), axis=2) + EPS
        gain = num**2 / den
        which = np.where(live, np.argmax(gain - 1e-12 * np.abs(np.arange(shifts.shape[0]) - zero), axis=1), zero)
        amp = np.where(live, num[np.arange(n_p), which] / den[np.arange(n_p), which], 0.0)
        # kernel step: c_p = a_p S_p h_j, solved per target
        sp = shifts[which]  # (P, M, M)
        sgs = np.transpose(sp, (0, 2, 1)) @ gl @ sp
        lhs_j = np.zeros((n_targets, M, M))
        rhs_j = np.zeros((n_targets, M))
        np.add.at(lhs_j, pair_target[live], (amp**2)[live, None, None] * sgs[live])
        np.add.at(rhs_j, pair_target[live], amp[live, None] * np.einsum("pkm,pk->pm", sp[live], r[live]))
        has = np.zeros(n_targets, dtype=bool)
        has[pair_target[live]] = True
        hj = np.linalg.solve(lhs_j[has] + EPS * np.eye(M), rhs_j[has][..., None])[..., 0]
        norm = np.linalg.norm(hj, axis=1)
        ok = norm > 0
        idx = np.nonzero(has)[0][ok]
        h[idx] = hj[ok] / norm[ok, None]
        scale = np.ones(n_targets)
        scale[idx] = norm[ok]
        amp = amp * scale[pair_target]
        c = amp[:, None] * np.einsum("pmk,pk->pm", sp, h[pair_target])
        beta = _beta(ps, c)
        cur = loss(ps, c, beta, lam)
        if abs(prev - cur) <= TOL * max(1.0, abs(cur)):
            converged = True
            break
        prev = cur
    used = np.unique(pair_target[live])
    n_params = used.size * M + int(live.sum()) * (2 if delays else 1) + beta.size
    extra: dict[str, Any] = {"lam": lam, "kernels": h, "pair_amplitude": amp, "converged": converged}
    if delays:
        extra["pair_delay_volumes"] = np.asarray(DELAYS)[which]
    return Fit("B1d" if delays else "B1", c, beta, cur, n_params, it, extra)


def fit_lowrank(ps: PairStats, k: int, lam_rel: float, init: Fit | None = None) -> Fit:
    """B2: K global kernels shared by every pair. ``init`` warm-starts the bank (see ``fit_shared``)."""
    lam = lam_rel * lam_scale(ps)
    if init is not None and "kernels" in init.extra:
        v = np.array(init.extra["kernels"], dtype=np.float64).T
        beta = init.beta.copy()
    else:
        start = fit_independent(ps, lam_rel)
        v = _top_vectors(start.c[ps.present], None, 0, k)  # (M, K)
        beta = start.beta
    gl = ps.G + lam * np.eye(M)
    prev = np.inf
    it = 0
    a = np.zeros((ps.Xy.shape[0], k))
    converged = False
    for it in range(1, MAX_ITER + 1):  # noqa: B007
        r = _resid_rhs(ps, beta)
        lhs = np.einsum("mk,pmn,nl->pkl", v, gl, v, optimize=True) + EPS * np.eye(k)
        a = np.linalg.solve(lhs, np.einsum("mk,pm->pk", v, r)[..., None])[..., 0]
        a[~ps.present] = 0.0
        # V update: vec(V) column-major; c_p = V a_p
        aa = (a[:, :, None] * a[:, None, :]).reshape(-1, k * k)
        big = (aa.T @ gl.reshape(-1, M * M)).reshape(k, k, M, M).transpose(0, 2, 1, 3).reshape(k * M, k * M)
        big = big + EPS * np.eye(k * M)
        rhs = np.einsum("pk,pm->km", a, r).reshape(k * M)
        v = np.linalg.solve(big, rhs).reshape(k, M).T
        q, rr = np.linalg.qr(v)
        sign = np.sign(np.where(np.diag(rr) == 0, 1.0, np.diag(rr)))
        v, a = q * sign, a @ (rr.T * sign)
        c = a @ v.T
        beta = _beta(ps, c)
        cur = loss(ps, c, beta, lam)
        if abs(prev - cur) <= TOL * max(1.0, abs(cur)):
            converged = True
            break
        prev = cur
    n_params = k * M + int(ps.present.sum()) * k + beta.size
    extra = {"lam": lam, "kernels": v.T, "pair_coef": a, "converged": converged}
    return Fit(f"B2-K{k}", c, beta, cur, n_params, it, extra)


FAMILIES = ("B0", "B1", "B1d", "B2-K1", "B2-K2", "B2-K4", "B2-K8", "B3")


def fit_family(
    family: str, ps: PairStats, pair_target: IntArray, n_targets: int, lam_rel: float, init: Fit | None = None
) -> Fit:
    if family == "B0":
        return fit_null(ps)
    if family == "B1":
        return fit_shared(ps, pair_target, n_targets, lam_rel, init=init)
    if family == "B1d":
        return fit_shared(ps, pair_target, n_targets, lam_rel, delays=True, init=init)
    if family.startswith("B2-K"):
        return fit_lowrank(ps, int(family[4:]), lam_rel, init=init)
    if family == "B3":
        return fit_independent(ps, lam_rel)
    raise ValueError(family)
