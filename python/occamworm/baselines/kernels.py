"""Kernel baselines B0-B3 on pair sufficient statistics (§4.1, OW-006).

All families share one quadratic loss and one ridge penalty on the pair kernel, so their scores differ only by
the constraint on ``c_p`` (§4.1: "equally legitimate regularization"):

    L = sum_p [c_p' G_p c_p + 2 c_p' XQ_p beta - 2 c_p' Xy_p + lam ||c_p||^2] + beta' QQ beta - 2 beta' Qy + yy

- B0 null:            c_p = 0
- B1 shared per target: c_p = a_p h_j            (h_j shared by the responders of target j)
- B1d with delays:     c_p = (a_p I + b_p D) h_j  (first-order per-pair delay; D differentiates the kernel)
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

from occamworm.baselines.data import KERNEL_KNOTS, M, PairStats

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
EPS = 1e-9
MAX_ITER = 300
TOL = 1e-9


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


def derivative_matrix() -> FloatArray:
    """D such that (D h)[m] approximates dh/dt (per volume) at knot m of the piecewise-linear kernel."""
    k = KERNEL_KNOTS.astype(np.float64)
    d = np.zeros((M, M))
    for m in range(M):
        lo, hi = max(m - 1, 0), min(m + 1, M - 1)
        d[m, hi] += 1.0 / (k[hi] - k[lo])
        d[m, lo] -= 1.0 / (k[hi] - k[lo])
    return d


def loss(ps: PairStats, c: FloatArray, beta: FloatArray, lam: float) -> float:
    quad = np.einsum("pm,pmk,pk->", c, ps.G, c)
    cross = 2.0 * np.einsum("pm,pmh,h->", c, ps.XQ, beta)
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


def fit_shared(ps: PairStats, pair_target: IntArray, n_targets: int, lam_rel: float, delays: bool = False) -> Fit:
    """B1 (and B1d with ``delays``): one kernel per stimulated target, per-pair amplitude (and delay)."""
    lam = lam_rel * lam_scale(ps)
    init = fit_independent(ps, lam_rel)
    h = _top_vectors(init.c, pair_target, n_targets, 1)  # (J, M)
    beta = init.beta
    d = derivative_matrix()
    n_coef = 2 if delays else 1
    coef = np.zeros((ps.Xy.shape[0], n_coef))
    prev = np.inf
    it = 0
    for it in range(1, MAX_ITER + 1):  # noqa: B007
        r = _resid_rhs(ps, beta)
        hp = h[pair_target]  # (P, M)
        basis = np.stack([hp, hp @ d.T], axis=2)[:, :, :n_coef]  # (P, M, n_coef)
        gb = np.einsum("pmk,pkc->pmc", ps.G + lam * np.eye(M), basis)
        lhs = np.einsum("pmc,pmd->pcd", basis, gb) + EPS * np.eye(n_coef)
        coef = np.linalg.solve(lhs, np.einsum("pmc,pm->pc", basis, r)[..., None])[..., 0]
        coef[~ps.present] = 0.0
        # kernel update per target: c_p = A_p h with A_p = a_p I + b_p D
        for j in range(n_targets):
            sel = np.nonzero((pair_target == j) & ps.present)[0]
            if sel.size == 0:
                continue
            a_mats = coef[sel, 0, None, None] * np.eye(M)
            if delays:
                a_mats = a_mats + coef[sel, 1, None, None] * d
            g = ps.G[sel] + lam * np.eye(M)
            lhs_h = np.einsum("pkm,pkl,pln->mn", a_mats, g, a_mats) + EPS * np.eye(M)
            rhs_h = np.einsum("pkm,pk->m", a_mats, r[sel])
            hj = np.linalg.solve(lhs_h, rhs_h)
            norm = np.linalg.norm(hj)
            if norm > 0:
                h[j] = hj / norm
                coef[sel] *= norm
        c = np.einsum("pmk,pk->pm", np.stack([h[pair_target], h[pair_target] @ d.T], axis=2)[:, :, :n_coef], coef)
        beta = _beta(ps, c)
        cur = loss(ps, c, beta, lam)
        if abs(prev - cur) <= TOL * max(1.0, abs(cur)):
            break
        prev = cur
    used = np.unique(pair_target[ps.present])
    n_params = used.size * M + int(ps.present.sum()) * n_coef + beta.size
    return Fit("B1d" if delays else "B1", c, beta, cur, n_params, it,
               {"lam": lam, "kernels": h, "pair_coef": coef})


def fit_lowrank(ps: PairStats, k: int, lam_rel: float) -> Fit:
    """B2: K global kernels shared by every pair."""
    lam = lam_rel * lam_scale(ps)
    init = fit_independent(ps, lam_rel)
    v = _top_vectors(init.c[ps.present], None, 0, k)  # (M, K)
    beta = init.beta
    gl = ps.G + lam * np.eye(M)
    prev = np.inf
    it = 0
    a = np.zeros((ps.Xy.shape[0], k))
    for it in range(1, MAX_ITER + 1):  # noqa: B007
        r = _resid_rhs(ps, beta)
        lhs = np.einsum("mk,pmn,nl->pkl", v, gl, v) + EPS * np.eye(k)
        a = np.linalg.solve(lhs, np.einsum("mk,pm->pk", v, r)[..., None])[..., 0]
        a[~ps.present] = 0.0
        # V update: vec(V) column-major; c_p = V a_p
        big = np.einsum("pk,pl,pmn->kmln", a, a, gl).reshape(k * M, k * M) + EPS * np.eye(k * M)
        rhs = np.einsum("pk,pm->km", a, r).reshape(k * M)
        v = np.linalg.solve(big, rhs).reshape(k, M).T
        q, rr = np.linalg.qr(v)
        sign = np.sign(np.where(np.diag(rr) == 0, 1.0, np.diag(rr)))
        v, a = q * sign, a @ (rr.T * sign)
        c = a @ v.T
        beta = _beta(ps, c)
        cur = loss(ps, c, beta, lam)
        if abs(prev - cur) <= TOL * max(1.0, abs(cur)):
            break
        prev = cur
    n_params = k * M + int(ps.present.sum()) * k + beta.size
    return Fit(f"B2-K{k}", c, beta, cur, n_params, it, {"lam": lam, "kernels": v.T, "pair_coef": a})


FAMILIES = ("B0", "B1", "B1d", "B2-K1", "B2-K2", "B2-K4", "B2-K8", "B3")


def fit_family(family: str, ps: PairStats, pair_target: IntArray, n_targets: int, lam_rel: float) -> Fit:
    if family == "B0":
        return fit_null(ps)
    if family == "B1":
        return fit_shared(ps, pair_target, n_targets, lam_rel)
    if family == "B1d":
        return fit_shared(ps, pair_target, n_targets, lam_rel, delays=True)
    if family.startswith("B2-K"):
        return fit_lowrank(ps, int(family[4:]), lam_rel)
    if family == "B3":
        return fit_independent(ps, lam_rel)
    raise ValueError(family)
