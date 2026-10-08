"""Cross-section field of a swept primitive: r(s, phi) ~ sum_n a_n(s) cos(n phi) + b_n(s) sin(n phi).

Per slab, points are put in polar coordinates about the spine (radius rho, angle phi in a frame
transported along the spine). The outer envelope per angle bin (a high radius quantile) is fit by
a Fourier series of order <= n_max with one-sided robust reweighting: bins that stick OUT of the
fit (a handle, a spout) are rejected, so the fit describes the body and the protrusion becomes
residual. n = 0 is a surface of revolution; n = 4 dominates for a box-like sweep.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


@dataclass
class Section:
    rho: np.ndarray       # (N,) radius of each point about the spine
    phi: np.ndarray       # (N,) angle of each point
    coef: np.ndarray      # (S, 2 n_max + 1) Fourier coefficients per slab [a0, a1, b1, ..., an, bn]
    rho_hat: np.ndarray   # (N,) fitted envelope radius at each point's (s, phi)
    n_max: int

    def amplitudes(self):
        """(S, n_max + 1): a0 and |c_n| = sqrt(a_n^2 + b_n^2), rotation invariant."""
        c = self.coef
        amp = [c[:, 0]] + [np.hypot(c[:, 2 * n - 1], c[:, 2 * n]) for n in range(1, self.n_max + 1)]
        return np.stack(amp, 1)


def _frames(T, ref=None):
    """Per-slab (u, v) perpendicular to the tangents, transported along the spine."""
    S = len(T)
    if ref is None:
        ref = np.array([1.0, 0, 0]) if abs(T[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    U = np.zeros((S, 3))
    u = ref - (ref @ T[0]) * T[0]
    u /= np.linalg.norm(u)
    for k in range(S):
        u = u - (u @ T[k]) * T[k]
        nu = np.linalg.norm(u)
        if nu < 1e-6:
            u = ref - (ref @ T[k]) * T[k]
            nu = np.linalg.norm(u)
        u = u / nu
        U[k] = u
    V = np.cross(T, U)
    return U, V


def _basis(phi, n):
    cols = [np.ones_like(phi)]
    for k in range(1, n + 1):
        cols += [np.cos(k * phi), np.sin(k * phi)]
    return np.stack(cols, -1)


def _robust_fit(A, y, h, iters, out_k):
    w = np.ones(len(y))
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(iters):
        c = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        res = y - A @ c
        sc = max(1.4826 * np.median(np.abs(res[w > 0] - np.median(res[w > 0]))) if (w > 0).sum() > 2 else h, h)
        w_new = (res < out_k * sc).astype(float)  # one-sided: protrusions are rejected
        if w_new.sum() < max(3, A.shape[1]):
            break
        w = w_new
    return c, w


def fit_section(P, spine, h, n_max=4, n_bins=24, q=0.9, iters=4, out_k=2.5):
    S = spine.n_slabs
    U, V = _frames(spine.tangents)
    sl = spine.slab
    X = P - spine.centres[sl]
    T = spine.tangents[sl]
    X = X - np.sum(X * T, 1)[:, None] * T
    rho = np.linalg.norm(X, axis=1)
    phi = np.arctan2(np.sum(X * V[sl], 1), np.sum(X * U[sl], 1))
    nb = n_bins
    bc = -np.pi + (np.arange(nb) + 0.5) * 2 * np.pi / nb
    pb = np.clip(((phi + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    D = 2 * n_max + 1
    coef = np.full((S, D), np.nan)
    order = np.argsort(sl, kind="stable")
    bounds = np.searchsorted(sl[order], np.arange(S + 1))
    for k in range(S):
        idx = order[bounds[k]:bounds[k + 1]]
        if len(idx) < 8:
            continue
        r_b = np.full(nb, np.nan)
        cnt = np.bincount(pb[idx], minlength=nb)
        for b in np.flatnonzero(cnt >= 2):
            r_b[b] = np.quantile(rho[idx][pb[idx] == b], q)
        ok = ~np.isnan(r_b)
        y = r_b[ok]
        # order selection on the same bins for every order: truncated squared loss (protruding
        # bins cost at most (out_k sigma)^2, so a handle cannot buy extra harmonics) + BIC penalty
        fits = [_robust_fit(_basis(bc[ok], n), y, h, iters, out_k)[0]
                for n in range(int(min(n_max, max(0, (ok.sum() - 1) // 3))) + 1)]
        r0 = y - _basis(bc[ok], 0) @ fits[0]
        sig = max(1.4826 * np.median(np.abs(r0 - np.median(r0))), h)
        best = None
        for n, c in enumerate(fits):
            res = y - _basis(bc[ok], n) @ c
            loss = np.sum(np.minimum(res ** 2, (out_k * sig) ** 2)) / sig ** 2
            bic = loss + (2 * n + 1) * np.log(len(y))
            if best is None or bic < best[0]:
                best = (bic, c)
        cc = np.zeros(D)
        cc[:len(best[1])] = best[1]
        coef[k] = cc
    ok = ~np.isnan(coef[:, 0])
    if not ok.any():
        coef[:] = 0.0
        coef[:, 0] = np.quantile(rho, q)
    else:
        for j in range(D):
            coef[:, j] = np.interp(np.arange(S), np.flatnonzero(ok), coef[ok, j])
        coef = median_filter(coef, size=(3, 1), mode="nearest")
    rho_hat = np.sum(coef[sl] * _basis(phi, n_max), 1)
    return Section(rho, phi, coef, rho_hat, n_max)
