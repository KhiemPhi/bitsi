"""BITSI v3: BITC channels measured against the fitted primitive, Chebyshev pieces, MDL splitting.

Lineage. BITSI v0 profiles a sweep with BITC (per slab: the fraction of points on the faces of the
object's box x the number of components), fits it with Chebyshev series and cuts at inflections.
v3 keeps that design and changes what is measured and how cuts are chosen:

  * Primitive-relative BITC. Per slab and per cross-section component k: rho_k = fraction of the
    component's points within eps of the PRIMITIVE's fitted cross-section (the box of BITC v0 is
    the special case "cross-section = bounding rectangle"), chi_k = its Euler characteristic.
    B(s) = sum_k rho_k chi_k (topological BITC), beta_0(s), beta_1(s).
  * The cross-section itself: Fourier amplitudes a_0(s), |c_1|(s) ... |c_4|(s) (bitsi2.section).
  * Appearance: per-slab histograms of the normal angle to the spine (6) and CIELAB colour (12),
    as square-root proportions.
  Every channel is divided by its noise (local-quadratic residual scatter, inflated for the
  autocorrelation of neighbouring slabs), so all channels are in independent-noise units.

  * Cuts = Chebyshev resolution with minimum description length. A piece of the sweep is ONE
    primitive while a short Chebyshev series in s describes every channel. For a candidate piece of
    m slabs, each channel takes the degree k <= 8 minimising RSS_k + (k + 1) log m (BIC with unit
    noise; nested degrees from one QR of the Chebyshev-Vandermonde matrix). The partition of the
    spine minimising sum(piece costs) + penalty * log S per cut is found exactly by dynamic
    programming. A step (a lid) or a kink (a neck) makes one series expensive and two short series
    cheap; a smooth taper does not. The number of parts falls out of the code length; no z
    threshold, no part count.

The rest (spines, cross-section fit, off-axis children, attachment topology) is BITSI v2.
"""
import numpy as np
from numpy.polynomial import chebyshev as Ch

from .bitsi2.section import _frames
from .bitsi2.tree import BITSI2
from .sweep_events import robust_noise
from .topo_bitsi import _components


def slab_channels(sp, se, P, N, C, own, h, eps_h=2.0, cell_h=3.0, window=3, min_frac=0.05):
    """(S, D) channel matrix + names for a node's own points along spine sp."""
    S = sp.n_slabs
    sl = sp.slab
    rho = se.rho if own is None or len(se.rho) == len(P) else se.rho[own]
    rho_hat = se.rho_hat if own is None or len(se.rho_hat) == len(P) else se.rho_hat[own]
    fit_ok = np.abs(rho - rho_hat) < eps_h * h
    # local 2-D coordinates per slab
    U, V = _frames(sp.tangents)
    X = P - sp.centres[sl]
    T = sp.tangents[sl]
    X = X - np.sum(X * T, 1)[:, None] * T
    u, v = np.sum(X * U[sl], 1), np.sum(X * V[sl], 1)
    cell = cell_h * h
    gu = ((u - u.min()) / cell).astype(int)
    gv = ((v - v.min()) / cell).astype(int)
    shape = (gu.max() + 2, gv.max() + 2)
    order = np.argsort(sl, kind="stable")
    bounds = np.searchsorted(sl[order], np.arange(S + 1))
    topo = np.zeros((S, 3))  # B = sum rho_k chi_k, beta0, beta1
    half = window // 2
    for s in range(S):
        idx = order[bounds[max(0, s - half)]:bounds[min(S, s + half + 1)]]
        if len(idx) < 5:
            continue
        comps = _components(gu[idx], gv[idx], shape, max(3, int(min_frac * len(idx))), 2)
        for mk, chi in comps:
            topo[s, 0] += fit_ok[idx[mk]].mean() * chi
            topo[s, 1] += 1
            topo[s, 2] += 1 - chi
    cols = [se.amplitudes(), topo]
    names = [f"a{j}" for j in range(se.amplitudes().shape[1])] + ["B", "beta0", "beta1"]
    n = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    ang = np.abs(np.sum(n * sp.tangents[sl], 1))
    H = np.zeros((S, 6))
    np.add.at(H, (sl, np.minimum((ang * 6).astype(int), 5)), 1)
    hists = [("n", H)]
    if C is not None:
        from skimage.color import rgb2lab
        c = np.asarray(C, float)
        c = c / 255.0 if c.max() > 1.5 else c
        lab = rgb2lab(c[None])[0]
        Hc = np.zeros((S, 12))
        np.add.at(Hc, (sl, np.clip((lab[:, 0] / 25).astype(int), 0, 3)), 1)
        np.add.at(Hc, (sl, 4 + np.clip(((lab[:, 1] + 40) / 20).astype(int), 0, 3)), 1)
        np.add.at(Hc, (sl, 8 + np.clip(((lab[:, 2] + 40) / 20).astype(int), 0, 3)), 1)
        hists.append(("c", Hc))
    for nm, Hm in hists:
        tot = Hm.sum(1, keepdims=True)
        cols.append(np.sqrt(Hm / np.maximum(tot, 1)))
        names += [f"{nm}{j}" for j in range(Hm.shape[1])]
    Y = np.hstack(cols)
    return Y, names


def standardize(Y, win=9):
    """Divide each channel by its noise, in units where neighbouring slabs are NOT independent.

    sigma^2 = mean squared residual of local quadratic fits (sliding windows of `win` slabs), the
    small-scale scatter around a smooth trend; rho = lag-1 autocorrelation of those residuals.
    Correlated noise carries fewer independent samples (effective n = n (1 - rho) / (1 + rho)), so
    the variance is inflated by (1 + rho) / (1 - rho). Without this, BIC sees signal everywhere.
    """
    keep = np.ptp(Y, 0) > 1e-12
    Y = Y[:, keep]
    S = len(Y)
    w = min(win, S)
    x = np.linspace(-1, 1, w)
    A = np.stack([np.ones(w), x, x * x], 1)
    Hm = np.eye(w) - A @ np.linalg.pinv(A)          # residual maker for a quadratic
    out = np.zeros_like(Y)
    for j in range(Y.shape[1]):
        R = np.array([Hm @ Y[i:i + w, j] for i in range(S - w + 1)])   # (windows, w)
        r = R[:, w // 2]                                              # centre residuals
        sig2 = max(np.mean(R ** 2) * w / max(w - 3, 1), (1e-3 * np.ptp(Y[:, j])) ** 2)
        rho = np.clip(np.corrcoef(r[:-1], r[1:])[0, 1] if len(r) > 3 and r.std() > 0 else 0.0, 0.0, 0.9)
        out[:, j] = Y[:, j] / np.sqrt(sig2 * (1 + rho) / (1 - rho))
    return out, keep


def piece_cost(Y, a, b, max_deg=8):
    """MDL cost of slabs [a, b) as one Chebyshev piece per channel, and the chosen degrees."""
    m = b - a
    Yp = Y[a:b]
    K = int(min(max_deg, m - 1))
    x = np.linspace(-1, 1, m) if m > 1 else np.zeros(1)
    Q, _ = np.linalg.qr(Ch.chebvander(x, K))
    proj = Q.T @ Yp                                  # (K+1, D)
    tot = np.sum(Yp ** 2, 0)
    rss = tot[None, :] - np.cumsum(proj ** 2, 0)     # RSS for degree 0..K
    pen = (np.arange(K + 1) + 1)[:, None] * np.log(max(m, 2))
    c = rss + pen
    k = np.argmin(c, 0)
    return float(c[k, np.arange(c.shape[1])].sum()), k


def mdl_partition(Y, min_len=3, penalty=1.0, max_deg=8):
    """Exact optimal partition of the S rows of Y into Chebyshev pieces (dynamic programming)."""
    S = len(Y)
    if S < 2 * min_len:
        return [], piece_cost(Y, 0, S, max_deg)[0]
    cost = {}
    for a in range(S):
        for b in range(a + min_len, S + 1):
            cost[a, b] = piece_cost(Y, a, b, max_deg)[0]
    cut_pen = penalty * np.log(S)
    F = np.full(S + 1, np.inf)
    F[0] = 0.0
    back = np.zeros(S + 1, int)
    for b in range(min_len, S + 1):
        for a in range(0, b - min_len + 1):
            if (a, b) not in cost or not np.isfinite(F[a]):
                continue
            v = F[a] + cost[a, b] + (cut_pen if a > 0 else 0.0)
            if v < F[b]:
                F[b], back[b] = v, a
    cuts, b = [], S
    while b > 0:
        a = back[b]
        if a > 0:
            cuts.append(a - 1)  # cut after slab a-1
        b = a
    return sorted(cuts), float(F[S])


class BITSI3(BITSI2):
    """BITSI v2 with the coaxial cuts replaced by primitive-relative BITC channels + Chebyshev MDL."""

    # Defaults = the variant selected on the dev split by the pre-set rule (RESULTS.md section 19):
    # cross-section channels only, cut penalty 10 log S.
    def __init__(self, penalty=10.0, channels=("section",), max_deg=8, min_len=3, **kw):
        kw.setdefault("cut_channels", ("hist",))
        super().__init__(**kw)
        self.penalty, self.channels, self.max_deg, self.min_len = penalty, channels, max_deg, min_len
        self.trace = []

    def _cuts(self, sp, se, P, N, C, own=None):
        Y, names = slab_channels(sp, se, P, N, C, own, self.h)
        sel = [i for i, nm in enumerate(names) if
               ("section" in self.channels and nm.startswith("a")) or
               ("topo" in self.channels and nm in ("B", "beta0", "beta1")) or
               ("hist" in self.channels and nm[0] in "nc" and nm[1:].isdigit())]
        Y = Y[:, sel]
        names = [names[i] for i in sel]
        Ys, keep = standardize(Y)
        cuts, code = mdl_partition(Ys, self.min_len, self.penalty, self.max_deg)
        self.trace.append(dict(Y=Y, names=names, cuts=cuts, code=code, edges=sp.edges))
        return cuts, [(c, 0.0, "mdl") for c in cuts]
