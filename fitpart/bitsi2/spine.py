"""Spines of swept primitives: per-point arc-length coordinate s plus a local frame per slab.

straight spine   s = (p - c) . a for an axis (a, c): the symmetry axis (axes.revolution_axis)
                 or the longest PCA axis. Slab centres lie on the axis line (symmetry axis) or
                 at slab centroids (PCA).
curved spine     a principal curve through the points: s = graph-geodesic distance (kNN graph)
                 from one end of the longest geodesic (double sweep), slab centres = slab
                 centroids, tangents = smoothed centroid differences. For handles, arms, cords.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from ..axes import closed_end_sign, revolution_axis
from ..profiles import pca_axes


@dataclass
class Spine:
    kind: str                 # "revolution" | "pca" | "geodesic"
    s: np.ndarray             # (N,) arc-length coordinate per point
    slab: np.ndarray          # (N,) slab index per point
    centres: np.ndarray       # (S, 3) spine point per slab
    tangents: np.ndarray      # (S, 3) unit tangent per slab
    edges: np.ndarray         # (S + 1,) slab edges in s
    score: float = 0.0        # axis score (revolution) or 0
    info: dict = field(default_factory=dict)

    @property
    def n_slabs(self):
        return len(self.centres)


def _slabs(s, spacing, min_slabs=6, max_slabs=80):
    L = s.max() - s.min()
    S = int(np.clip(np.ceil(L / max(spacing, 1e-12)), min_slabs, max_slabs))
    edges = np.linspace(s.min(), s.max() + 1e-9, S + 1)
    return edges, np.clip(np.digitize(s, edges) - 1, 0, S - 1)


def _smooth_rows(X, w=2):
    out = X.copy()
    for i in range(len(X)):
        out[i] = X[max(0, i - w):i + w + 1].mean(0)
    return out


def straight_spine(P, axis, centre=None, spacing=None, kind="pca", score=0.0):
    a = axis / np.linalg.norm(axis)
    c = P.mean(0) if centre is None else centre
    s = (P - c) @ a
    edges, slab = _slabs(s, spacing)
    S = len(edges) - 1
    mids = 0.5 * (edges[:-1] + edges[1:])
    if kind == "revolution":
        centres = c + np.outer(mids, a)
    else:
        # slab centroids, smoothed along s, moved onto the slab mid-plane
        cen = np.array([P[slab == k].mean(0) if (slab == k).any() else c + mids[k] * a for k in range(S)])
        cen = _smooth_rows(cen, 1)
        centres = cen + np.outer(mids - (cen - c) @ a, a)
    return Spine(kind, s, slab, centres, np.tile(a, (S, 1)), edges, score)


def revolution_spine(P, N, spacing):
    a, c, score = revolution_axis(P, N)
    a = a * closed_end_sign(P, a, c)
    return straight_spine(P, a, c, spacing, kind="revolution", score=score)


def pca_spine(P, spacing):
    return straight_spine(P, pca_axes(P)[0], None, spacing, kind="pca")


def geodesic_spine(P, spacing, k=10, max_pts=3000, seed=0):
    n = len(P)
    idx = np.arange(n) if n <= max_pts else np.random.default_rng(seed).choice(n, max_pts, replace=False)
    Q = P[idx]
    kk = min(k, len(Q) - 1)
    d, nb = cKDTree(Q).query(Q, kk + 1)
    rows = np.repeat(np.arange(len(Q)), kk)
    G = coo_matrix((d[:, 1:].ravel(), (rows, nb[:, 1:].ravel())), shape=(len(Q), len(Q))).tocsr()
    G = G.maximum(G.T)
    ncomp, comp = connected_components(G, directed=False)
    if ncomp > 1:  # bridge components to the largest one through nearest points
        big = np.bincount(comp).argmax()
        tb = cKDTree(Q[comp == big])
        bi = np.flatnonzero(comp == big)
        G = G.tolil()
        for cc in range(ncomp):
            if cc == big:
                continue
            m = np.flatnonzero(comp == cc)
            dd, j = tb.query(Q[m])
            i0 = int(np.argmin(dd))
            G[m[i0], bi[j[i0]]] = G[bi[j[i0]], m[i0]] = dd[i0]
        G = G.tocsr()
    d0 = dijkstra(G, indices=0)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    da = dijkstra(G, indices=a)
    sq = np.where(np.isfinite(da), da, np.nanmax(da[np.isfinite(da)]))
    if n > max_pts:  # propagate to all points by nearest sample
        s = sq[cKDTree(Q).query(P)[1]]
    else:
        s = sq
    edges, slab = _slabs(s, spacing)
    S = len(edges) - 1
    centres = np.array([P[slab == k].mean(0) if (slab == k).any() else np.full(3, np.nan) for k in range(S)])
    ok = ~np.isnan(centres[:, 0])
    for j in range(3):
        centres[:, j] = np.interp(np.arange(S), np.flatnonzero(ok), centres[ok, j])
    centres = _smooth_rows(centres, 1)
    T = np.gradient(centres, axis=0)
    T = _smooth_rows(T, 2)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    return Spine("geodesic", s, slab, centres, T, edges, 0.0, dict(end=int(idx[a])))
