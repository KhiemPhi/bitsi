"""Topological BITSI: a sweep function that carries geometry and topology together.

For a sweep direction d and slab t, split the slab's cross-section (occupancy grid in the plane
normal to d) into connected components C_k, ignoring specks. For each component:

    rho_k(t)  boundary-fit ratio against the component's OWN 2-D box (local, unlike BITSI's
              global object box): fraction of its points within eps of that box's edges
    chi_k(t)  Euler characteristic, 1 - (enclosed holes), ignoring tiny holes

    B_d(t) = sum_k rho_k(t) * chi_k(t)          (integration of rho against Euler characteristic)

With rho = 1 this is the Euler characteristic curve of the sweep (the ECT); chi is additive under
union by inclusion-exclusion, so the stack of an object is the sum of its parts' stacks up to
where parts touch. Topological events (a component or hole appears / disappears, persisting for
`persist` slabs) are breakpoints; between them B is smooth and gets its own Chebyshev series.

All quantities are computed on a slab grid fixed by the OBJECT (frame, slab edges, cell origin),
so any subset of points (a part) can be evaluated on the same grid for additivity tests.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from .profiles import median_nn_spacing, sweep_frame

ST8 = np.ones((3, 3), dtype=int)


@dataclass
class SweepGrid:
    R: np.ndarray        # (3, 3) rows d, u, v
    center: np.ndarray   # (3,)
    lo: np.ndarray       # (3,) min coordinates in the frame
    hi: np.ndarray
    spacing: float
    cell: float
    eps: float
    n_slabs: int

    @property
    def t(self):
        return self.lo[0] + (np.arange(self.n_slabs) + 0.5) * self.spacing


def make_grid(points, d, h=None, spacing_h=1.5, cell_h=2.5, eps_h=1.0):
    P = np.asarray(points, float)
    h = median_nn_spacing(P) if h is None else h
    R = sweep_frame(P, d)
    c = P.mean(0)
    q = (P - c) @ R.T
    lo, hi = q.min(0), q.max(0)
    spacing = spacing_h * h
    n = max(int(np.ceil((hi[0] - lo[0]) / spacing)), 1)
    return SweepGrid(R, c, lo, hi, spacing, cell_h * h, eps_h * h, n)


def _components(gu, gv, shape, min_pts, min_hole_cells):
    """Occupied components of one slab with (point mask, chi). Holes are empty 4-connected
    regions not touching the border, with >= min_hole_cells cells, attributed to the component
    that surrounds them."""
    g = np.zeros(shape, dtype=np.uint8)
    g[gu, gv] = 1
    lab, n = label(g, ST8)
    pt_comp = lab[gu, gv]
    pad = np.pad(lab, 1)
    empty, ne = label(pad == 0)
    border = set(np.unique(np.concatenate([empty[0], empty[-1], empty[:, 0], empty[:, -1]])).tolist())
    holes = np.zeros(n + 1, dtype=int)
    for e in range(1, ne + 1):
        if e in border:
            continue
        m = empty == e
        if m.sum() < min_hole_cells:
            continue
        # surrounding component: majority occupied label on the hole's 1-cell dilation ring
        ring = np.zeros_like(m)
        ring[1:, :] |= m[:-1, :]
        ring[:-1, :] |= m[1:, :]
        ring[:, 1:] |= m[:, :-1]
        ring[:, :-1] |= m[:, 1:]
        labs = pad[ring & (pad > 0)]
        if len(labs):
            holes[np.bincount(labs).argmax()] += 1
    comps = []
    for k in range(1, n + 1):
        mk = pt_comp == k
        if mk.sum() >= min_pts:
            comps.append((mk, 1 - holes[k]))
    return comps


def slab_channels(points, grid, min_pts_frac=0.02, min_pts=3, min_hole_cells=2, obj_box=None, window=1):
    """Per-slab channels of `points` on the object's grid.

    Returns dict of (S,) arrays: count, beta0, beta1, chi, rho_sum, topo_bitsi, old_bitsi.
    old_bitsi uses the global box of `points` itself (BITSI's definition), or obj_box if given.
    window: topology and rho of slab s use the points of slabs s - window//2 .. s + window//2
    (a thicker section, so sparse surface samples connect); count stays per slab.
    """
    P = np.asarray(points, float)
    q = (P - grid.center) @ grid.R.T
    S = grid.n_slabs
    slab = np.clip(((q[:, 0] - grid.lo[0]) / grid.spacing).astype(int), 0, S - 1)
    inside = (q[:, 0] >= grid.lo[0] - 1e-9) & (q[:, 0] <= grid.hi[0] + 1e-9)
    gu = np.clip(((q[:, 1] - grid.lo[1]) / grid.cell).astype(int), 0, None)
    gv = np.clip(((q[:, 2] - grid.lo[2]) / grid.cell).astype(int), 0, None)
    shape = (int((grid.hi[1] - grid.lo[1]) / grid.cell) + 2, int((grid.hi[2] - grid.lo[2]) / grid.cell) + 2)
    gu, gv = np.minimum(gu, shape[0] - 1), np.minimum(gv, shape[1] - 1)
    blo, bhi = (q.min(0), q.max(0)) if obj_box is None else obj_box
    face = np.minimum(q - blo, bhi - q).min(1)
    out = {k: np.zeros(S) for k in ("count", "beta0", "beta1", "chi", "rho_sum", "topo_bitsi", "old_bitsi")}
    order = np.argsort(slab, kind="stable")
    bounds = np.searchsorted(slab[order], np.arange(S + 1))
    half = window // 2
    for s in range(S):
        idx = order[bounds[s]:bounds[s + 1]]
        idx = idx[inside[idx]]
        if len(idx) == 0:
            continue
        out["count"][s] = len(idx)
        idx = order[bounds[max(0, s - half)]:bounds[min(S, s + half + 1)]]
        idx = idx[inside[idx]]
        comps = _components(gu[idx], gv[idx], shape, max(min_pts, int(min_pts_frac * len(idx))), min_hole_cells)
        b0 = len(comps)
        out["beta0"][s] = b0
        out["old_bitsi"][s] = (face[idx] < grid.eps).mean() * b0
        for mk, chi in comps:
            uv = q[idx[mk]][:, 1:]
            lo2, hi2 = uv.min(0), uv.max(0)
            d_edge = np.minimum(uv - lo2, hi2 - uv).min(1)
            rho = float((d_edge < grid.eps).mean())
            out["beta1"][s] += 1 - chi
            out["chi"][s] += chi
            out["rho_sum"][s] += rho
            out["topo_bitsi"][s] += rho * chi
    return out


def topo_breakpoints(beta0, beta1, persist=2):
    """Slab boundaries where (beta0, beta1) changes to a value that then holds for >= persist slabs."""
    sig = np.stack([beta0, beta1], 1)
    S = len(sig)
    cuts = []
    s = 1
    while s < S:
        if not np.array_equal(sig[s], sig[s - 1]):
            hold = s + persist <= S and all(np.array_equal(sig[s], sig[j]) for j in range(s, s + persist))
            if hold:
                cuts.append(s - 1)  # cut after slab s-1
        s += 1
    return cuts
