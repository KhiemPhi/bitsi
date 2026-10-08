"""Primitive decomposition: an object = one swept body primitive + what it does not explain.

1. Axis. The body is modelled as a surface of revolution about an axis found from the normals
   (axes.revolution_axis): on such a surface every normal line meets the axis.
2. Profile. Along the axis, the body's outer radius R(s) per slab: a high percentile of the
   radius of points whose normal line meets the axis (body points), interpolated over empty
   slabs and median-smoothed. R(s) is the primitive's generating function (a BITSI-style sweep
   profile, measuring radius instead of slice topology).
3. Coaxial parts (lid, cap, neck, base) change the profile: Chebyshev inflection cuts on R(s)
   that persist at the coarsest `persist` levels (cheb.multiscale_cuts) split the body along s.
4. Off-axis parts (handle, spout, knob) are the residual: points farther than R(s) + margin
   from the axis. Residual points are grouped into connected components (radius graph); each
   component of >= min_part_frac of the points is a part, smaller ones go back to the body.

No labels, no images, no learned features. Returns per-point labels and the fitted primitive.
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .axes import closed_end_sign, revolution_axis
from .cheb import multiscale_cuts
from .profiles import median_nn_spacing


def frame(a):
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def body_profile(P, N, a, c, n_slabs=40, q=0.9, meet_cos=0.95, smooth=5):
    """(s, r, slab index, slab centres, R per slab, body-point mask) for axis (a, c)."""
    Q = P - c
    s = Q @ a
    X = Q - np.outer(s, a)
    r = np.linalg.norm(X, axis=1)
    Nn = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    nin = Nn - np.outer(Nn @ a, a)
    m = np.linalg.norm(nin, axis=1)
    meets = (m > 0.5) & (np.abs(np.sum(nin * X, 1)) / (m * r + 1e-12) > meet_cos)
    edges = np.linspace(s.min(), s.max(), n_slabs + 1)
    sl = np.clip(np.digitize(s, edges) - 1, 0, n_slabs - 1)
    tc = 0.5 * (edges[:-1] + edges[1:])
    R = np.full(n_slabs, np.nan)
    for k in range(n_slabs):
        mk = (sl == k) & meets
        if mk.sum() >= 10:
            R[k] = np.quantile(r[mk], q)
    ok = ~np.isnan(R)
    if ok.sum() == 0:
        R[:] = np.quantile(r, q)
    else:
        R = np.interp(tc, tc[ok], R[ok])
    R = median_filter(R, smooth, mode="nearest")
    return s, r, sl, tc, R, meets


def primitive_decomposition(points, normals, colors=None, n_slabs=40, margin_h=3.0, persist=2, min_part_frac=0.01,
                            graph_h=3.0, axial_cuts=True, residual=True, axial="cheb", max_cover=0.5):
    """axial: "cheb" = Chebyshev inflection cuts of the radius profile R(s); "hog" = significant
    normal/colour histogram events (sweep_events, z > 3) of the BODY points along the symmetry
    axis, i.e. the HOG sweep detector run in the primitive's frame with the residual removed;
    "hog_pca" = the same events along the object's longest PCA axis (the HOG baseline's direction)."""
    P = np.asarray(points, float)
    N = np.asarray(normals, float)
    h = median_nn_spacing(P)
    a, c, score = revolution_axis(P, N)
    a = a * closed_end_sign(P, a, c)
    s, r, sl, tc, R, meets = body_profile(P, N, a, c, n_slabs)
    # 1. off-axis parts: residual components that break the rotational symmetry
    part = np.full(len(P), -1)
    nres = 0
    if residual:
        idx = np.flatnonzero(r > R[sl] + margin_h * h)
        if len(idx):
            pairs = cKDTree(P[idx]).query_pairs(graph_h * h, output_type="ndarray")
            G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
            ncomp, comp = connected_components(G, directed=False)
            sizes = np.bincount(comp, minlength=ncomp)
            u, v = frame(a)
            Q = P[idx] - c
            az = np.arctan2(Q @ v, Q @ u)
            for k in np.argsort(-sizes):
                if sizes[k] < min_part_frac * len(P):
                    break
                mk = comp == k
                # an off-axis part covers only part of the azimuth; a component wrapping the
                # axis (a cap wider than the smoothed profile) is coaxial and stays with the body
                bins = ((az[mk] + np.pi) / (2 * np.pi) * 36).astype(int) % 36
                if np.count_nonzero(np.bincount(bins, minlength=36)) / 36 > max_cover:
                    continue
                part[idx[mk]] = nres
                nres += 1
    res = part >= 0
    body = ~res
    # 2. coaxial parts: cuts of the body along the axis
    lab = np.zeros(len(P), int)
    cuts = []
    if axial_cuts and axial == "cheb":
        cs, _ = multiscale_cuts(tc, R)
        cuts = sorted(cu.t for cu in cs if min(cu.levels) <= persist)
        lab = np.searchsorted(np.asarray(cuts), s)
    elif axial_cuts and axial in ("hog", "hog_pca") and body.sum() > 50:
        from .profiles import pca_axes
        from .sweep_events import events_along
        d = a if axial == "hog" else pca_axes(P)[0]
        ev, prof = events_along(P[body], d, N[body], None if colors is None else colors[body], ("hist",))
        cuts = sorted(cu.t for cu in ev)
        lab[body] = np.searchsorted(np.asarray(cuts), prof["coord"])
    lab[res] = lab.max() + 1 + part[res]
    return lab, dict(axis=a, centre=c, axis_score=score, t=tc, R=R, cuts=cuts, residual=res,
                     n_axial=len(cuts) + 1, n_residual=nres)
