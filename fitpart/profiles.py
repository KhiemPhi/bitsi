"""BITC profiles swept along arbitrary directions (Sec. 4.3, "Boundary-fit profiles").

Vectorized re-implementation of `bitsi_slicer.slicing.accumulate_slices`: for each
slab perpendicular to a sweep direction, the profile value is

    (fraction of slab points within eps of the object's box faces)
        x (number of occupied connected components in the slab cross-section)

The box is the object's box in the sweep frame [d, u, v], where u, v are the
principal axes of the points projected onto the plane perpendicular to d. For the
three PCA directions this is the PCA-aligned box, matching the original code's OBB.
"""
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree


def median_nn_spacing(points):
    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))


def pca_axes(points):
    c = points - points.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt  # rows = principal axes, largest variance first


def fibonacci_hemisphere(n):
    """n roughly uniform unit directions on the upper hemisphere (z >= 0)."""
    i = np.arange(n) + 0.5
    z = 1.0 - i / n  # (0, 1]
    r = np.sqrt(1.0 - z ** 2)
    phi = np.pi * (1.0 + 5 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def sweep_directions(points, n_extra=9, min_angle_deg=15.0):
    """The three PCA axes plus hemisphere directions, in the PCA frame.

    Extra directions within `min_angle_deg` (as lines, sign-free) of an
    already-chosen direction are dropped, so the count can be below 3 + n_extra.
    """
    axes = pca_axes(points)
    dirs = [a for a in axes]
    cos_min = np.cos(np.deg2rad(min_angle_deg))
    for h in fibonacci_hemisphere(n_extra):
        d = h @ axes  # hemisphere expressed in the PCA frame
        d /= np.linalg.norm(d)
        if all(abs(d @ e) < cos_min for e in dirs):
            dirs.append(d)
    return np.array(dirs)


def sweep_frame(points, d):
    """Orthonormal frame with d first and the in-plane principal axes after it."""
    d = d / np.linalg.norm(d)
    c = points - points.mean(0)
    proj = c - np.outer(c @ d, d)
    _, _, vt = np.linalg.svd(proj, full_matrices=False)
    u = vt[0] - (vt[0] @ d) * d
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return np.stack([d, u, v])


def bitc_profile(points, d, spacing=None, eps=None, grid_res=None,
                 use_components=True, min_slab_points=1):
    """BITC profile of `points` swept along direction `d`.

    Parameters default to multiples of the median nearest-neighbour spacing h so the
    profile is scale-free: slab thickness 1.5h, face tolerance eps = 1.0h, and
    cross-section grid cells of 2.5h (coarse enough that a solid section is one
    connected component at typical sampling density).

    Returns a dict with
      t        : slab centres along d (in the frame's units), shape (S,)
      values   : profile value per slab (0 for empty slabs), shape (S,)
      valid    : slab has >= min_slab_points points, shape (S,)
      slab_of  : slab index of every point, shape (N,)
      coord    : coordinate of every point along d, shape (N,)
      frame    : (3, 3) rows d, u, v
    """
    points = np.asarray(points, dtype=np.float64)
    h = median_nn_spacing(points)
    spacing = 1.5 * h if spacing is None else spacing
    eps = 1.0 * h if eps is None else eps
    grid_res = 2.5 * h if grid_res is None else grid_res

    R = sweep_frame(points, d)
    q = (points - points.mean(0)) @ R.T  # coordinates in [d, u, v]
    lo, hi = q.min(0), q.max(0)

    # distance of each point to the nearest of the six box faces
    face_dist = np.minimum(q - lo, hi - q).min(1)
    near = face_dist < eps

    n_slabs = max(int(np.ceil((hi[0] - lo[0]) / spacing)), 1)
    slab_of = np.minimum(((q[:, 0] - lo[0]) / spacing).astype(int), n_slabs - 1)
    t = lo[0] + (np.arange(n_slabs) + 0.5) * spacing

    counts = np.bincount(slab_of, minlength=n_slabs)
    near_counts = np.bincount(slab_of, weights=near, minlength=n_slabs)
    valid = counts >= min_slab_points
    ratio = np.where(counts > 0, near_counts / np.maximum(counts, 1), 0.0)

    if use_components:
        gu = ((q[:, 1] - lo[1]) / grid_res).astype(int)
        gv = ((q[:, 2] - lo[2]) / grid_res).astype(int)
        shape = (gu.max() + 1, gv.max() + 1)
        order = np.argsort(slab_of, kind="stable")
        bounds = np.searchsorted(slab_of[order], np.arange(n_slabs + 1))
        comps = np.zeros(n_slabs)
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        for s in range(n_slabs):
            idx = order[bounds[s]:bounds[s + 1]]
            if len(idx) == 0:
                continue
            grid = np.zeros(shape, dtype=np.uint8)
            grid[gu[idx], gv[idx]] = 1
            comps[s] = label(grid, structure)[1]
        values = ratio * comps
    else:
        values = ratio

    return dict(t=t, values=values, valid=valid, slab_of=slab_of, coord=q[:, 0],
                frame=R, spacing=spacing, eps=eps, grid_res=grid_res, h=h)


def _holes(grid):
    """beta_1 of a binary cross-section: empty components that do not reach the border."""
    pad = np.pad(grid, 1)
    lab, n = label(1 - pad)  # 4-connected empty regions
    if n == 0:
        return 0
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return int(n - len(set(border.tolist()) - {0}))


def multichannel_profiles(points, d, colors=None, channels=("geo", "topo", "color"), spacing=None,
                          eps=None, grid_res=None):
    """Several 1-D profiles over the same slabs along direction d.

    geo   : BITC (boundary ratio x occupied components), as `bitc_profile`
    topo  : beta_0 (occupied components) and beta_1 (enclosed holes) of each slab's cross-section
    color : mean CIELAB L, a, b of each slab's points (empty slabs interpolated)

    Returns the `bitc_profile` dict plus `channels`: {name: values} with names geo, beta0, beta1,
    L, a, b (only those requested / available).
    """
    base = bitc_profile(points, d, spacing=spacing, eps=eps, grid_res=grid_res, use_components=True)
    out = {}
    if "geo" in channels:
        out["geo"] = base["values"]
    S = len(base["t"])
    slab_of = base["slab_of"]
    if "topo" in channels:
        R = base["frame"]
        q = (np.asarray(points, float) - np.asarray(points, float).mean(0)) @ R.T
        lo = q.min(0)
        gr = base["grid_res"]
        gu = ((q[:, 1] - lo[1]) / gr).astype(int)
        gv = ((q[:, 2] - lo[2]) / gr).astype(int)
        shape = (gu.max() + 1, gv.max() + 1)
        b0 = np.zeros(S)
        b1 = np.zeros(S)
        order = np.argsort(slab_of, kind="stable")
        bounds = np.searchsorted(slab_of[order], np.arange(S + 1))
        st = np.ones((3, 3), dtype=int)
        for s in range(S):
            idx = order[bounds[s]:bounds[s + 1]]
            if len(idx) == 0:
                continue
            g = np.zeros(shape, dtype=np.uint8)
            g[gu[idx], gv[idx]] = 1
            b0[s] = label(g, st)[1]
            b1[s] = _holes(g)
        out["beta0"], out["beta1"] = b0, b1
    if "color" in channels and colors is not None:
        from skimage.color import rgb2lab
        c = np.asarray(colors, float)
        if c.max() > 1.5:
            c = c / 255.0
        lab = rgb2lab(c[None])[0]
        cnt = np.bincount(slab_of, minlength=S)
        for k, nm in enumerate("Lab"):
            m = np.bincount(slab_of, weights=lab[:, k], minlength=S) / np.maximum(cnt, 1)
            ok = cnt > 0
            if ok.any() and not ok.all():
                m = np.interp(np.arange(S), np.where(ok)[0], m[ok])
            out[nm] = m
    base["channels"] = out
    return base
