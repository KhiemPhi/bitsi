"""Part-count-free cut selection on multi-channel sweep profiles (change-point detection).

Each slab along a sweep becomes one feature vector:
  scalar channels  BITC, beta_0, beta_1, mean CIELAB L, a, b
  histograms       normal orientation vs the sweep axis (6 bins), CIELAB colour (12 bins),
                   as square-root proportions (Euclidean distance ~ Hellinger)
  gripper          fraction of the slab's points lying in cross-section pieces whose minor
                   width is <= G (metric), i.e. "is there something the jaws can close on here"
Every feature is divided by its robust noise (first-difference MAD), so the features are on a
common unit-noise scale. Cuts are chosen by optimal partitioning:

    min over cut sets  sum_segments sum_{s in seg} ||x_s - mean_seg||^2  +  (D + 1) log S * #cuts

the BIC penalty for a unit-variance Gaussian with a mean change in D dims at an unknown location.
Candidate locations are either every slab boundary, or only the multi-scale Chebyshev inflection
positions of the channels (the "Chebyshev candidates" variant). No part count, no threshold.
"""
import numpy as np
from scipy.ndimage import label

from .cheb import multiscale_cuts
from .profiles import multichannel_profiles, pca_axes, sweep_directions
from .sweep_events import robust_noise


def slab_features(points, normals, colors, d, scale=None, G=0.08, channels=("scalar", "hist", "grip")):
    """(S, D) standardized feature matrix, the profile dict, and feature group names."""
    prof = multichannel_profiles(points, d, colors, channels=("geo", "topo", "color"))
    S = len(prof["t"])
    slab_of = prof["slab_of"]
    cols, names = [], []
    if "scalar" in channels:
        for nm, y in prof["channels"].items():
            cols.append(np.asarray(y, float)[:, None])
            names.append(nm)
    if "hist" in channels:
        H = []
        if normals is not None:
            n = np.asarray(normals, float)
            n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
            b = np.minimum((np.abs(n @ d) * 6).astype(int), 5)
            Hn = np.zeros((S, 6))
            np.add.at(Hn, (slab_of, b), 1)
            H.append(("hist_normal", Hn))
        if colors is not None:
            from skimage.color import rgb2lab
            c = np.asarray(colors, float)
            c = c / 255.0 if c.max() > 1.5 else c
            lab = rgb2lab(c[None])[0]
            Hc = np.zeros((S, 12))
            np.add.at(Hc, (slab_of, np.clip((lab[:, 0] / 25).astype(int), 0, 3)), 1)
            np.add.at(Hc, (slab_of, 4 + np.clip(((lab[:, 1] + 40) / 20).astype(int), 0, 3)), 1)
            np.add.at(Hc, (slab_of, 8 + np.clip(((lab[:, 2] + 40) / 20).astype(int), 0, 3)), 1)
            H.append(("hist_color", Hc))
        for nm, Hm in H:
            tot = Hm.sum(1, keepdims=True)
            P = np.sqrt(Hm / np.maximum(tot, 1))
            empty = tot[:, 0] == 0
            if empty.any() and (~empty).any():  # carry neighbours into empty slabs
                for j in range(P.shape[1]):
                    P[empty, j] = np.interp(np.where(empty)[0], np.where(~empty)[0], P[~empty, j])
            cols.append(P)
            names += [nm] * P.shape[1]
    if "grip" in channels and scale is not None:
        cols.append(grip_profile(points, prof, scale, G)[:, None])
        names.append("grip")
    X = np.hstack(cols) if cols else np.zeros((S, 0))
    for j in range(X.shape[1]):
        X[:, j] = X[:, j] / robust_noise(X[:, j])
    return X, prof, names


def grip_profile(points, prof, scale, G):
    """Fraction of each slab's points in cross-section pieces of minor width <= G (metres)."""
    R = prof["frame"]
    P = np.asarray(points, float)
    q = (P - P.mean(0)) @ R.T
    lo = q.min(0)
    gr = prof["grid_res"]
    gu = ((q[:, 1] - lo[1]) / gr).astype(int)
    gv = ((q[:, 2] - lo[2]) / gr).astype(int)
    shape = (gu.max() + 1, gv.max() + 1)
    slab_of = prof["slab_of"]
    S = len(prof["t"])
    out = np.zeros(S)
    order = np.argsort(slab_of, kind="stable")
    bounds = np.searchsorted(slab_of[order], np.arange(S + 1))
    st = np.ones((3, 3), dtype=int)
    for s in range(S):
        idx = order[bounds[s]:bounds[s + 1]]
        if len(idx) < 3:
            continue
        g = np.zeros(shape, dtype=np.int32)
        g[gu[idx], gv[idx]] = 1
        lab, n = label(g, st)
        comp = lab[gu[idx], gv[idx]]
        ok = 0
        for c in range(1, n + 1):
            m = comp == c
            if m.sum() < 3:
                continue
            uv = q[idx[m]][:, 1:] * scale
            uv = uv - uv.mean(0)
            # minor extent of the piece (PCA), ~ how wide the jaws must open
            vt = np.linalg.svd(uv, full_matrices=False)[2] if len(uv) > 2 else np.eye(2)
            minor = np.ptp(uv @ vt[-1]) if len(uv) > 2 else 0.0
            if minor <= G:
                ok += m.sum()
        out[s] = ok / len(idx)
    return out


def optimal_partition(X, candidates, penalty, model="linear"):
    """Exact optimal partitioning of rows of X with cuts allowed only after rows in `candidates`.

    model="mean": piecewise-constant segments (SSE about the segment mean);
    model="linear": piecewise-linear segments (SSE about a per-column line in the slab index), so
    a gradually tapering or shading part costs nothing and only real changes are cut.
    A cut 'after row i' separates rows <= i from rows > i. Returns the sorted list of cut rows.
    """
    S = len(X)
    x = np.arange(S, dtype=float)
    c0 = np.concatenate([[0.0], np.cumsum(np.ones(S))])
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    cy = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
    cxy = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X * x[:, None], 0)])
    cyy = np.concatenate([[0.0], np.cumsum((X ** 2).sum(1))])

    def cost(a, b):  # rows a..b-1
        n = c0[b] - c0[a]
        if n <= 0:
            return 0.0
        sy = cy[b] - cy[a]
        base = (cyy[b] - cyy[a]) - (sy @ sy) / n
        if model == "mean" or n < 3:
            return base
        sx = c1[b] - c1[a]
        sxx = (c2[b] - c2[a]) - sx * sx / n
        if sxx <= 0:
            return base
        sxy = (cxy[b] - cxy[a]) - sx * sy / n
        return base - (sxy @ sxy) / sxx

    ends = sorted(set(int(c) + 1 for c in candidates if 0 <= c < S - 1)) + [S]
    # F[e] = best cost of rows [0, e) with a segment ending at e
    F = {0: 0.0}
    back = {}
    for k, e in enumerate(ends):
        best, arg = np.inf, None
        for s in [0] + ends[:k]:
            v = F[s] + cost(s, e) + (penalty if s > 0 else 0.0)
            if v < best:
                best, arg = v, s
        F[e], back[e] = best, arg
    cuts = []
    e = S
    while e > 0:
        s = back[e]
        if s > 0:
            cuts.append(s - 1)
        e = s
    return sorted(cuts)


def chebyshev_candidates(prof, X, max_deg=16):
    """Slab boundaries nearest to multi-scale Chebyshev inflections of every feature column."""
    t = prof["t"]
    S = len(t)
    cand = set()
    for j in range(X.shape[1]):
        cuts, _ = multiscale_cuts(t, X[:, j], max_deg=max_deg)
        for c in cuts:
            cand.add(int(np.clip(np.searchsorted(t, c.t) - 1, 0, S - 2)))
    return sorted(cand)


def mdl_cuts(points, d, normals=None, colors=None, scale=None, G=0.08,
             channels=("scalar", "hist", "grip"), candidates="all", penalty_scale=1.0, model="linear"):
    """Cut positions (in the sweep coordinate) chosen by BIC optimal partitioning."""
    X, prof, _ = slab_features(points, normals, colors, d, scale, G, channels)
    S, D = X.shape
    if S < 4 or D == 0:
        return [], prof, 0.0
    cand = range(S - 1) if candidates == "all" else chebyshev_candidates(prof, X)
    k = 2 * D + 1 if model == "linear" else D + 1  # parameters added by one cut
    pen = penalty_scale * k * np.log(S)
    cut_rows = optimal_partition(X, cand, pen, model)
    t = prof["t"]
    ts = [0.5 * (t[i] + t[i + 1]) for i in cut_rows]
    # evidence for choosing the sweep direction: cost reduction of the chosen partition
    def sse(a, b):
        seg = X[a:b]
        if len(seg) < 3 or model == "mean":
            return ((seg - seg.mean(0)) ** 2).sum() if len(seg) else 0.0
        xx = np.arange(len(seg), dtype=float)
        A = np.c_[np.ones_like(xx), xx]
        coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
        return ((seg - A @ coef) ** 2).sum()
    bounds = [0] + [i + 1 for i in cut_rows] + [S]
    gain = sse(0, S) - sum(sse(a, b) for a, b in zip(bounds[:-1], bounds[1:])) - pen * len(cut_rows)
    return ts, prof, float(gain)


def recursive_mdl(points, graph, normals=None, colors=None, scale=None, G=0.08,
                  channels=("scalar", "hist", "grip"), candidates="all", gripper_stop=False,
                  min_frac=0.02, max_leaves=48, model="linear"):
    """Part-count-free recursive decomposition.

    Each piece is swept along its PCA axes and hemisphere directions; the direction whose BIC
    partition gains most is applied (all its cuts at once), each slab group is split into
    connected components, and pieces are refined until no direction yields a cut. With
    gripper_stop, a piece that a gripper of opening G can already hold (box test on its
    min-volume OBB) is not split further: the part "fits".
    Returns point labels of the leaves.
    """
    from .divisive import split_by_cut
    from .feasibility import box_test
    n = len(points)
    min_points = max(int(min_frac * n), 20)
    leaves, stack = [], [np.arange(n)]
    while stack:
        idx = stack.pop()
        if len(leaves) + len(stack) >= max_leaves or len(idx) < 2 * min_points:
            leaves.append(idx)
            continue
        if gripper_stop and scale is not None and box_test(points[idx] * scale, G):
            leaves.append(idx)
            continue
        best = None
        P = points[idx]
        for d in sweep_directions(P, n_extra=6):
            ts, prof, gain = mdl_cuts(P, d, None if normals is None else normals[idx],
                                      None if colors is None else colors[idx], scale, G, channels, candidates,
                                      model=model)
            if ts and (best is None or gain > best[0]):
                best = (gain, d, ts, prof)
        if best is None:
            leaves.append(idx)
            continue
        parts = split_by_cut(points, idx, graph, best[1], best[2], best[3], min_points)
        if parts is None:
            leaves.append(idx)
            continue
        stack.extend(parts)
    lab = np.empty(n, dtype=int)
    for k, idx in enumerate(leaves):
        lab[idx] = k
    return lab
