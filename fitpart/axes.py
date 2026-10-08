"""Sweep-axis selection from surface normals (label-free).

The axis of a generalized cylinder is the direction most perpendicular to the surface normals:
the eigenvector with the smallest eigenvalue of the normal tensor T = sum_i n_i n_i^T. For a mug
or bottle body it is the symmetry axis regardless of the mesh frame; for a pan it is the handle
direction. Confidence = 1 - lambda_min / lambda_mid (1 = perfect cylinder-like sweep, 0 = no
preferred direction).
"""
import numpy as np


def normal_tensor_axis(normals, weights=None):
    n = np.asarray(normals, float)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    w = np.ones(len(n)) if weights is None else np.asarray(weights, float)
    T = (n * w[:, None]).T @ n / w.sum()
    vals, vecs = np.linalg.eigh(T)  # ascending
    axis = vecs[:, 0]
    conf = 1.0 - vals[0] / max(vals[1], 1e-12)
    return axis, float(conf), vals


def base_down_sign(points, normals, axis, cap_cos=0.85):
    """+1 if the closed end (cap normals parallel to the axis) is at the low end along +axis,
    else -1. An open container (cup, mug) has a cap only at its base."""
    a = axis / np.linalg.norm(axis)
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    s = points @ a
    cap = np.abs(n @ a) > cap_cos
    if cap.sum() < 10:
        return 1
    mid = 0.5 * (s.min() + s.max())
    return 1 if (s[cap] < mid).sum() >= (s[cap] >= mid).sum() else -1


def rotation_to_up(axis):
    """Rotation taking `axis` to +z."""
    u = axis / np.linalg.norm(axis)
    z = np.array([0, 0, 1.0])
    v = np.cross(u, z)
    c = u @ z
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _fib_hemisphere(n):
    i = np.arange(n) + 0.5
    z = i / n                                  # z in (0, 1]: one direction per +/- pair
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z ** 2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def _revolution_score(P, N, dirs, tol, min_inplane=0.5, iters=3):
    """For each candidate axis direction: project points and normals into the plane normal to it.
    On a surface of revolution every projected normal line passes through one point (the axis).
    Fit that point by robust least squares; score = fraction of ALL points whose normal line
    passes within tol of it. Caps (normals along the axis) and handles (normals around their own
    centreline) do not vote."""
    scores, centres = np.zeros(len(dirs)), np.zeros((len(dirs), 2))
    for k, a in enumerate(dirs):
        u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
        v = np.cross(a, u)
        p = np.stack([P @ u, P @ v], 1)
        n = np.stack([N @ u, N @ v], 1)
        m = np.linalg.norm(n, axis=1)
        keep = m > min_inplane
        if keep.sum() < 20:
            continue
        p, n = p[keep], n[keep] / m[keep, None]
        t = np.stack([-n[:, 1], n[:, 0]], 1)    # line normal: distance from c to the line = |t.(c - p)|
        w = np.ones(len(p))
        for _ in range(iters):
            A = (t * w[:, None]).T @ t
            b = (t * w[:, None]).T @ np.sum(t * p, 1)
            c = np.linalg.lstsq(A, b, rcond=None)[0]
            r = np.abs(np.sum(t * (c - p), 1))
            w = (r < 2 * tol).astype(float) + 1e-3
        scores[k] = (r < tol).sum() / len(P)
        centres[k] = c
    return scores, centres


def revolution_axis(points, normals, n_dirs=600, tol_frac=0.02, refine=True, max_pts=4000, seed=0):
    """Label-free symmetry axis of a (roughly) rotationally symmetric body: the direction whose
    normal lines best meet a common line. Returns (axis, point_on_axis, score)."""
    P = np.asarray(points, float); N = np.asarray(normals, float)
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    if len(P) > max_pts:
        idx = np.random.default_rng(seed).choice(len(P), max_pts, replace=False)
        P, N = P[idx], N[idx]
    c0 = P.mean(0); P = P - c0
    tol = tol_frac * np.linalg.norm(np.ptp(P, 0))
    dirs = _fib_hemisphere(n_dirs)
    s, _ = _revolution_score(P, N, dirs, tol)
    best = dirs[np.argmax(s)]
    if refine:  # local search within ~6 degrees
        rng = np.random.default_rng(seed)
        loc = best + 0.1 * rng.standard_normal((200, 3)); loc /= np.linalg.norm(loc, axis=1, keepdims=True)
        loc = np.vstack([best, loc])
        s2, _ = _revolution_score(P, N, loc, tol)
        best = loc[np.argmax(s2)]
    s, c = _revolution_score(P, N, best[None], tol)
    a = best / np.linalg.norm(best)
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return a, c0 + c[0, 0] * u + c[0, 1] * v, float(s[0])


def closed_end_sign(points, axis, centre, end_frac=0.08, core_frac=0.35):
    """+1 if the closed end is at the low end along +axis, else -1. A closed end is a disc: it has
    points near the axis; an open end (a rim) has none. Counts points within core_frac of the
    body radius of the axis, in the end_frac of the length at each end."""
    a = axis / np.linalg.norm(axis)
    q = points - centre
    s = q @ a
    r = np.linalg.norm(q - np.outer(s, a), axis=1)
    R = np.median(r)
    L = s.max() - s.min()
    core = r < core_frac * R
    lo = (core & (s < s.min() + end_frac * L)).sum()
    hi = (core & (s > s.max() - end_frac * L)).sum()
    return 1 if lo >= hi else -1
