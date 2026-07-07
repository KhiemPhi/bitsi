"""Antipodal grasp sampling + force-closure test on a part point cloud.

This is the *analytic* grasp metric. It is used two ways:
  1. training time  -> reward for the RL cutting policy (grasp-success on the
     decomposed part), no simulator needed;
  2. sim time       -> propose candidate grasps for bitsi_sim to validate under
     physics (grasp_eval.py).

A parallel-jaw grasp on a pair of surface points (p_i, p_j) is in force closure
(planar, Coulomb friction, 2 contacts) iff BOTH contact normals lie within the
friction cone about the grasp line -- i.e. the line p_i->p_j is anti-parallel to
each surface normal to within atan(mu). This is the standard antipodal condition.
"""
from __future__ import annotations

import numpy as np

from .geometry import Grasp, _unit


def estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Estimate outward-ish normals via local PCA (smallest-eigenvector).

    Sign is not disambiguated globally; the antipodal test below is sign-robust
    (it checks |alignment|). If you already have normals (ShapeNetPart provides
    them), pass those instead -- they are cleaner.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    normals = np.zeros_like(pts)
    if n < 3:
        normals[:, 2] = 1.0
        return normals
    # brute-force kNN (parts are small; fine for a few thousand points)
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    kk = min(k, n)
    _, idx = tree.query(pts, k=kk)
    for i in range(n):
        nbrs = pts[idx[i]]
        cov = np.cov((nbrs - nbrs.mean(axis=0)).T)
        w, v = np.linalg.eigh(cov)
        normals[i] = v[:, 0]  # smallest eigenvalue -> surface normal
    return normals


def _friction_ok(line_dir: np.ndarray, normal: np.ndarray, cos_thresh: float) -> bool:
    """True if ``normal`` is within the friction cone about ``line_dir`` (sign-robust)."""
    return abs(float(np.dot(_unit(line_dir), _unit(normal)))) >= cos_thresh


def sample_antipodal_grasps(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    max_width: float = 0.085,
    friction: float = 0.5,
    n_samples: int = 512,
    seed: int = 0,
    min_pairs: int = 1,
) -> list[Grasp]:
    """Sample antipodal (force-closure) parallel-jaw grasps on a point cloud.

    Args:
        points:   (N,3) part surface points (world frame).
        normals:  (N,3) surface normals; estimated via PCA if None.
        max_width: gripper stroke (m); pairs farther apart than this are rejected.
        friction: Coulomb coefficient -> cone half-angle atan(friction).
        n_samples: number of candidate point pairs to try.
        seed:     RNG seed (vary per call for stochastic reward estimates).
    Returns:
        Grasps sorted by score (best first). Empty list if none found.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return []
    if normals is None:
        normals = estimate_normals(pts)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)

    rng = np.random.default_rng(seed)
    cos_thresh = np.cos(np.arctan(friction))
    grasps: list[Grasp] = []

    ia = rng.integers(0, len(pts), size=n_samples)
    ib = rng.integers(0, len(pts), size=n_samples)
    for i, j in zip(ia, ib):
        if i == j:
            continue
        pi, pj = pts[i], pts[j]
        line = pj - pi
        width = float(np.linalg.norm(line))
        if width < 1e-4 or width > max_width:
            continue
        if not (_friction_ok(line, normals[i], cos_thresh)
                and _friction_ok(line, normals[j], cos_thresh)):
            continue
        jaw_axis = _unit(line)
        center = 0.5 * (pi + pj)
        # approach: perpendicular to the jaw axis, biased to come from +Z (top-down)
        up = np.array([0.0, 0.0, 1.0])
        approach = -_unit(up - np.dot(up, jaw_axis) * jaw_axis)
        if np.linalg.norm(approach) < 1e-6:  # jaw axis is vertical; pick any perp
            approach = _unit(np.cross(jaw_axis, np.array([1.0, 0.0, 0.0])))
        # score: antipodal alignment margin (higher = more robust)
        score = min(abs(np.dot(jaw_axis, _unit(normals[i]))),
                    abs(np.dot(jaw_axis, _unit(normals[j]))))
        grasps.append(Grasp(center=center, approach=approach,
                            jaw_axis=jaw_axis, width=width, score=float(score)))

    grasps.sort(key=lambda g: g.score, reverse=True)
    return grasps if len(grasps) >= min_pairs else grasps


def grasp_success_rate(points: np.ndarray, normals: np.ndarray | None = None,
                       max_width: float = 0.085, **kw) -> float:
    """Analytic graspability in [0,1]: does a force-closure grasp exist?

    Returns the (clipped) best antipodal score, or 0.0 if no valid grasp exists.
    Suitable as a dense-ish reward signal for the cutting policy.
    """
    g = sample_antipodal_grasps(points, normals, max_width=max_width, **kw)
    return float(g[0].score) if g else 0.0
