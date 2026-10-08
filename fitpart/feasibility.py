"""Gripper feasibility (Sec. 3):

    feasible(S) <=> l_3(S) <= G  and  exists antipodal (c1, c2) in S with ||c1 - c2|| <= G

l_3 is the smallest extent of the minimum-volume oriented bounding box. The
antipodal test uses a friction cone of half-angle atan(mu) around each contact's
(sign-free) normal. Units must be metric: pass `scale` (metres per point-cloud unit)
when the cloud is normalized.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def obb_extents(points):
    """Sorted extents (l1 >= l2 >= l3) of the minimum-volume OBB (PCA OBB fallback)."""
    points = np.asarray(points, float)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    try:
        box = pc.get_minimal_oriented_bounding_box()
        return np.sort(np.asarray(box.extent))[::-1]
    except Exception:  # Qhull fails on flat or degenerate sets: PCA box extents
        c = points - points.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        return np.sort(np.ptp(c @ vt.T, axis=0))[::-1]


def box_test(points, G):
    if len(points) < 4:
        return False
    return bool(obb_extents(points)[2] <= G)


def antipodal_test(points, normals, G, mu=0.5, n_samples=300, min_sep=None, seed=0):
    """True if some pair of points forms an antipodal grasp no wider than G."""
    if len(points) < 4:
        return False
    cos_cone = np.cos(np.arctan(mu))
    tree = cKDTree(points)
    if min_sep is None:
        d, _ = tree.query(points[: min(len(points), 500)], k=2)
        min_sep = 3 * np.median(d[:, 1])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), min(n_samples, len(points)), replace=False)
    nrm = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    for i in idx:
        nb = np.asarray(tree.query_ball_point(points[i], G))
        if len(nb) == 0:
            continue
        v = points[nb] - points[i]
        dist = np.linalg.norm(v, axis=1)
        ok = dist > min_sep
        if not ok.any():
            continue
        v, nb, dist = v[ok], nb[ok], dist[ok]
        u = v / dist[:, None]
        c1 = np.abs(u @ nrm[i]) >= cos_cone
        c2 = np.abs(np.einsum("ij,ij->i", u, nrm[nb])) >= cos_cone
        if np.any(c1 & c2):
            return True
    return False


def feasible(points, normals, G, scale=1.0, mu=0.5):
    """Box test first (cheap, necessary), then the antipodal test; parts whose normals are
    unreliable (thin wires, rims) fall back to grasps across locally thin structure."""
    p = np.asarray(points) * scale
    if not box_test(p, G):
        return False
    if antipodal_test(p, normals, G, mu, min_sep=0.001):
        return True
    from .grasping import Gripper, thin_grasps
    g = Gripper(max_opening=G)
    return len(thin_grasps(p, cKDTree(p), p, g, np.random.default_rng(0), max_grasps=1)) > 0
