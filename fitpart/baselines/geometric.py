"""Training-free geometric decomposition baselines that give per-point labels.

coacd_segment     CoACD approximate convex decomposition (SIGGRAPH 2022) of a Poisson surface
                  reconstructed from the cloud; each point goes to the hull it is deepest inside.
superquadric_segment  hierarchical EMS multi-superquadric recovery (Liu et al., CVPR 2022; the
                  authors' Python `hierarchical_ems`, ext/EMS-superquadric_fitting); each point goes
                  to the superquadric it is closest to (radial distance).

Both run in the `pointcloud` env (numba, coacd). Settings are the authors' defaults, with the
DBSCAN radius and minimum cluster size scaled to the object (the authors set them per input).
"""
import os, sys, types
import numpy as np

EMS_SRC = os.path.expanduser("~/bitsi/ext/EMS-superquadric_fitting/Python/src")


def _poisson_mesh(P, N, depth=8):
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(N)
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
    dens = np.asarray(dens)
    m.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))
    m = m.crop(pc.get_axis_aligned_bounding_box().scale(1.05, pc.get_center()))
    return np.asarray(m.vertices), np.asarray(m.triangles)


def coacd_segment(P, N, threshold=0.05, max_hulls=-1):
    import coacd
    from scipy.spatial import ConvexHull
    c, s = P.mean(0), np.linalg.norm(P.max(0) - P.min(0))
    V, F = _poisson_mesh((P - c) / s, N)
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(V, F), threshold=threshold, max_convex_hull=max_hulls)
    Q = (P - c) / s
    depth = np.full((len(P), len(parts)), -np.inf)
    for k, (v, _) in enumerate(parts):
        eq = ConvexHull(np.asarray(v)).equations           # n.x + d <= 0 inside
        depth[:, k] = -(Q @ eq[:, :3].T + eq[:, 3]).max(1)  # > 0 inside, < 0 outside
    return depth.argmax(1)


def _ems():
    if "mayavi" not in sys.modules:  # EMS.utilities imports mayavi only for plotting
        mv = types.ModuleType("mayavi")
        mv.mlab = None
        sys.modules["mayavi"] = mv
        sys.modules["mayavi.mlab"] = mv
    if EMS_SRC not in sys.path:
        sys.path.insert(0, EMS_SRC)
    from EMS.EMS_recovery import EMS_recovery
    return EMS_recovery


def superquadric_segment(P, N=None, max_layer=5, eps_frac=0.05, min_frac=0.01):
    """Hierarchical EMS as in the authors' multiquadric_test.py, in a unit-diagonal frame."""
    from sklearn.cluster import DBSCAN
    EMS_recovery = _ems()
    c, s = P.mean(0), np.linalg.norm(P.max(0) - P.min(0))
    Q = (P - c) / s
    eps, min_pts = eps_frac, max(10, int(min_frac * len(P)))
    idx_seg = {0: [np.arange(len(Q))]}
    quadrics = []
    for h in range(max_layer):
        idx_seg[h + 1] = []
        for ids in idx_seg[h]:
            try:
                x, p = EMS_recovery(Q[ids], 0.9, 20, 1e-3, 2e-1, 2, 0.3, 2, True, False)
            except ValueError:  # degenerate cluster (EMS: "x0 is infeasible"); leave it to the others
                continue
            quadrics.append(x)
            out = ids[p < 0.1]
            if np.sum(p) < 0.8 * len(ids) and len(out) >= min_pts:
                lab = DBSCAN(eps=eps, min_samples=min_pts).fit(Q[out]).labels_
                idx_seg[h + 1] += [out[lab == k] for k in range(lab.max() + 1)]
    # assign every point to the superquadric with the smallest radial distance
    D = np.stack([_sq_radial_distance(x, Q) for x in quadrics], 1)
    return D.argmin(1)


def _sq_radial_distance(x, Q):
    """|1 - F^(-eps1/2)| * |X| for superquadric x (EMS `superquadric` object), X in its frame."""
    shape, scale, R, t = np.asarray(x.shape, float), np.asarray(x.scale, float), np.asarray(x.RotM), np.asarray(x.translation)
    X = (Q - t) @ R
    e1, e2 = np.maximum(shape, 1e-2)
    a = np.abs(X) / scale + 1e-12
    F = ((a[:, 0] ** (2 / e2) + a[:, 1] ** (2 / e2)) ** (e2 / e1) + a[:, 2] ** (2 / e1))
    return np.abs(1 - F ** (-e1 / 2)) * np.linalg.norm(X, axis=1)
