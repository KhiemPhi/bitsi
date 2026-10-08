"""Collision-aware antipodal grasp sampling for a parallel-jaw gripper.

Simulation-free graspability (GPR proxy) and a shared grasp sampler for the
language-guided grasping comparison: every method's selected part is passed to the
same sampler, so differences come from part selection only.

A grasp is (c1, c2, approach): contacts on the part whose connecting line lies in
both friction cones, width <= opening, and an approach direction perpendicular to
the closing axis for which the two fingers and the palm do not intersect the rest
of the object. Gripper dimensions default to a Franka Hand (metres).
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Gripper:
    max_opening: float = 0.08     # G
    finger_length: float = 0.045  # along the approach direction
    finger_width: float = 0.02    # across, perpendicular to closing and approach
    finger_thickness: float = 0.01
    palm_depth: float = 0.02
    clearance: float = 0.003


def _in_box(pts, center, axes, half):
    loc = (pts - center) @ axes.T
    return np.all(np.abs(loc) <= half, axis=1)


def grasp_collides(obj_tree, obj_pts, c1, c2, approach, g, part_mask=None, tol_pts=2):
    """True if fingers or palm intersect object points.

    Fingers sit just outside the contacts along the closing axis; the palm sits
    behind the contact midpoint along -approach, beyond the finger length.
    """
    close = c2 - c1
    w = np.linalg.norm(close)
    x = close / w                                  # closing axis
    z = approach / np.linalg.norm(approach)       # approach (towards the object)
    y = np.cross(z, x)
    axes = np.stack([x, y, z])
    mid = 0.5 * (c1 + c2)
    half_f = np.array([g.finger_thickness / 2, g.finger_width / 2, g.finger_length / 2])
    # finger centres: outside each contact, extending back from the contact depth
    f_off = (w / 2 + g.clearance + g.finger_thickness / 2)
    back = -z * (g.finger_length / 2 - 0.005)
    boxes = [(mid - x * f_off + back, half_f), (mid + x * f_off + back, half_f)]
    palm_half = np.array([w / 2 + g.clearance + g.finger_thickness, g.finger_width / 2, g.palm_depth / 2])
    boxes.append((mid - z * (g.finger_length - 0.005 + g.palm_depth / 2), palm_half))
    r = np.linalg.norm(palm_half) + g.finger_length
    near = np.asarray(obj_tree.query_ball_point(mid, r))
    if len(near) == 0:
        return False
    pts = obj_pts[near]
    hits = 0
    for c, h in boxes:
        hits += _in_box(pts, c, axes, h).sum()
    return hits > tol_pts


def thin_grasps(part_pts, obj_tree, obj_pts, g, rng, n_contacts=100, n_approach=8, max_grasps=20,
                radius=0.015, max_thickness=0.01):
    """Grasps across locally thin structure (wires, rims, plate edges), where estimated
    normals are unreliable: at a sampled point, local PCA over a 1.5 cm ball gives the
    thinnest direction; if the local thickness is below max_thickness, the jaws close
    along it (contacts at the local extremes) and the grasp is kept if collision-free."""
    tree = cKDTree(part_pts)
    out = []
    for i in rng.permutation(len(part_pts))[:n_contacts]:
        nb = tree.query_ball_point(part_pts[i], radius)
        if len(nb) < 6:
            continue
        Q = part_pts[nb] - part_pts[nb].mean(0)
        _, sv, vt = np.linalg.svd(Q, full_matrices=False)
        for axis in (vt[2], vt[1]):  # thinnest direction first, then the in-between one
            proj = Q @ axis
            thick = proj.max() - proj.min()
            if thick > max_thickness or thick > g.max_opening:
                continue
            ctr = part_pts[nb].mean(0)
            c1, c2 = ctr + axis * proj.min(), ctr + axis * proj.max()
            if np.linalg.norm(c2 - c1) < 1e-4:
                c1, c2 = ctr - axis * 5e-4, ctr + axis * 5e-4
            a0 = np.cross(axis, vt[0])
            a0 /= np.linalg.norm(a0) + 1e-12
            a1 = np.cross(axis, a0)
            for th in rng.permutation(n_approach) * (2 * np.pi / n_approach):
                app = np.cos(th) * a0 + np.sin(th) * a1
                if not grasp_collides(obj_tree, obj_pts, c1, c2, app, g):
                    out.append((c1, c2, app))
                    break
            if out and len(out) >= max_grasps:
                return out
    return out


def local_tangent(part_tree, part_pts, p, radius=0.015, min_pts=10):
    """Direction along which the part extends near p (largest local PCA axis), or None."""
    nb = part_tree.query_ball_point(p, radius)
    if len(nb) < min_pts:
        return None
    Q = part_pts[nb] - part_pts[nb].mean(0)
    return np.linalg.svd(Q, full_matrices=False)[2][0]


def sample_grasps(part_pts, part_nrm, obj_pts, g=None, mu=0.5, n_contacts=200, n_approach=8,
                  max_grasps=20, seed=0, min_sep=0.001, thin_fallback=True, across_part=False,
                  max_tangent_cos=0.34, max_mid_gap=0.008):
    """Collision-free antipodal grasps on a part (all arrays in metres).

    min_sep (1 mm) only stops a point pairing with itself; same-surface neighbours already
    fail the friction-cone test. When normal-based contacts yield nothing, grasps across
    locally thin structure are tried (see thin_grasps).
    across_part: grasp a bar-like part (a handle) the way a hand wraps it. The jaws close across
    the bar, not along it (|closing . tangent| <= max_tangent_cos, i.e. >= 70 deg), the jaw
    midpoint must lie on the bar (within max_mid_gap of a part point, so pinching the two ends of
    a C-shaped handle across its opening is rejected), and the approach is perpendicular to the
    tangent too, so the gripper's finger plane is perpendicular to the handle.
    Returns a list of (c1, c2, approach) tuples.
    """
    g = g or Gripper()
    if len(part_pts) < 4:
        return []
    rng = np.random.default_rng(seed)
    cos_cone = np.cos(np.arctan(mu))
    ptree = cKDTree(part_pts)
    otree = cKDTree(obj_pts)
    nrm = part_nrm / (np.linalg.norm(part_nrm, axis=1, keepdims=True) + 1e-12)
    grasps = []
    for i in rng.permutation(len(part_pts))[:n_contacts]:
        nb = np.asarray(ptree.query_ball_point(part_pts[i], g.max_opening))
        if len(nb) == 0:
            continue
        v = part_pts[nb] - part_pts[i]
        dist = np.linalg.norm(v, axis=1)
        ok = dist > min_sep
        if not ok.any():
            continue
        nb, v, dist = nb[ok], v[ok], dist[ok]
        u = v / dist[:, None]
        good = (np.abs(u @ nrm[i]) >= cos_cone) & (np.abs(np.einsum("ij,ij->i", u, nrm[nb])) >= cos_cone)
        if not good.any():
            continue
        j = nb[good][rng.integers(good.sum())]
        c1, c2 = part_pts[i], part_pts[j]
        x = (c2 - c1) / np.linalg.norm(c2 - c1)
        if across_part:
            mid = 0.5 * (c1 + c2)
            if ptree.query(mid)[0] > max_mid_gap:
                continue
            t = local_tangent(ptree, part_pts, mid)
            if t is None or abs(x @ t) > max_tangent_cos:
                continue
            t = t - (t @ x) * x
            t /= np.linalg.norm(t)
            n0 = np.cross(x, t)                   # perpendicular to closing and tangent
            for ph in np.radians([0, 180, 20, -20, 160, 200]):
                app = np.cos(ph) * n0 + np.sin(ph) * t
                if not grasp_collides(otree, obj_pts, c1, c2, app, g):
                    grasps.append((c1, c2, app))
                    break
            if len(grasps) >= max_grasps:
                break
            continue
        # approach directions perpendicular to the closing axis
        a0 = np.cross(x, [0, 0, 1.0])
        if np.linalg.norm(a0) < 1e-3:
            a0 = np.cross(x, [0, 1.0, 0])
        a0 /= np.linalg.norm(a0)
        a1 = np.cross(x, a0)
        for th in rng.permutation(n_approach) * (2 * np.pi / n_approach):
            app = np.cos(th) * a0 + np.sin(th) * a1
            if not grasp_collides(otree, obj_pts, c1, c2, app, g):
                grasps.append((c1, c2, app))
                break
        if len(grasps) >= max_grasps:
            break
    if not grasps and thin_fallback and not across_part:
        grasps = thin_grasps(part_pts, otree, obj_pts, g, rng, max_grasps=max_grasps, n_approach=n_approach)
    return grasps


def graspable(part_pts, part_nrm, obj_pts, g=None, **kw):
    """Part counts as graspable if at least one collision-free antipodal grasp exists."""
    return len(sample_grasps(part_pts, part_nrm, obj_pts, g, max_grasps=1, **kw)) > 0


def grasp_success_rate(part_pts, part_nrm, obj_pts, g=None, n=20, **kw):
    """Fraction of n contact attempts that yield a collision-free antipodal grasp."""
    gs = sample_grasps(part_pts, part_nrm, obj_pts, g, n_contacts=n, max_grasps=n, **kw)
    return len(gs) / n
