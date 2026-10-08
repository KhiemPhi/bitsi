"""Non-learned part detection: geometric definitions of gripper-relevant parts scored on
hierarchy nodes.

Every node gets descriptors from geometry alone (metric box extents, share of the object,
height along the up axis, number of separate attachment patches to the rest of the
object, and how far it protrudes beyond the rest). Each part name has a hand-written
fuzzy rule over those descriptors. No learned features, no images.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .selection import node_membership


def soft_le(x, t, w):
    """~1 when x << t, 0.5 at t, ~0 when x >> t (logistic with width w)."""
    return 1.0 / (1.0 + np.exp((x - t) / w))


def soft_ge(x, t, w):
    return 1.0 - soft_le(x, t, w)


def pca_extents(P):
    c = P - P.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    q = c @ vt.T
    return np.ptp(q, axis=0), vt  # sorted by variance, ~ l1 >= l2 >= l3


def local_cross_section(P, radius=0.04, n_samples=60, rng=None):
    """Median second extent of local PCA in a ball: small for wires and straps (locally 1-D),
    large for sheets and blobs. P in metres."""
    rng = rng or np.random.default_rng(0)
    if len(P) < 8:
        return 0.0
    tree = cKDTree(P)
    vals = []
    for i in rng.choice(len(P), min(n_samples, len(P)), replace=False):
        nb = tree.query_ball_point(P[i], radius)
        if len(nb) < 6:
            continue
        e, _ = pca_extents(P[nb])
        vals.append(e[1])
    return float(np.median(vals)) if vals else 0.0


def node_descriptors(points, sp, pairs, hier, up=(0.0, 1.0, 0.0), scale=1.0, nodes=None,
                     max_points=1500, seed=0):
    """Descriptors for the given nodes (default: all nodes between 0.1% and 50% of points)."""
    rng = np.random.default_rng(seed)
    s = np.bincount(sp).astype(float)
    Hm, sizes = node_membership(hier, s)
    n = len(sp)
    up = np.asarray(up, float) / np.linalg.norm(up)
    hgt = points @ up
    h0, h1 = hgt.min(), hgt.max()
    obj_ext, _ = pca_extents(points[rng.choice(n, min(n, 4000), replace=False)])
    M = sp.max() + 1
    if nodes is None:
        nodes = np.where((sizes >= 0.001 * n) & (sizes <= 0.5 * n))[0]
    sp_pts = [np.where(sp == i)[0] for i in range(M)]
    out = {}
    for h in nodes:
        mem = Hm[h]
        idx = np.concatenate([sp_pts[i] for i in np.where(mem)[0]])
        if len(idx) > max_points:
            idx = rng.choice(idx, max_points, replace=False)
        P = points[idx] * scale
        ext, axes = pca_extents(P)
        # attachment patches: node superpoints adjacent to outside superpoints, grouped by
        # adjacency among themselves
        a_in, b_in = mem[pairs[:, 0]], mem[pairs[:, 1]]
        cross = a_in ^ b_in
        bnd = np.unique(np.where(a_in[cross] , pairs[cross, 0], pairs[cross, 1]))
        if len(bnd):
            lut = -np.ones(M, int)
            lut[bnd] = np.arange(len(bnd))
            both = (lut[pairs[:, 0]] >= 0) & (lut[pairs[:, 1]] >= 0)
            e = pairs[both]
            g = coo_matrix((np.ones(len(e)), (lut[e[:, 0]], lut[e[:, 1]])), shape=(len(bnd), len(bnd)))
            n_patch = connected_components(g, directed=False)[0]
            bnd_frac = s[bnd].sum() / s[mem].sum()
        else:
            n_patch, bnd_frac = 0, 0.0
        nh = (hgt[idx] - h0) / max(h1 - h0, 1e-9)
        out[int(h)] = dict(
            frac=float(sizes[h] / n),
            l1=float(ext[0]), l2=float(ext[1]), l3=float(ext[2]),
            rel_l1=float(ext[0] / (obj_ext[0] * scale)),
            elong=float(ext[0] / max(ext[1], 1e-6)), flat=float(ext[1] / max(ext[2], 1e-6)),
            n_patch=int(n_patch), bnd_frac=float(bnd_frac),
            h_mean=float(nh.mean()), h_min=float(nh.min()), h_max=float(nh.max()),
            cross=local_cross_section(P, rng=rng),
            vertical=float(abs(axes[0] @ up)),
        )
    return out


def _rule(name, d):
    """Fuzzy score in [0, 1] of a node's descriptors for a part name (metres)."""
    small = soft_le(d["frac"], 0.25, 0.05)
    free = soft_le(d["bnd_frac"], 0.5, 0.1)        # touches the rest only at its attachments
    not_sliver = soft_ge(d["frac"], 0.008, 0.002)
    if name in ("handle", "grip"):
        loop = soft_ge(d["n_patch"], 1.5, 0.2)                 # two attachment patches
        thin = soft_le(d["cross"], 0.035, 0.008)               # local cross-section, not the box
        long = soft_ge(d["l1"], 0.06, 0.015)
        return small * not_sliver * free * thin * long * (0.4 + 0.6 * loop)
    if name in ("knob", "button", "switch"):
        tiny = soft_le(d["frac"], 0.03, 0.008)
        compact = soft_le(d["l1"], 0.06, 0.015) * soft_le(d["elong"], 3.0, 0.7)
        single = soft_le(abs(d["n_patch"] - 1), 0.5, 0.15)
        return tiny * compact * single * soft_le(d["bnd_frac"], 0.7, 0.1)
    if name in ("lid", "cap"):
        top = soft_ge(d["h_min"], 0.6, 0.08)
        spans = soft_ge(d["rel_l1"], 0.5, 0.1)
        flatish = soft_ge(d["flat"], 1.5, 0.4)
        return small * not_sliver * top * spans * flatish
    if name == "leg":
        reaches_floor = soft_le(d["h_min"], 0.08, 0.03)
        tall = soft_ge(d["h_max"] - d["h_min"], 0.3, 0.08)
        thin = soft_le(d["l2"], 0.08, 0.02)
        return small * not_sliver * free * reaches_floor * tall * thin * soft_ge(d["elong"], 3.0, 0.7)
    if name == "wheel":
        disc = soft_le(abs(d["l1"] / max(d["l2"], 1e-6) - 1.0), 0.3, 0.1) * soft_ge(d["flat"], 2.0, 0.5)
        low = soft_le(d["h_max"], 0.4, 0.08)
        return small * not_sliver * free * disc * low
    return None  # no geometric definition for this part name


def geometric_scores(name, desc, n_nodes):
    """Per-node score array for a part name, or None if the name has no definition."""
    if name not in DEFINED:
        return None
    r = np.zeros(n_nodes)
    for h, d in desc.items():
        r[h] = _rule(name, d)
    return r


DEFINED = ("handle", "grip", "knob", "button", "switch", "lid", "cap", "leg", "wheel")
