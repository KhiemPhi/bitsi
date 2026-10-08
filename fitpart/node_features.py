"""Part-agnostic features of hierarchy nodes for the learned (non-deep) part detector.

Three groups, all computed from the point cloud alone:

  geometry   metric extents of the node's PCA box, share of the object, elongation and
             flatness, local cross-section (thin wire vs sheet), height along the up axis,
             verticality, fraction of the node on its boundary with the rest
  topology   connected components of the node (beta_0), number of separate attachment
             patches to the rest, H0/H1 persistent homology of the node and of the node
             plus its one-ring of neighbouring superpoints (a handle closes a loop with
             the body: long-lived H1 in the collar but not in the node alone)
  chebyshev  the node's own BITC profile along its main axis, fit with the multi-scale
             Chebyshev detector: number of persistent cuts, strongest cut, chopped degree
  context    size relative to its parent and the linkage at which it was formed
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .cheb import multiscale_cuts
from .geometric_detect import local_cross_section, pca_extents
from .profiles import bitc_profile
from .selection import node_membership

FEATURE_NAMES = [
    # geometry
    "frac", "l1", "l2", "l3", "rel_l1", "elong", "flat", "cross", "h_mean", "h_min", "h_max",
    "vertical", "bnd_frac",
    # topology
    "beta0", "n_patch", "h1_node_max", "h1_node_2nd", "h1_collar_max", "h1_collar_gain", "h0_gap",
    # chebyshev
    "cheb_ncuts", "cheb_ncuts_persist", "cheb_max_strength", "cheb_degree",
    # context
    "frac_of_parent", "linkage",
]


def _persistence(P, maxdim=1):
    from ripser import ripser
    d = ripser(P, maxdim=maxdim)["dgms"]
    h0 = d[0][np.isfinite(d[0][:, 1])]
    h0_life = np.sort(h0[:, 1] - h0[:, 0])[::-1] if len(h0) else np.zeros(1)
    h1_life = np.sort(d[1][:, 1] - d[1][:, 0])[::-1] if len(d) > 1 and len(d[1]) else np.zeros(1)
    return h0_life, h1_life


def _subsample(idx, k, rng):
    return idx if len(idx) <= k else rng.choice(idx, k, replace=False)


def node_features(points, sp, pairs, hier, scale, up=(0.0, 1.0, 0.0), min_frac=0.001, max_frac=0.5,
                  n_topo=120, seed=0):
    """Feature matrix (n_candidates, len(FEATURE_NAMES)) and the candidate node ids."""
    rng = np.random.default_rng(seed)
    s = np.bincount(sp).astype(float)
    Hm, sizes = node_membership(hier, s)
    n, M = len(sp), sp.max() + 1
    up = np.asarray(up, float) / np.linalg.norm(up)
    hgt = points @ up
    h0, h1 = hgt.min(), hgt.max()
    obj_ext, _ = pca_extents(points[rng.choice(n, min(n, 4000), replace=False)])
    sp_pts = [np.where(sp == i)[0] for i in range(M)]
    parent = {}
    for a, b, new, _ in hier.merges:
        parent[a] = new
        parent[b] = new
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(M, M)).tocsr()
    adj = adj.maximum(adj.T)
    cand = np.where((sizes >= min_frac * n) & (sizes <= max_frac * n))[0]
    X = np.zeros((len(cand), len(FEATURE_NAMES)))
    for row, h in enumerate(cand):
        mem = Hm[h]
        ids = np.where(mem)[0]
        idx = np.concatenate([sp_pts[i] for i in ids])
        P = points[_subsample(idx, 1500, rng)] * scale
        ext, axes = pca_extents(P)
        # boundary / attachment patches on the superpoint graph
        a_in, b_in = mem[pairs[:, 0]], mem[pairs[:, 1]]
        cross_e = a_in ^ b_in
        bnd = np.unique(np.where(a_in[cross_e], pairs[cross_e, 0], pairs[cross_e, 1]))
        collar = np.unique(np.where(a_in[cross_e], pairs[cross_e, 1], pairs[cross_e, 0]))
        if len(bnd):
            sub = adj[bnd][:, bnd]
            n_patch = connected_components(sub, directed=False)[0]
            bnd_frac = s[bnd].sum() / s[ids].sum()
        else:
            n_patch, bnd_frac = 0, 0.0
        beta0 = connected_components(adj[ids][:, ids], directed=False)[0]
        # persistent homology (metric units), node alone and node + one-ring collar
        Pn = points[_subsample(idx, n_topo, rng)] * scale
        h0n, h1n = _persistence(Pn)
        if len(collar):
            cidx = np.concatenate([sp_pts[i] for i in collar])
            Pc = points[np.concatenate([_subsample(idx, n_topo // 2, rng), _subsample(cidx, n_topo // 2, rng)])] * scale
            _, h1c = _persistence(Pc)
        else:
            h1c = np.zeros(1)
        L1 = max(ext[0], 1e-6)
        # the node's own BITC profile + multi-scale Chebyshev cuts along its main axis
        try:
            prof = bitc_profile(P, axes[0])
            cuts, info = multiscale_cuts(prof["t"], prof["values"], max_deg=16)
            ncut, npers = len(cuts), sum(c.level <= 2 for c in cuts)
            smax = max((c.strength for c in cuts), default=0.0)
            deg = info.get("n_top", 0) if info else 0
        except Exception:
            ncut = npers = deg = 0
            smax = 0.0
        nh = (hgt[idx] - h0) / max(h1 - h0, 1e-9)
        par = parent.get(h)
        X[row] = [
            sizes[h] / n, ext[0], ext[1], ext[2], ext[0] / (obj_ext[0] * scale),
            ext[0] / max(ext[1], 1e-6), ext[1] / max(ext[2], 1e-6),
            local_cross_section(P, rng=rng), nh.mean(), nh.min(), nh.max(), abs(axes[0] @ up), bnd_frac,
            beta0, n_patch, h1n[0] / L1, (h1n[1] if len(h1n) > 1 else 0.0) / L1, h1c[0] / L1,
            (h1c[0] - h1n[0]) / L1, (h0n[1] if len(h0n) > 1 else 0.0) / L1,
            ncut, npers, smax, deg,
            sizes[h] / sizes[par] if par is not None else 1.0,
            hier.linkage.get(int(h), np.inf) if np.isfinite(hier.linkage.get(int(h), np.inf)) else 1.0,
        ]
    return X, cand
