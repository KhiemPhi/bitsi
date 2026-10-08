"""Divisive (ITSI-style) hierarchy with Chebyshev cuts.

Global planar cuts also slice through unrelated regions (a plane at a mug's
handle junction crosses the body), so feeding them into bottom-up merging adds
little. Here each piece is re-swept on its own: the strongest Chebyshev cut of the
piece splits it, each side is broken into connected components, and the pieces
are refined best-first. This is ITSI's recursive slicing with the jump threshold
replaced by inflection-point strength.
"""
import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import connected_components

from .affinity import detect_all_cuts, detect_all_cuts_mc


@dataclass
class DNode:
    idx: np.ndarray                 # point indices
    parent: int = -1
    children: list = field(default_factory=list)
    split_strength: float = 0.0     # strength of the cut that split this node (0 = leaf)
    cut: tuple = None               # (direction, t)


def best_cut_mc(points, colors, idx, gref, channels):
    """Strongest pooled multi-channel cut of a piece (strengths normalized by the root)."""
    if len(idx) < 30:
        return None
    sets = detect_all_cuts_mc(points[idx], None if colors is None else colors[idx], channels=channels, gref=gref)
    best = None
    for d, prof, cuts in sets:
        for c in cuts:
            if best is None or c.strength > best[0]:
                best = (c.strength, d, c.t, prof)
    return best


def best_cut_events(points, normals, colors, idx, channels):
    """Most significant sweep event (z-score) of a piece over candidate directions."""
    from .sweep_events import detect_events
    if len(idx) < 30:
        return None
    ev = detect_events(points[idx], None if normals is None else normals[idx],
                       None if colors is None else colors[idx], channels)
    best = None
    for d, prof, cuts, _ in ev:
        for c in cuts:
            if best is None or c.strength > best[0]:
                best = (c.strength, d, c.t, prof)
    return best


def best_cut(points, idx, ref_slope, detector_kwargs, persist_levels=None):
    """Strongest cut of the piece, strength normalized by the root's steepest slope.

    persist_levels: if set, the split uses every cut on the winning direction that
    already appears at the coarsest `persist_levels` Chebyshev levels (a multi-way,
    threshold-free split), returned as a list of locations in place of a single t.
    """
    if len(idx) < 30:
        return None
    sets = detect_all_cuts(points[idx], norm="none", **detector_kwargs)
    best = None
    for d, prof, cuts in sets:
        for c in cuts:
            # persistence factor x slope in raw BITC units per unit of normalized
            # position, divided by the root's steepest slope so pieces are comparable
            if c.raw_slope > 0:
                s = c.strength / max(c.slope, 1e-12) * c.raw_slope / ref_slope
            else:  # jump-threshold detector: rank by jump size
                s = c.slope
            if best is None or s > best[0]:
                best = (s, d, c.t, prof, cuts)
    if best is None:
        return None
    s, d, t, prof, cuts = best
    if persist_levels:
        ts = sorted({c.t for c in cuts if c.level <= persist_levels} | {t})
        return (s, d, ts, prof)
    return (s, d, t, prof)


def split_by_cut(points, idx, graph, d, t, prof, min_points):
    ts = np.atleast_1d(t)
    slab = np.searchsorted(np.sort(ts), prof["coord"])
    parts = []
    for mask in (slab == k for k in range(len(ts) + 1)):
        sub = idx[mask]
        if len(sub) == 0:
            continue
        n_cc, cc = connected_components(graph[sub][:, sub], directed=False)
        for k in range(n_cc):
            parts.append(sub[cc == k])
    big = [p for p in parts if len(p) >= min_points]
    small = [p for p in parts if len(p) < min_points]
    if len(big) < 2:
        return None
    # attach small fragments to the nearest big part
    for s in small:
        cen = points[s].mean(0)
        j = int(np.argmin([np.linalg.norm(points[b].mean(0) - cen) for b in big]))
        big[j] = np.concatenate([big[j], s])
    return big


def divisive_tree(points, graph, min_frac=0.02, max_leaves=32, detector_kwargs=None, persist_levels=None,
                  colors=None, normals=None):
    """Best-first divisive hierarchy. Returns list of DNode (0 = root) and split order."""
    detector_kwargs = detector_kwargs or {}
    n = len(points)
    min_points = max(int(min_frac * n), 20)
    channels = detector_kwargs.get("channels")
    events = detector_kwargs.get("events")
    if events:
        pass
    elif channels:
        _, gref = detect_all_cuts_mc(points, colors, channels=channels, return_gmax=True)
    else:
        root_sets = detect_all_cuts(points, norm="none", **detector_kwargs)
        ref = max((c.raw_slope for _, _, cs in root_sets for c in cs), default=1.0) or 1.0
    nodes = [DNode(idx=np.arange(n))]
    heap = []

    def push(i):
        if events:
            bc = best_cut_events(points, normals, colors, nodes[i].idx, events)
        elif channels:
            bc = best_cut_mc(points, colors, nodes[i].idx, gref, channels)
        else:
            bc = best_cut(points, nodes[i].idx, ref, detector_kwargs, persist_levels)
        if bc is not None:
            heapq.heappush(heap, (-bc[0], i, bc))

    push(0)
    order = []
    n_leaves = 1
    while heap and n_leaves < max_leaves:
        negs, i, (s, d, t, prof) = heapq.heappop(heap)
        parts = split_by_cut(points, nodes[i].idx, graph, d, t, prof, min_points)
        if parts is None:
            continue
        nodes[i].split_strength = s
        nodes[i].cut = (d, t)
        for p in parts:
            nodes.append(DNode(idx=p, parent=i))
            nodes[i].children.append(len(nodes) - 1)
            push(len(nodes) - 1)
        order.append(i)
        n_leaves += len(parts) - 1
    return nodes, order


def leaves_after(nodes, order, n_splits):
    """Leaf labels after the first n_splits splits (best-first order)."""
    done = set(order[:n_splits])
    leaves = []
    stack = [0]
    while stack:
        i = stack.pop()
        if i in done:
            stack.extend(nodes[i].children)
        else:
            leaves.append(i)
    lab = np.empty(len(nodes[0].idx), dtype=int)
    for k, i in enumerate(leaves):
        lab[nodes[i].idx] = k
    return lab


def flat_k(nodes, order, k):
    """Labels with as close to k leaves as the split order allows (without exceeding k
    unless the first split already does)."""
    best = leaves_after(nodes, order, 0)
    for s in range(1, len(order) + 1):
        lab = leaves_after(nodes, order, s)
        if lab.max() + 1 > k:
            break
        best = lab
    return best


def node_recall(nodes, gt, thresh=0.5):
    g_ids = np.unique(gt)
    best = np.zeros(len(g_ids))
    for nd in nodes:
        lab = gt[nd.idx]
        for j, g in enumerate(g_ids):
            inter = (lab == g).sum()
            iou = inter / (len(nd.idx) + (gt == g).sum() - inter)
            best[j] = max(best[j], iou)
    return float((best >= thresh).mean())
