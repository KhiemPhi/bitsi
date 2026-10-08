"""Topology-informed slicing: the Reeb graph of a sweep.

Along a sweep direction, each slab's cross-section splits into pieces (connected components of its
occupancy grid). Pieces of neighbouring slabs that overlap are linked. The resulting slice graph,
with chains of one-in/one-out pieces compressed into arcs, is the (discrete) Reeb graph of the
height function along the sweep:

  * nodes  = topological events: a piece appears, vanishes, splits or merges
  * arcs   = groups of slices with a fixed, simple topology -> candidate parts
  * cycles = loops (split -> parallel tracks -> merge), e.g. a mug handle

Segmentation is per piece, not per slab: between the split and the merge of a loop, the handle's
pieces form their own arc while the body continues, so a cut never slices through the body.
"""
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label

from .profiles import median_nn_spacing, pca_axes, sweep_frame

ST8 = np.ones((3, 3), dtype=int)


def slice_pieces(points, d, cell_h=6.0, spacing_h=1.5, window=5, min_frac=0.05, min_pts=5, solid=True):
    """Pieces per slab along d. Returns (piece_of_point, pieces) where pieces is a list of dicts
    {slab, cells (set of (u, v)), idx (point indices)} and piece_of_point is -1 for points that
    fell in specks. Topology of slab s uses the points of slabs s-window//2..s+window//2; each point
    is assigned to a piece of its own slab."""
    P = np.asarray(points, float)
    h = median_nn_spacing(P)
    R = sweep_frame(P, d)
    q = (P - P.mean(0)) @ R.T
    lo = q.min(0)
    spacing = spacing_h * h
    cell = cell_h * h
    S = max(int(np.ceil((q[:, 0].max() - lo[0]) / spacing)), 1)
    slab = np.minimum(((q[:, 0] - lo[0]) / spacing).astype(int), S - 1)
    gu = ((q[:, 1] - lo[1]) / cell).astype(int)
    gv = ((q[:, 2] - lo[2]) / cell).astype(int)
    shape = (gu.max() + 1, gv.max() + 1)
    order = np.argsort(slab, kind="stable")
    bounds = np.searchsorted(slab[order], np.arange(S + 1))
    half = window // 2
    piece_of = -np.ones(len(P), int)
    pieces = []
    for s in range(S):
        own = order[bounds[s]:bounds[s + 1]]
        if len(own) == 0:
            continue
        win = order[bounds[max(0, s - half)]:bounds[min(S, s + half + 1)]]
        g = np.zeros(shape, dtype=bool)
        g[gu[win], gv[win]] = True
        if solid:  # hollow surface samples: close small gaps, fill enclosed holes -> solid section
            g = binary_fill_holes(binary_closing(np.pad(g, 1), ST8))[1:-1, 1:-1] | g
        lab, n = label(g, ST8)
        comp_own = lab[gu[own], gv[own]]
        thr = max(min_pts, int(min_frac * len(win)))
        sizes = np.bincount(lab[gu[win], gv[win]], minlength=n + 1)
        for k in range(1, n + 1):
            if sizes[k] < thr:
                continue
            mk = comp_own == k
            if not mk.any():
                continue
            cells = lab == k
            pid = len(pieces)
            pieces.append(dict(slab=s, mask=cells, idx=own[mk]))
            piece_of[own[mk]] = pid
    return piece_of, pieces, slab


def slice_pieces_knn(points, d, spacing_h=1.5, window=3, radius_h=2.5, min_frac=0.03, min_pts=8):
    """Pieces per slab from the cloud's own connectivity: points of the window (slabs
    s - window//2 .. s + window//2) linked when closer than radius_h * spacing; each connected
    component is a piece; each point is assigned to the piece of its own slab. No grid, no
    morphology, so a 1 cm gap between a handle and a body stays open. Pieces of neighbouring slabs
    are linked when their windows share points (see reeb_graph_shared)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    P = np.asarray(points, float)
    h = median_nn_spacing(P)
    d = np.asarray(d, float) / np.linalg.norm(d)
    s_coord = P @ d
    lo = s_coord.min()
    spacing = spacing_h * h
    S = max(int(np.ceil((s_coord.max() - lo) / spacing)), 1)
    slab = np.minimum(((s_coord - lo) / spacing).astype(int), S - 1)
    order = np.argsort(slab, kind="stable")
    bounds = np.searchsorted(slab[order], np.arange(S + 1))
    half = window // 2
    piece_of = -np.ones(len(P), int)
    pieces = []
    for s in range(S):
        win = order[bounds[max(0, s - half)]:bounds[min(S, s + half + 1)]]
        if len(win) == 0 or bounds[s] == bounds[s + 1]:
            continue
        pairs = cKDTree(P[win]).query_pairs(radius_h * h, output_type="ndarray")
        n = len(win)
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
        nc, lab = connected_components(g, directed=False)
        thr = max(min_pts, int(min_frac * n))
        own = slab[win] == s
        for k in range(nc):
            mk = lab == k
            if mk.sum() < thr or not (mk & own).any():
                continue
            pid = len(pieces)
            pieces.append(dict(slab=s, win=set(win[mk].tolist()), idx=win[mk & own], mask=None))
            piece_of[win[mk & own]] = pid
    return piece_of, pieces, slab


def reeb_graph_shared(pieces):
    """Edges between pieces of neighbouring slabs whose windows share points."""
    by_slab = {}
    for i, p in enumerate(pieces):
        by_slab.setdefault(p["slab"], []).append(i)
    edges = []
    for s, ids in by_slab.items():
        for i in ids:
            for j in by_slab.get(s + 1, []):
                if pieces[i]["win"] & pieces[j]["win"]:
                    edges.append((i, j))
    return edges


def reeb_graph(pieces, dilate=False):
    """Edges between pieces of neighbouring slabs whose cross-sections overlap (optionally dilated)."""
    by_slab = {}
    for i, p in enumerate(pieces):
        by_slab.setdefault(p["slab"], []).append(i)
    edges = []
    for s, ids in by_slab.items():
        nxt = by_slab.get(s + 1, [])
        for i in ids:
            mi = binary_dilation(pieces[i]["mask"], ST8) if dilate else pieces[i]["mask"]
            for j in nxt:
                if (mi & pieces[j]["mask"]).any():
                    edges.append((i, j))
    return edges


def arcs_and_loops(pieces, edges):
    """Compress the slice graph into Reeb arcs; return (arc_of_piece, n_arcs, loops) where loops is
    a list of arc-id lists forming independent cycles of the arc graph."""
    import networkx as nx
    n = len(pieces)
    up = [[] for _ in range(n)]
    down = [[] for _ in range(n)]
    for i, j in edges:
        up[i].append(j)
        down[j].append(i)
    arc = -np.ones(n, int)
    na = 0
    order = sorted(range(n), key=lambda i: pieces[i]["slab"])
    for i in order:
        if arc[i] >= 0:
            continue
        # continue the arc of the single predecessor if it has a single successor (regular point)
        if len(down[i]) == 1 and len(up[down[i][0]]) == 1 and arc[down[i][0]] >= 0:
            arc[i] = arc[down[i][0]]
        else:
            arc[i] = na
            na += 1
    G = nx.MultiGraph()
    G.add_nodes_from(range(na))
    for i, j in edges:
        if arc[i] != arc[j]:
            G.add_edge(arc[i], arc[j])
    # loops: independent cycles of the arc graph (parallel edges count as a 2-cycle)
    loops = [c for c in nx.cycle_basis(nx.Graph(G))]
    for u, v in set((min(a, b), max(a, b)) for a, b in G.edges()):
        if G.number_of_edges(u, v) > 1:
            loops.append([u, v])
    return arc, na, loops, G


def choose_scale(points, d, cells=(2.5, 3.5, 5.0, 6.0), windows=(3, 5)):
    """Label-free choice of the cross-section grid: the finest (cell, window) whose slabs are not
    fragmented (median pieces per occupied slab <= 2); falls back to the coarsest."""
    for c in cells:
        for w in windows:
            piece_of, pieces, slab = slice_pieces(points, d, cell_h=c, window=w)
            per = np.bincount([p["slab"] for p in pieces])
            if len(per) and np.median(per[per > 0]) <= 2:
                return dict(cell_h=c, window=w)
    return dict(cell_h=cells[-1], window=windows[-1])


def prune_short_arcs(arc, pieces, G, min_span):
    """Persistence along the sweep: arcs spanning fewer than min_span slabs are flicker; merge each
    into its most connected neighbour (repeatedly). Returns (arc relabelled, contracted MultiGraph)."""
    import networkx as nx
    span = {}
    for i, p in enumerate(pieces):
        a = arc[i]
        lo, hi = span.get(a, (p["slab"], p["slab"]))
        span[a] = (min(lo, p["slab"]), max(hi, p["slab"]))
    G = nx.MultiGraph(G)
    changed = True
    while changed:
        changed = False
        for a in sorted(G.nodes, key=lambda a: span[a][1] - span[a][0]):
            if span[a][1] - span[a][0] + 1 >= min_span or G.degree(a) == 0:
                continue
            nb = list(G.neighbors(a))
            b = max(nb, key=lambda b: (G.number_of_edges(a, b), span[b][1] - span[b][0]))
            for u, v in list(G.edges(a)):
                w = v if u == a else u
                if w != b:
                    G.add_edge(b, w)
            G.remove_node(a)
            arc[arc == a] = b
            span[b] = (min(span[a][0], span[b][0]), max(span[a][1], span[b][1]))
            changed = True
            break
    # drop self-loops created by contraction
    G.remove_edges_from([(u, v) for u, v in G.edges() if u == v])
    return arc, G


def reeb_segmentation(points, d=None, min_part_frac=0.02, auto_scale=False, min_span_frac=0.10, mode="grid",
                      normals=None, **kw):
    """Per-point part labels from Reeb arcs along d (default: the PCA direction whose sweep shows
    the most loops, ties -> longest axis). Small arcs are merged into the neighbouring arc they
    share the most adjacencies with. Returns (labels, info)."""
    P = np.asarray(points, float)
    if d is None and normals is not None:
        from .axes import normal_tensor_axis
        d = normal_tensor_axis(normals)[0]
    dirs = [d] if d is not None else list(pca_axes(P))
    best = None
    for k, dd in enumerate(dirs):
        if mode == "knn":
            piece_of, pieces, slab = slice_pieces_knn(P, dd, **kw)
            edges = reeb_graph_shared(pieces)
        else:
            kk = dict(kw, **choose_scale(P, dd)) if auto_scale else kw
            piece_of, pieces, slab = slice_pieces(P, dd, **kk)
            edges = reeb_graph(pieces)
        arc, na, loops, G = arcs_and_loops(pieces, edges)
        S = max(p["slab"] for p in pieces) + 1 if pieces else 1
        arc, G = prune_short_arcs(arc, pieces, G, max(3, int(min_span_frac * S)))
        import networkx as nx
        loops = [c for c in nx.cycle_basis(nx.Graph(G))]
        for u, v in set((min(a, b), max(a, b)) for a, b in G.edges()):
            if G.number_of_edges(u, v) > 1:
                loops.append([u, v])
        na = int(arc.max()) + 1 if len(arc) else 0
        score = (len(loops), -k)
        if best is None or score > best[0]:
            best = (score, dd, piece_of, pieces, arc, na, loops, G)
    _, dd, piece_of, pieces, arc, na, loops, G = best
    lab = np.where(piece_of >= 0, arc[np.maximum(piece_of, 0)], -1)
    # unassigned (speck) points join the nearest assigned point
    if (lab < 0).any() and (lab >= 0).any():
        from scipy.spatial import cKDTree
        ok = lab >= 0
        lab[~ok] = lab[ok][cKDTree(P[ok]).query(P[~ok])[1]]
    # merge small arcs into their most connected neighbour
    sizes = np.bincount(lab, minlength=max(na, lab.max() + 1))
    parent = list(range(na))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a in np.argsort(sizes):
        if sizes[a] >= min_part_frac * len(P) or sizes[a] == 0:
            continue
        nb = [b for b in G.neighbors(a)] if (a in G and len(G)) else []
        if not nb:
            continue
        b = max(nb, key=lambda b: G.number_of_edges(a, b) * 1e6 + sizes[find(b)])
        parent[find(a)] = find(b)
    lab = np.array([find(a) for a in lab])
    _, lab = np.unique(lab, return_inverse=True)
    loop_arcs = sorted({a for c in loops for a in c})
    return lab, dict(direction=dd, n_arcs=na, loops=loops, loop_arcs=loop_arcs, arc_of_point=np.where(
        piece_of >= 0, arc[np.maximum(piece_of, 0)], -1))
