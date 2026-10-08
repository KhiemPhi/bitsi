"""Superpoints (Sec. 4.1): over-segmentation by normal-aware geodesic Voronoi cells.

Seeds are chosen by farthest point sampling; every point joins the seed that is
closest along a kNN graph whose edge lengths are inflated where normals disagree,
so cells stop at creases and stay within one part.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree


def farthest_point_sampling(points, m, seed=0):
    n = len(points)
    m = min(m, n)
    rng = np.random.default_rng(seed)
    idx = np.empty(m, dtype=int)
    idx[0] = rng.integers(n)
    d = np.linalg.norm(points - points[idx[0]], axis=1)
    for i in range(1, m):
        idx[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(points - points[idx[i]], axis=1))
    return idx


def knn_graph(points, normals=None, k=10, normal_weight=2.0):
    """Symmetric sparse kNN graph; edge length = distance * (1 + w * (1 - |n_i . n_j|))."""
    n = len(points)
    dist, nbr = cKDTree(points).query(points, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = nbr[:, 1:].ravel()
    w = dist[:, 1:].ravel()
    if normals is not None:
        cos = np.abs(np.einsum("ij,ij->i", normals[rows], normals[cols]))
        w = w * (1 + normal_weight * (1 - cos))
    w = np.maximum(w, 1e-12)
    g = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    return g.maximum(g.T)


def superpoints(points, normals=None, m=None, k=10, normal_weight=2.0, seed=0):
    """Returns labels (N,) in [0, M) and the kNN graph used."""
    n = len(points)
    if m is None:
        m = int(np.clip(n // 8, 50, 500))
    g = knn_graph(points, normals, k, normal_weight)
    # make sure every connected component of the graph receives a seed
    n_cc, cc = connected_components(g, directed=False)
    seeds = list(farthest_point_sampling(points, m, seed))
    have = set(cc[seeds])
    for c in range(n_cc):
        if c not in have:
            seeds.append(int(np.where(cc == c)[0][0]))
    _, _, src = dijkstra(g, directed=False, indices=seeds, min_only=True, return_predecessors=True)
    lut = {s: i for i, s in enumerate(seeds)}
    labels = np.array([lut[s] for s in src])
    # relabel compactly (a seed can lose all of its points to ties)
    _, labels = np.unique(labels, return_inverse=True)
    return labels, g


def superpoint_adjacency(labels, graph):
    """Set of undirected superpoint pairs (i < j) that share a kNN edge."""
    coo = graph.tocoo()
    a, b = labels[coo.row], labels[coo.col]
    mask = a != b
    pairs = np.stack([np.minimum(a[mask], b[mask]), np.maximum(a[mask], b[mask])], 1)
    return np.unique(pairs, axis=0)
