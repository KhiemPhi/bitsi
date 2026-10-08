"""Point-cloud denoising and regularization before sweep analysis (training-free).

1. statistical outlier removal (points whose mean distance to k neighbours is > std_ratio
   standard deviations above average)
2. surface projection: each point moved onto the plane fit (PCA) to its k nearest neighbours
   (one moving-least-squares-style step), pulling scattered samples onto the surface
3. uniform resampling on a voxel grid, so every slab samples the surface at the same density
Labels (and colours) follow the kept points; voxel resampling keeps the point nearest each
voxel's centroid, so per-point labels stay valid.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def denoise(points, colors=None, labels=None, k_sor=20, std_ratio=2.0, k_proj=16, voxel=None,
            project=True):
    P = np.asarray(points, float)
    keep = np.arange(len(P))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    _, ind = pc.remove_statistical_outlier(nb_neighbors=k_sor, std_ratio=std_ratio)
    keep = keep[np.asarray(ind)]
    Q = P[keep]
    if project and len(Q) > k_proj:
        tree = cKDTree(Q)
        _, nb = tree.query(Q, k=k_proj)
        nbp = Q[nb]                                  # (n, k, 3)
        mu = nbp.mean(1)
        cov = np.einsum("nki,nkj->nij", nbp - mu[:, None], nbp - mu[:, None])
        _, vecs = np.linalg.eigh(cov)
        nrm = vecs[:, :, 0]                          # smallest-variance direction
        Q = Q - np.einsum("ni,ni->n", Q - mu, nrm)[:, None] * nrm
    if voxel is not None and voxel > 0:
        key = np.floor((Q - Q.min(0)) / voxel).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        cen = np.zeros((inv.max() + 1, 3))
        np.add.at(cen, inv, Q)
        cen /= np.bincount(inv)[:, None]
        dist = np.linalg.norm(Q - cen[inv], axis=1)
        order = np.lexsort((dist, inv))
        first = order[np.r_[True, inv[order][1:] != inv[order][:-1]]]
        sel = np.sort(first)
        Q, keep = Q[sel], keep[sel]
    out = [Q, keep]
    if colors is not None:
        out.append(np.asarray(colors)[keep])
    if labels is not None:
        out.append(np.asarray(labels)[keep])
    return tuple(out)
