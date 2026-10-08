"""Language-guided selection (Sec. 4.5).

    r(h, q) = min_phi exp(z_h . t_q) / (exp(z_h . t_q) + exp(z_h . t_phi))

with canonical phrases phi in {"object", "things", "stuff", "texture"} (LERF), and

    h* = argmax_h r(h, q)  s.t.  feasible(h)

with near-ties (within eps_r in log-odds) broken toward the smaller node.
"""
import numpy as np

CANONICAL = ("object", "things", "stuff", "texture")


def relevance(z, t_q, t_canon, temperature=100.0):
    """z (n, D) normalized node embeddings; t_q (D,); t_canon (C, D). Returns (n,).

    Dot products are multiplied by the encoder's logit scale (about 100 for CLIP and
    SigLIP) before the pairwise softmax. The ranking does not depend on it, but the
    tie tolerance eps_r and the floor r_min do. Rows of z that are all zero (no mask
    ever covered the region) get relevance 0.
    """
    sq = temperature * (z @ t_q)
    sc = temperature * (z @ t_canon.T)  # (n, C)
    r = 1.0 / (1.0 + np.exp(np.clip(sc - sq[:, None], -50, 50)))
    r = r.min(1)
    r[np.linalg.norm(z, axis=1) < 1e-6] = 0.0
    return r


def node_embeddings(hier, z_sp, w_sp):
    """Visibility-weighted mean of superpoint embeddings for every node of H."""
    D = z_sp.shape[1]
    n = hier.n_nodes
    acc = np.zeros((n, D))
    wt = np.zeros(n)
    acc[:hier.n_leaves] = z_sp * w_sp[:, None]
    wt[:hier.n_leaves] = w_sp
    for a, b, new, _ in hier.merges:
        acc[new] = acc[a] + acc[b]
        wt[new] = wt[a] + wt[b]
    z = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    return z, wt


def node_membership(hier, n_points_per_sp):
    """(n_nodes, M) boolean membership and per-node point counts."""
    n, M = hier.n_nodes, hier.n_leaves
    H = np.zeros((n, M), bool)
    H[np.arange(M), np.arange(M)] = True
    for a, b, new, _ in hier.merges:
        H[new] = H[a] | H[b]
    return H, H @ n_points_per_sp


def node_embeddings_from_masks(hier, sp, mask_emb, mask_frac, mask_vis, min_frac=0.5, min_iou=0.3,
                               power=2.0):
    """z_h = normalized sum of crop embeddings weighted by IoU(mask, h)^power.

    A mask's 3D footprint is the set of superpoints with >= min_frac of their visible
    pixels inside it; the IoU is measured in points, against the part of h visible in
    that mask's view. Masks with IoU < min_iou are ignored. A node no mask matches
    gets the zero vector (relevance 0).
    """
    s = np.bincount(sp).astype(float)
    H, _ = node_membership(hier, s)
    Hf = H.astype(float)
    m = (mask_frac.astype(float) >= min_frac) * s[None, :]          # (K, M) point-weighted
    v = mask_vis.astype(float) * s[None, :]
    inter = Hf @ m.T                                                 # (n, K)
    hv = Hf @ v.T
    union = m.sum(1)[None, :] + hv - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    w = np.where(iou >= min_iou, iou ** power, 0.0)
    z = w @ mask_emb.astype(float)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def select(r, sizes, feasible_mask=None, eps_r=0.0, r_min=None, min_size=0):
    """Index of the selected node, plus a status string.

    Nodes smaller than min_size points are never selected (single superpoints carry
    embeddings from one or two mask crops and are unreliable).

    status: "ok", "infeasible" (the best match exists but no feasible node reaches r_min),
            or "none" (nothing relevant).
    """
    r = np.asarray(r, float)
    big = np.asarray(sizes) >= min_size
    r = np.where(big, r, -np.inf)
    cand = big.copy() if feasible_mask is None else (np.asarray(feasible_mask, bool) & big)
    if not cand.any():
        return int(np.argmax(r)), "infeasible"
    best = r[cand].max()
    if r_min is not None and best < r_min:
        return int(np.argmax(r)), ("infeasible" if r.max() >= r_min else "none")
    # near-ties are judged on log-odds, since relevance saturates near 0 when the
    # canonical phrases win everywhere (common for untextured renders)
    lo = np.log(np.clip(r, 1e-300, None)) - np.log1p(-np.clip(r, None, 1 - 1e-12))
    near = np.where(cand & (lo >= lo[cand].max() - eps_r))[0]
    return int(near[np.argmin(np.asarray(sizes)[near])]), "ok"


def label_segments(z_seg, part_names, encode_text, background="other"):
    """Open-vocabulary semantic labels for a flat segmentation.

    Each segment gets the part name with the highest relevance, or -1 (background)
    when no part name beats all canonical phrases (relevance <= 0.5).
    Returns labels (n_seg,) indexing part_names, and the relevance matrix.
    """
    t = encode_text([f"the {p}" for p in part_names] + list(CANONICAL))
    tp, tc = t[:len(part_names)], t[len(part_names):]
    R = np.stack([relevance(z_seg, tq, tc) for tq in tp], 1)  # (n_seg, n_parts)
    lab = R.argmax(1)
    lab[R.max(1) <= 0.5] = -1
    return lab, R


def select_multi(r, sizes, members, min_size=0, rel=0.5, max_k=8):
    """All instances of a part: the best node, then further nodes disjoint from those
    already chosen whose score is at least `rel` times the best (e.g. both handles of a
    pot, four chair legs). members: node id -> superpoint ids. Returns node ids."""
    r = np.where(np.asarray(sizes) >= min_size, np.asarray(r, float), -np.inf)
    order = np.argsort(-r)
    if not np.isfinite(r[order[0]]) or r[order[0]] <= 0:
        return [int(order[0])]
    best = r[order[0]]
    chosen, used = [], set()
    for h in order:
        if r[h] < rel * best or len(chosen) >= max_k:
            break
        m = set(members[h].tolist())
        if m & used:
            continue
        chosen.append(int(h))
        used |= m
    return chosen
