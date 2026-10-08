"""Language selection with SAM 3 text-prompted masks lifted onto the hierarchy.

For a query q, SAM 3 returns scored instance masks in every rendered view. Each
mask's 3D footprint is the set of superpoints it covers in that view. A node h of
the hierarchy is scored by

    r3(h, q) = mean over views v where h is visible of  max_k  score_k * IoU_v(mask_k, h)

so a node scores high when, in most views that see it, SAM 3 outlines exactly it
for that phrase. This replaces crop-embedding relevance (kept as an ablation).
"""
import numpy as np
import torch

from . import lifting
from .selection import node_membership

_cache = {}


def sam3(device="cuda"):
    if "sam3" not in _cache:
        from transformers import Sam3Model, Sam3Processor
        m = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float16).to(device).eval()
        p = Sam3Processor.from_pretrained("facebook/sam3")
        _cache["sam3"] = (m, p)
    return _cache["sam3"]


@torch.no_grad()
def query_masks(images, queries, threshold=0.3, device="cuda"):
    """For every image and query: list of (mask bool HxW, score). Returns [q][v] lists."""
    from PIL import Image
    m, proc = sam3(device)
    out = [[None] * len(images) for _ in queries]
    for v, img in enumerate(images):
        pil = Image.fromarray(img)
        inp = proc(images=[pil] * len(queries), text=list(queries), return_tensors="pt").to(device)
        inp["pixel_values"] = inp["pixel_values"].half()
        o = m(**inp)
        res = proc.post_process_instance_segmentation(
            o, threshold=threshold, mask_threshold=0.5, target_sizes=inp.get("original_sizes").tolist())
        for qi, r in enumerate(res):
            masks = r["masks"].cpu().numpy().astype(bool) if len(r["masks"]) else np.zeros((0,) + img.shape[:2], bool)
            out[qi][v] = list(zip(masks, r["scores"].float().cpu().numpy().tolist()))
    return out


def lift_query(per_view, sp_pix, M, min_frac=0.5):
    """Per view: (K_v, M) superpoint coverage of each mask, scores, and visible superpoints."""
    lifted = []
    for v, items in enumerate(per_view):
        masks = [mk for mk, _ in items]
        scores = np.array([s for _, s in items])
        frac, cnt = lifting.mask_membership(masks, sp_pix[v], M)
        lifted.append(dict(cover=(frac >= min_frac).astype(np.uint8), scores=scores.astype(np.float32),
                           vis=cnt > 0))
    return lifted


def node_scores(hier, sp, lifted):
    """r3(h, q) for every node, from one query's lifted masks."""
    s = np.bincount(sp).astype(float)
    H, _ = node_membership(hier, s)
    Hf = H.astype(float)
    total = np.zeros(hier.n_nodes)
    nvis = np.zeros(hier.n_nodes)
    for lv in lifted:
        vis_w = lv["vis"] * s
        hv = Hf @ vis_w                     # visible points of each node in this view
        seen = hv > 0
        nvis += seen
        if len(lv["scores"]) == 0:
            continue
        m = lv["cover"].astype(float) * s[None, :]   # (K, M)
        inter = Hf @ m.T                             # (n, K)
        union = m.sum(1)[None, :] + hv[:, None] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        total += np.where(seen, (iou * lv["scores"][None, :]).max(1), 0.0)
    return total / np.maximum(nvis, 1)


def superpoint_scores(sp, lifted):
    """No-hierarchy baseline: per-superpoint score = mean over views where it is visible
    of the best score of any SAM 3 mask covering it."""
    M = sp.max() + 1
    tot = np.zeros(M)
    nv = np.zeros(M)
    for lv in lifted:
        nv += lv["vis"]
        if len(lv["scores"]):
            tot += (lv["cover"] * lv["scores"][:, None]).max(0)
    return tot / np.maximum(nv, 1)


def union_score(H, sp, lifted):
    """SAM 3 soft-union node score (current best)."""
    s = np.bincount(sp).astype(float)
    Hf = node_membership(H, s)[0].astype(float)
    tot = Hf @ s
    num = np.zeros(H.n_nodes)
    den = np.zeros(H.n_nodes)
    for lv in lifted:
        hv = Hf @ (lv["vis"] * s)
        w = hv / np.maximum(tot, 1)
        if len(lv["scores"]):
            u = (lv["cover"].astype(float) * lv["scores"][:, None]).max(0)
            inter = Hf @ (u * s)
            union = (u * s).sum() + hv - inter
            num += w * np.where(union > 0, inter / np.maximum(union, 1e-12), 0)
        den += w
    return num / np.maximum(den, 1e-9)


def refine_region(region, sp_score, sp_seen, pairs, high=0.5, low=0.05, min_seen=2, rings=2):
    """Superpoint-level refinement of a selected region (bool over superpoints).

    Drops members SAM 3 saw in >= min_seen views but scored below `low`, then grows into
    adjacent superpoints scored >= `high`, up to `rings` steps.
    """
    r = region.copy()
    r &= ~((sp_score < low) & (sp_seen >= min_seen))
    if not r.any():
        r = region.copy()
    for _ in range(rings):
        a, b = pairs[:, 0], pairs[:, 1]
        frontier = np.zeros_like(r)
        frontier[b[r[a] & ~r[b]]] = True
        frontier[a[r[b] & ~r[a]]] = True
        add = frontier & (sp_score >= high)
        if not add.any():
            break
        r |= add
    return r


def superpoint_seen(sp, lifted):
    """Number of views in which each superpoint is visible."""
    return np.sum([lv["vis"] for lv in lifted], axis=0)
