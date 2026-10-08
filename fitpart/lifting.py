"""Lifting 2D foundation models onto superpoints (Sec. 4.2).

  * SAM automatic masks per view -> mask co-occurrence affinity A_mask
  * DINOv2 patch features        -> per-superpoint features -> A_feat (cosine)
  * vision-language mask crops    -> per-superpoint embeddings z_i (selection only)

Models are loaded lazily and cached at module level so a batch run pays once.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

SAM_CKPT = "/home/khiem/Downloads/Automated-Data-Collection-and-Annotation-fall24/seg_anything_with_annotation_code/sam_vit_h_4b8939.pth"
VL_MODELS = {
    "siglip": "google/siglip-so400m-patch14-384",
    "clip": "openai/clip-vit-large-patch14",
}
_cache = {}

try:  # transformers 5 prints a progress bar per weight while loading
    from transformers.utils import logging as _hf_logging
    _hf_logging.disable_progress_bar()
    _hf_logging.set_verbosity_error()
except Exception:
    pass


# ---------------------------------------------------------------- models
def sam_generator(points_per_side=32, device="cuda"):
    key = ("sam", points_per_side)
    if key not in _cache:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to(device).eval()
        # fp16 image encoder: ~25% faster, masks unchanged (mean best IoU 0.9999 vs fp32)
        enc_forward = sam.image_encoder.forward

        def fp16_forward(x):
            with torch.autocast("cuda", dtype=torch.float16):
                return enc_forward(x).float()

        sam.image_encoder.forward = fp16_forward
        _cache[key] = SamAutomaticMaskGenerator(
            sam, points_per_side=points_per_side, pred_iou_thresh=0.86,
            stability_score_thresh=0.90, min_mask_region_area=0)
    return _cache[key]


def dino(device="cuda"):
    if "dino" not in _cache:
        from transformers import AutoModel
        m = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval().half()
        _cache["dino"] = m
    return _cache["dino"]


import os
TEXT_DEVICE = os.environ.get("FITPART_TEXT_DEVICE", "cuda")


def vl_model(name="siglip", device="cuda"):
    key = ("vl", name, device)
    if key not in _cache:
        from transformers import AutoModel, AutoProcessor
        repo = VL_MODELS[name]
        dtype = torch.float16 if device == "cuda" else torch.float32
        m = AutoModel.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
        p = AutoProcessor.from_pretrained(repo)
        _cache[key] = (m, p)
    return _cache[key]


def _features(out):
    """transformers >= 5 can return a model-output object from get_*_features."""
    if isinstance(out, torch.Tensor):
        return out
    for k in ("pooler_output", "image_embeds", "text_embeds"):
        if getattr(out, k, None) is not None:
            return getattr(out, k)
    return out[0]


_text_cache = {}


def precompute_text(phrases, name="siglip", device=None, batch=64):
    """Encode phrases once (e.g. in a parent process before forking workers)."""
    phrases = [p for p in phrases if (name, p) not in _text_cache]
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    for i in range(0, len(phrases), batch):
        emb = _encode_text(phrases[i:i + batch], name, device or TEXT_DEVICE)
        for p, e in zip(phrases[i:i + batch], emb):
            _text_cache[(name, p)] = e


def encode_text(phrases, name="siglip", device=None):
    if all((name, p) in _text_cache for p in phrases):
        return np.stack([_text_cache[(name, p)] for p in phrases])
    return _encode_text(list(phrases), name, device or TEXT_DEVICE)


@torch.no_grad()
def _encode_text(phrases, name="siglip", device="cuda"):
    m, p = vl_model(name, device)
    kw = dict(padding="max_length", max_length=64) if name == "siglip" else dict(padding=True)
    inp = p(text=list(phrases), return_tensors="pt", truncation=True, **kw).to(device)
    t = _features(m.get_text_features(**inp)).float()
    return F.normalize(t, dim=-1).cpu().numpy()


@torch.no_grad()
def encode_images(images, name="siglip", device="cuda", batch=32):
    m, p = vl_model(name, device)
    outs = []
    for i in range(0, len(images), batch):
        inp = p(images=images[i:i + batch], return_tensors="pt").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(m.dtype)
        outs.append(_features(m.get_image_features(**inp)).float())
    if not outs:
        return np.zeros((0, 1))
    return F.normalize(torch.cat(outs), dim=-1).cpu().numpy()


# ---------------------------------------------------------------- per-view lifting
def run_sam(images, points_per_side=32):
    gen = sam_generator(points_per_side)
    out = []
    for img in images:
        masks = gen.generate(img)
        masks = sorted(masks, key=lambda m: m["area"])  # finest first
        out.append([m["segmentation"] for m in masks])
    return out


def sp_pixel_index(pix2pt, sp_labels):
    """Superpoint id per pixel (-1 = background)."""
    sp_pix = np.where(pix2pt >= 0, sp_labels[np.maximum(pix2pt, 0)], -1)
    return sp_pix


def mask_membership(masks, sp_pix, M, min_frac=0.5):
    """(n_masks, M) fraction of each superpoint's visible pixels inside each mask,
    and per-superpoint visible pixel count in this view."""
    flat = sp_pix.reshape(-1)
    vis = flat >= 0
    cnt = np.bincount(flat[vis], minlength=M).astype(float)
    frac = np.zeros((len(masks), M))
    for k, m in enumerate(masks):
        sel = m.reshape(-1) & vis
        frac[k] = np.bincount(flat[sel], minlength=M) / np.maximum(cnt, 1)
    return frac, cnt


def finest_mask_labels(frac, cnt, min_frac=0.5):
    """m_v(i): the finest (smallest) mask holding at least min_frac of superpoint i's
    visible pixels, -1 if none or not visible. Masks are sorted finest first."""
    M = frac.shape[1]
    lab = np.full(M, -1)
    for i in range(M):
        if cnt[i] == 0:
            continue
        hits = np.where(frac[:, i] >= min_frac)[0]
        if len(hits):
            lab[i] = hits[0]
    return lab


def mask_affinity(per_view_frac, per_view_cnt, mode="jaccard", min_frac=0.5):
    """A_mask over superpoints from all views.

    mode="finest": the draft's definition, agreement of the finest-mask labels over
                   views where both superpoints are visible.
    mode="jaccard": multi-granularity version, over all masks of all views,
                   |masks containing both| / |masks containing either| (counted only
                   in views where both are visible).
    Returns (M, M) affinity (NaN where never co-visible) and co-visibility counts.
    """
    M = per_view_frac[0].shape[1]
    num = np.zeros((M, M))
    den = np.zeros((M, M))
    covis = np.zeros((M, M))
    for frac, cnt in zip(per_view_frac, per_view_cnt):
        vis = (cnt > 0).astype(float)
        both_vis = np.outer(vis, vis)
        covis += both_vis
        if mode == "finest":
            lab = finest_mask_labels(frac, cnt, min_frac)
            ok = lab >= 0
            same = (lab[:, None] == lab[None, :]) & ok[:, None] & ok[None, :]
            num += same * both_vis
            den += both_vis * (ok[:, None] & ok[None, :])
        else:
            inm = (frac >= min_frac).astype(float)  # (n_masks, M)
            both = inm.T @ inm
            either = inm.sum(0)[:, None] + inm.sum(0)[None, :] - both
            num += both * both_vis
            den += either * both_vis
    aff = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    np.fill_diagonal(aff, 1.0)
    return aff, covis


@torch.no_grad()
def dino_superpoint_features(images, sp_pix, M, device="cuda"):
    """Mean DINOv2 patch feature over each superpoint's visible pixels in all views."""
    model = dino(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    H, W = images.shape[1:3]
    side = 518  # 37 x 37 patches of 14 px
    feats = torch.zeros((M, model.config.hidden_size), device=device)
    cnt = torch.zeros(M, device=device)
    for img, spp in zip(images, sp_pix):
        x = torch.as_tensor(img, device=device).permute(2, 0, 1)[None].float() / 255
        x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)
        x = ((x - mean) / std).half()
        tok = model(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS
        g = side // 14
        fm = tok.reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
        fm = F.interpolate(fm, size=(H, W), mode="bilinear", align_corners=False)[0]  # (C,H,W)
        lab = torch.as_tensor(spp, device=device).reshape(-1)
        ok = lab >= 0
        f = fm.reshape(fm.shape[0], -1).T[ok]
        feats.index_add_(0, lab[ok], f)
        cnt.index_add_(0, lab[ok], torch.ones_like(lab[ok], dtype=torch.float))
    feats = feats / cnt.clamp(min=1)[:, None]
    return feats.cpu().numpy(), cnt.cpu().numpy()


def crop_mask(img, mask, margin=0.15, blurred=None, min_side=32):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    my, mx = int(h * margin) + 4, int(w * margin) + 4
    H, W = mask.shape
    y0, y1, x0, x1 = max(0, y0 - my), min(H, y1 + my), max(0, x0 - mx), min(W, x1 + mx)
    if blurred is None:
        blurred = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(6)))
    comp = np.where(mask[..., None], img, blurred)[y0:y1, x0:x1]
    if min(comp.shape[:2]) < min_side:
        return None
    return Image.fromarray(comp)


def vl_superpoint_embeddings(images, masks_per_view, per_view_frac, M, name="siglip",
                             max_masks_per_view=60):
    """z_i: normalized average of mask-crop embeddings covering superpoint i, weighted by
    the fraction of i's visible pixels inside each mask (OpenMask3D-style)."""
    crops, owners = [], []
    for v, (img, masks) in enumerate(zip(images, masks_per_view)):
        blurred = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(6)))
        for k, m in enumerate(masks[-max_masks_per_view:]):  # keep the largest masks if too many
            kk = len(masks) - min(len(masks), max_masks_per_view) + k
            c = crop_mask(img, m, blurred=blurred)
            if c is not None:
                crops.append(c)
                owners.append((v, kk))
    emb = encode_images(crops, name)
    D = emb.shape[1]
    z = np.zeros((M, D))
    wsum = np.zeros(M)
    for e, (v, k) in zip(emb, owners):
        w = per_view_frac[v][k]
        z += w[:, None] * e[None, :]
        wsum += w
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return z, wsum, emb, owners
