"""Stage 2 (CPU + text encoder): assemble every ablation from cached lifts and score it.

Metrics per object (Sec. 5.1):
  hierarchy recall   GT parts (instances if available, else semantic parts) matched by
                     some node of H with IoU >= 0.5
  oracle-K mIoU      class-agnostic, tree cut at the GT part count (no threshold)
  theta mIoU         class-agnostic, tree cut at linkage theta (default segmentation)
  semantic mIoU      text queries (part names): theta-cut segments labeled by relevance;
                     per-shape part IoU (ShapeNet convention: absent in both = 1)
  selection acc.     per part name present: selected node IoU with the GT part >= 0.5

usage: python eval_cache.py CACHE_DIR OUT_DIR [theta] [part_names_json]
For ShapeNetPart no part names are used (semantic metrics skipped).
"""
import glob
import json
import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart import lifting
from fitpart.metrics import class_agnostic_miou, hierarchy_recall
from fitpart.pipeline import Config, FitPart
from fitpart.feasibility import feasible
from fitpart.grasping import graspable
from fitpart.scales import GRASPABLE_PARTS, metric_scale
from fitpart.selection import CANONICAL, node_membership, relevance, select, select_multi

THETAS = (0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.96)

ABLATIONS = {
    # name: assemble kwargs
    "full": dict(),
    "geo:none": dict(geometry="none"),
    "geo:jump": dict(geometry="divisive_jump"),
    "geo:single": dict(geometry="divisive_single"),
    "geo:extrema": dict(geometry="divisive_extrema"),
    "geo:axis": dict(geometry="divisive_axis"),
    "geo:global": dict(geometry="global"),
    "sem only (lam=1)": dict(lam=1.0),
    "geo only (lam=0)": dict(lam=0.0),
    "mask finest": dict(mask_mode="finest"),
    "no long-range": dict(long_range_tau=None),
    "combine: product": dict(combine="product"),
    "combine: min": dict(combine="min"),
    "product, geo:none": dict(combine="product", geometry="none"),
    "sem: PartField": dict(sem_source="partfield"),
    "sem: PartField, geo:none": dict(sem_source="partfield", geometry="none"),
    "sem: lifted+PartField": dict(sem_source="both"),
    "sem: lifted+PartField, geo:none": dict(sem_source="both", geometry="none"),
    "sem: lifted+PartField, geo:jump": dict(sem_source="both", geometry="divisive_jump"),
}


class PartIoU(list):
    """Per-shape part IoUs (ShapeNet convention: absent in both = 1) that also carry the
    raw intersection/union counts, for category-accumulated IoU (PartSLIP-style)."""
    iu = None

    def tolist(self):
        return {"iou": list(self), "iu": self.iu}


def per_shape_part_iou(gt, pred, n_parts):
    ious, iu = [], []
    for p in range(n_parts):
        g, q = gt == p, pred == p
        i, u = int((g & q).sum()), int((g | q).sum())
        ious.append(1.0 if u == 0 else i / u)
        iu.append((i, u))
    out = PartIoU(ious)
    out.iu = iu
    return out


def text_bank(part_names, vl, category=None):
    """Part prompts, LERF canonical prompts, and a background prompt (the whole object)."""
    bg = [f"a {category.lower()}"] if category else ["an object"]
    phrases = [f"the {p}" for p in part_names] + list(CANONICAL) + bg
    t = lifting.encode_text(phrases, vl)
    n = len(part_names)
    return t[:n], t[n:n + len(CANONICAL)], t[-1]


def evaluate(L, fp, kw, theta, part_names, tbank, relevance_mode="lerf", with_grasp=False):
    res = fp.assemble(L, **kw)
    sp, H = res.sp, res.hier
    sem = L["sem"]
    ins = L.get("ins")
    # class-agnostic metrics: every GT instance (or part) is a region, and the unlabeled
    # remainder (-1, e.g. a bottle's body in PartNet-E) is one more region
    gt_parts = (ins if ins is not None else sem).copy()
    gt_parts[sem < 0] = gt_parts.max() + 1
    labeled = np.ones(len(sem), bool)
    out = {}
    out["recall"] = hierarchy_recall(H, sp[labeled], gt_parts[labeled])[0] if labeled.any() else np.nan
    K = len(np.unique(gt_parts[labeled]))
    sizes = np.bincount(sp)
    lab_k, _ = H.flat(n_clusters=K, sizes=sizes, min_frac=0.02)
    out["miou_oracleK"] = class_agnostic_miou(gt_parts[labeled], lab_k[sp][labeled])[0]
    for th in THETAS:
        lab_s, _ = H.flat(theta=th, sizes=sizes, min_frac=0.02)
        out[f"miou@{th}"] = class_agnostic_miou(gt_parts[labeled], lab_s[sp][labeled])[0]
        out[f"nseg@{th}"] = int(lab_s.max() + 1)
    lab_t, nodes_t = H.flat(theta=theta, sizes=sizes, min_frac=0.02)
    out["miou_theta"] = class_agnostic_miou(gt_parts[labeled], lab_t[sp][labeled])[0]
    out["n_seg_theta"] = int(lab_t.max() + 1)
    if part_names is None or res.z_nodes is None:
        return out
    tp, tc, tbg = tbank
    # semantic segmentation from theta-cut segments: each segment takes the part name it
    # matches best, or background (-1) when the whole-object prompt matches better
    seg_z = np.stack([res.z_nodes[n] if n >= 0 else _mean_z(res, H.members[n]) for n in nodes_t])
    S = seg_z @ np.vstack([tp, tbg[None]]).T
    seg_lab = S.argmax(1)
    seg_lab[seg_lab == len(tp)] = -1
    pred = seg_lab[lab_t[sp]]
    out["sem_iou"] = per_shape_part_iou(sem, pred, len(part_names)).tolist()
    # selection
    node_sizes = np.zeros(H.n_nodes)
    node_sizes[:H.n_leaves] = sizes
    for a, b, new, _ in H.merges:
        node_sizes[new] = node_sizes[a] + node_sizes[b]
    # selection engines: SigLIP crop relevance (LERF or raw cosine), and SAM 3 masks when cached
    scorers = {"": (lambda p: relevance(res.z_nodes, tp[p], tc)) if relevance_mode == "lerf"
               else (lambda p: res.z_nodes @ tp[p])}
    if "sam3" in L:
        from fitpart.sam3_select import node_scores, union_score
        s3 = {}

        def sam3_score(p):
            if p not in s3:
                fn = union_score if SAM3_SCORE == "union" else node_scores
                s3[p] = fn(H, sp, L["sam3"][part_names[p]])
            return s3[p]
        scorers["_sam3"] = sam3_score
    min_size = fp.cfg.min_select_frac * len(sp)
    if "sam3" in L:
        # baseline: SAM 3 masks lifted straight to superpoints, no hierarchy (score >= 0.5)
        from fitpart.sam3_select import superpoint_scores
        S = np.stack([superpoint_scores(sp, L["sam3"][n]) for n in part_names], 1)  # (M, P)
        for th, tag in ((0.5, ""), (0.3, "@0.3")):
            lab_sp = np.where(S.max(1) >= th, S.argmax(1), -1)
            out["sem_iou_sam3direct" + tag] = per_shape_part_iou(sem, lab_sp[sp], len(part_names)).tolist()
            sel = []
            for p in np.unique(sem[sem >= 0]):
                m = (S[:, p] >= th)[sp]
                g = sem == p
                sel.append(float((m & g).sum() / max((m | g).sum(), 1)))
            out["sel_iou_sam3direct" + tag] = sel
        if with_grasp and "points" in L:
            out["grasp_sam3direct"] = region_grasp_eval(L, part_names, lambda p: (S[:, p] >= 0.5)[sp])
    if TASKS.get(L["cat"]):
        out["task"] = task_eval(L, res, part_names, tc, node_sizes, fp, min_size)
    for sfx, score in scorers.items():
        # semantic segmentation by selection: each part name selects a node; overlaps go
        # to the more relevant name; everything unselected is background
        best_r = np.full(len(sp), -np.inf)
        pred_sel = np.full(len(sp), -1)
        for p in range(len(part_names)):
            r = score(p)
            h, _ = select(r, node_sizes, None, fp.cfg.eps_r, min_size=min_size)
            m = np.isin(sp, H.members[h]) & (r[h] > best_r)
            pred_sel[m] = p
            best_r[m] = r[h]
        out["sem_iou_sel" + sfx] = per_shape_part_iou(sem, pred_sel, len(part_names)).tolist()
        if sfx == "_sam3":  # multi-instance selection (all handles, all legs)
            best_r = np.full(len(sp), -np.inf)
            pred_m = np.full(len(sp), -1)
            for p in range(len(part_names)):
                r = score(p)
                for h in select_multi(r, node_sizes, H.members, min_size=min_size):
                    m = np.isin(sp, H.members[h]) & (r[h] > best_r)
                    pred_m[m] = p
                    best_r[m] = r[h]
            out["sem_iou_selmulti_sam3"] = per_shape_part_iou(sem, pred_m, len(part_names)).tolist()
            # superpoint-level refinement of each part's selected nodes
            from fitpart.sam3_select import refine_region, superpoint_scores, superpoint_seen
            M = sp.max() + 1
            best_r = np.full(M, -np.inf)
            lab_sp = np.full(M, -1)
            for p in range(len(part_names)):
                r = score(p)
                hs = select_multi(r, node_sizes, H.members, min_size=min_size)
                region = np.zeros(M, bool)
                for h in hs:
                    region[H.members[h]] = True
                lv = L["sam3"][part_names[p]]
                region = refine_region(region, superpoint_scores(sp, lv), superpoint_seen(sp, lv), L["pairs"])
                val = max(r[h] for h in hs)
                m = region & (val > best_r)
                lab_sp[m] = p
                best_r[m] = val
            out["sem_iou_selrefine_sam3"] = per_shape_part_iou(sem, lab_sp[sp], len(part_names)).tolist()
        sel = []
        for p in np.unique(sem[sem >= 0]):
            h, _ = select(score(p), node_sizes, None, fp.cfg.eps_r, min_size=min_size)
            m = np.isin(sp, H.members[h])
            g = sem == p
            sel.append(float((m & g).sum() / (m | g).sum()))
        out["sel_iou" + sfx] = sel
        if with_grasp and "points" in L:
            gr = grasp_eval(L, res, part_names, score, node_sizes, fp)
            if gr:
                out["grasp" + sfx] = gr["grasp"]
    return out


SAM3_SUFFIX = os.environ.get("FITPART_SAM3_SUFFIX", ".sam3.pkl")
EXTRA_GEO = [g for g in os.environ.get("FITPART_EXTRA_GEO", "").split(",") if g]
# restrict which ablations run (semicolon-separated names); empty = all
ONLY_AB = [a for a in os.environ.get("FITPART_ONLY_ABLATIONS", "").split(";") if a]
ONLY_GEO = ["divisive", "divisive_jump"] + EXTRA_GEO if ONLY_AB else []
SAM3_SCORE = os.environ.get("FITPART_SAM3_SCORE", "union")  # union (soft union of masks) | max (original)

TASKS = {}
if os.environ.get("FITPART_TASKS"):
    TASKS = {c: v for c, v in json.load(open(os.environ["FITPART_TASKS"])).items() if not c.startswith("_")}


def task_eval(L, res, part_names, tc, node_sizes, fp, min_size):
    """Task instructions: raw instruction vs (oracle-)rewritten part phrase, per engine.
    Returns a list of {mode: IoU with the GT target part}."""
    sp, H, sem = res.sp, res.hier, L["sem"]
    vl = fp.cfg.vl_model
    recs = []
    for t in TASKS[L["cat"]]:
        if t["part"] not in part_names:
            continue
        p = part_names.index(t["part"])
        g = sem == p
        if not g.any():
            continue
        rec = {"task": t["task"], "part": t["part"]}
        tq = lifting.encode_text([t["task"], f"the {t['part']}"], vl)
        cand = {"siglip_raw": relevance(res.z_nodes, tq[0], tc),
                "siglip_rewrite": relevance(res.z_nodes, tq[1], tc)}
        if "sam3" in L:
            from fitpart.sam3_select import node_scores, union_score
            if t["task"] in L["sam3"]:
                cand["sam3_raw"] = union_score(H, sp, L["sam3"][t["task"]])
            cand["sam3_rewrite"] = union_score(H, sp, L["sam3"][t["part"]])
        for k, r in cand.items():
            h, _ = select(r, node_sizes, None, fp.cfg.eps_r, min_size=min_size)
            m = np.isin(sp, H.members[h])
            rec[k] = float((m & g).sum() / (m | g).sum())
        recs.append(rec)
    return recs


def region_grasp_eval(L, part_names, region_of):
    """Grasp records for methods that return a point mask per part name (no hierarchy)."""
    P = L["points"].astype(float)
    N = L["normals"].astype(float)
    scale = metric_scale(P, L["cat"])
    if scale is None:
        return []
    Pm, sem, recs = P * scale, L["sem"], []
    for p in np.unique(sem[sem >= 0]):
        if part_names[p] not in GRASPABLE_PARTS:
            continue
        m, g = region_of(p), sem == p
        iou = float((m & g).sum() / max((m | g).sum(), 1))
        ok = graspable(Pm[m], N[m], Pm) if m.sum() >= 4 else False
        rec = dict(iou=iou, graspable=bool(ok), status="ok")
        recs.append({"part": part_names[p], "relevance": rec, "feasible": rec})
    return recs


def grasp_eval(L, res, part_names, score, node_sizes, fp, top=40):
    """GPR on graspable GT parts (Sec. 5.1), with and without the feasibility constraint.

    For each graspable part name present in the object: select a node for "the <name>"
    (a) by relevance alone and (b) among feasible nodes only. Record whether the
    selection is correct (IoU >= 0.5 with the GT part) and whether a collision-free
    antipodal grasp exists on it (object rescaled to metric with the category prior).
    """
    P = L["points"].astype(float)
    N = L["normals"].astype(float)
    scale = metric_scale(P, L["cat"])
    if scale is None:
        return {}
    Pm = P * scale
    sp, H, sem = res.sp, res.hier, L["sem"]
    out = {"grasp": []}
    feas_cache = {}
    for p in np.unique(sem[sem >= 0]):
        if part_names[p] not in GRASPABLE_PARTS:
            continue
        r = score(p)
        g = sem == p
        rec = {"part": part_names[p]}
        for mode in ("relevance", "feasible"):
            fmask = None
            if mode == "feasible":
                cand = np.argsort(-r)[:top]
                for h in cand:
                    if h not in feas_cache:
                        m = np.isin(sp, H.members[h])
                        feas_cache[h] = feasible(Pm[m], N[m], fp.cfg.gripper_width)
                fmask = np.zeros(len(r), bool)
                fmask[[h for h in cand if feas_cache[h]]] = True
            h, status = select(r, node_sizes, fmask, fp.cfg.eps_r, min_size=fp.cfg.min_select_frac * len(sp))
            m = np.isin(sp, H.members[h])
            iou = float((m & g).sum() / (m | g).sum())
            ok = graspable(Pm[m], N[m], Pm) if m.sum() >= 4 else False
            rec[mode] = dict(iou=iou, graspable=bool(ok), status=status)
        out["grasp"].append(rec)
    return out


def partfield_eval(L, fp, part_names):
    """Baseline: PartField features -> (a) KMeans with oracle K (PartField's protocol) and
    (b) an average-linkage hierarchy on our superpoints (cosine of mean features), which
    then goes through the same SAM 3 selection as FitPart."""
    from sklearn.cluster import KMeans
    from fitpart.hierarchy import agglomerate
    from fitpart.sam3_select import node_scores, union_score
    sp, pairs, sem = L["sp"], L["pairs"], L["sem"]
    ins = L.get("ins")
    gt_parts = (ins if ins is not None else sem).copy()
    gt_parts[sem < 0] = gt_parts.max() + 1
    F = L["pf_feat"].astype(np.float32)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
    K = len(np.unique(gt_parts))
    out = {"miou_kmeans": class_agnostic_miou(gt_parts, KMeans(K, n_init=4, random_state=0).fit_predict(F))[0]}
    M = sp.max() + 1
    Fs = np.zeros((M, F.shape[1]))
    np.add.at(Fs, sp, F)
    Fs /= np.linalg.norm(Fs, axis=1, keepdims=True) + 1e-8
    w = np.clip((Fs[pairs[:, 0]] * Fs[pairs[:, 1]]).sum(1), 0, 1)
    H = agglomerate(M, pairs, w)
    sizes = np.bincount(sp)
    out["recall"] = hierarchy_recall(H, sp, gt_parts)[0]
    lab_k, _ = H.flat(n_clusters=K, sizes=sizes, min_frac=0.02)
    out["miou_oracleK"] = class_agnostic_miou(gt_parts, lab_k[sp])[0]
    if part_names is None or "sam3" not in L:
        return out
    node_sizes = np.zeros(H.n_nodes)
    node_sizes[:H.n_leaves] = sizes
    for a, b, new, _ in H.merges:
        node_sizes[new] = node_sizes[a] + node_sizes[b]
    min_size = fp.cfg.min_select_frac * len(sp)
    best_r = np.full(len(sp), -np.inf)
    pred = np.full(len(sp), -1)
    scores = {}
    for p, n in enumerate(part_names):
        r = scores[p] = union_score(H, sp, L["sam3"][n])
        h, _ = select(r, node_sizes, None, fp.cfg.eps_r, min_size=min_size)
        m = np.isin(sp, H.members[h]) & (r[h] > best_r)
        pred[m] = p
        best_r[m] = r[h]
    out["sem_iou_sel_sam3"] = per_shape_part_iou(sem, pred, len(part_names)).tolist()
    best_r = np.full(len(sp), -np.inf)
    pred_m = np.full(len(sp), -1)
    for p in range(len(part_names)):
        for h in select_multi(scores[p], node_sizes, H.members, min_size=min_size):
            m = np.isin(sp, H.members[h]) & (scores[p][h] > best_r)
            pred_m[m] = p
            best_r[m] = scores[p][h]
    out["sem_iou_selmulti_sam3"] = per_shape_part_iou(sem, pred_m, len(part_names)).tolist()
    sel = []
    for p in np.unique(sem[sem >= 0]):
        h, _ = select(scores[p], node_sizes, None, fp.cfg.eps_r, min_size=min_size)
        m = np.isin(sp, H.members[h])
        g = sem == p
        sel.append(float((m & g).sum() / (m | g).sum()))
    out["sel_iou_sam3"] = sel
    if "points" in L:
        from types import SimpleNamespace
        gr = grasp_eval(L, SimpleNamespace(sp=sp, hier=H), part_names, lambda p: scores[p], node_sizes, fp)
        if gr:
            out["grasp_sam3"] = gr["grasp"]
    return out


def recompute_geometry(L, fp):
    """Recompute every cached A_geo variant with the current geometry code (cheap), so
    caches made before a geometry fix stay usable. Superpoints are kept as cached."""
    from fitpart.affinity import detect_all_cuts, geometric_affinity
    from fitpart.data import load_shapenetpart, shapenetpart_categories
    from fitpart.pipeline import divisive_pair_affinity
    from fitpart.superpoints import knn_graph
    if "points" not in L:  # early ShapeNetPart caches: reload the object
        sid = {v: k for k, v in shapenetpart_categories().items()}[L["cat"]]
        P, N, _ = load_shapenetpart(sid, L["oid"])
        L["points"], L["normals"] = P.astype(np.float32), N.astype(np.float32)
    P, N = L["points"].astype(float), L["normals"].astype(float)
    g = knn_graph(P, N)
    colors = None
    for name in EXTRA_GEO:
        L["A_geo"].setdefault(name, None)
    if EXTRA_GEO:
        from fitpart.data import PARTNETE_ROOT, load_partnete
        d = os.path.join(PARTNETE_ROOT, "data", "test", L["cat"], L["oid"])
        if os.path.isdir(d):
            colors = load_partnete(d)[2]
    for name in list(L["A_geo"]):
        if name == "none":
            continue
        if ONLY_GEO and name not in ONLY_GEO:
            continue
        kind, det = fp.GEOMETRY_VARIANTS[name]
        if kind == "divisive":
            L["A_geo"][name], _ = divisive_pair_affinity(P, g, L["sp"], L["pairs"], det, colors=colors, normals=N)
        else:
            L["A_geo"][name] = geometric_affinity(P, L["sp"], L["pairs"], detect_all_cuts(P, **det))


def _mean_z(res, members):
    z = (res.z_sp[members] * res.w_sp[members, None]).sum(0)
    return z / (np.linalg.norm(z) + 1e-12)


_G = {}


def _init_worker(g, text_cache):
    import torch
    torch.set_num_threads(1)
    lifting._text_cache.update(text_cache)
    _G.update(g)
    _G["fp"] = FitPart(g["cfg"])


def _eval_file(f):
    """All ablations and baselines for one cached object; failures are logged and skipped."""
    import traceback
    try:
        return _eval_file_inner(f)
    except Exception:
        print("FAILED", f, traceback.format_exc(), flush=True)
        return os.path.basename(f).split("__")[0], {}


def _eval_file_inner(f):
    fp, meta, theta, vl = _G["fp"], _G["meta"], _G["theta"], _G["vl"]
    L = pickle.load(open(f, "rb"))
    if os.path.exists(f[:-4] + SAM3_SUFFIX):
        L["sam3"] = pickle.load(open(f[:-4] + SAM3_SUFFIX, "rb"))
    if os.path.exists(f[:-4] + ".partfield.npy"):
        L["pf_feat"] = np.load(f[:-4] + ".partfield.npy")
    if os.environ.get("FITPART_REGEO", "1") == "1":
        recompute_geometry(L, fp)
    names = meta.get(L["cat"]) if meta else None
    bank = text_bank(names, vl, L["cat"]) if names is not None else None
    out = {}
    abl = dict(ABLATIONS)
    for gname in EXTRA_GEO:
        abl[f"geo:{gname}"] = dict(geometry=gname)
        abl[f"sem: PartField, geo:{gname}"] = dict(sem_source="partfield", geometry=gname)
    if ONLY_AB:
        abl = {k: v for k, v in abl.items() if k in ONLY_AB or any(k.endswith("geo:" + g) for g in EXTRA_GEO)}
    for ab, kw in abl.items():
        if "A_feat" not in L and ab not in ("geo:none", "geo:jump", "geo:single", "geo:extrema",
                                            "geo:axis", "geo:global", "full"):
            continue
        out[ab] = evaluate(L, fp, kw, theta, names, bank,
                           with_grasp=ab in ("full", "geo:none", "geo:jump", "sem: PartField",
                                             "sem: PartField, geo:none") or any(g in ab for g in EXTRA_GEO))
    if names is not None and not ONLY_AB:
        out["raw cosine"] = evaluate(L, fp, {}, theta, names, bank, relevance_mode="cosine")
    if "pf_feat" in L:
        out["baseline: PartField"] = partfield_eval(L, fp, names)
    return L["cat"], out


def main():
    cache, out_dir = sys.argv[1], sys.argv[2]
    theta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    names_json = sys.argv[4] if len(sys.argv) > 4 else None
    vl = os.environ.get("FITPART_VL", "siglip")
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(names_json)) if names_json else None
    fp = FitPart(Config(vl_model=vl))
    files = sorted(f for f in glob.glob(os.path.join(cache, "*.pkl")) if ".sam3" not in os.path.basename(f))
    # every phrase any object will need, encoded once here; forked workers only look them up
    if meta:
        cats = {os.path.basename(f).split("__")[0] for f in files}
        phrases = set(CANONICAL)
        for c in cats:
            phrases |= {f"the {p}" for p in meta.get(c, [])} | {f"a {c.lower()}"}
            for t in TASKS.get(c, []):
                phrases |= {t["task"], f"the {t['part']}"}
        lifting.precompute_text(sorted(phrases), vl)
    workers = int(os.environ.get("FITPART_WORKERS", "12"))
    from multiprocessing import get_context
    # spawn, not fork: a forked child that touches torch after the parent has started
    # torch's CPU thread pool (text encoding) deadlocks
    init = (dict(cfg=fp.cfg, meta=meta, theta=theta, vl=vl), dict(lifting._text_cache))
    with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=init) as pool:
        results = pool.map(_eval_file, files, chunksize=1)
    rows = defaultdict(lambda: defaultdict(list))  # ablation -> category -> list
    for cat, per_ab in results:
        for ab, r in per_ab.items():
            rows[ab][cat].append(r)
    pickle.dump({k: dict(v) for k, v in rows.items()}, open(os.path.join(out_dir, "rows.pkl"), "wb"))

    def cat_mean(rs, key):
        if key.startswith("acc:"):  # category-accumulated part IoU: sum I / sum U per part
            k = key[4:]
            iu = [r[k]["iu"] for r in rs if k in r]
            if not iu:
                return np.nan
            I = np.sum([[x[0] for x in a] for a in iu], 0)
            U = np.sum([[x[1] for x in a] for a in iu], 0)
            return float(np.mean(I[U > 0] / U[U > 0])) if (U > 0).any() else np.nan
        if key.startswith("sem_iou"):
            return np.mean([np.mean(r[key]["iou"]) for r in rs if key in r]) if any(key in r for r in rs) else np.nan
        if key.startswith("sel_acc"):
            k2 = "sel_iou" + key[len("sel_acc"):]
            v = [x >= 0.5 for r in rs if k2 in r for x in r[k2]]
            return np.mean(v) if v else np.nan
        v = [r[key] for r in rs if key in r]
        return np.nanmean(v) if v else np.nan

    def grasp_stats(by_cat, mode, sfx=""):
        recs = [g[mode] for rs in by_cat.values() for r in rs for g in r.get("grasp" + sfx, [])]
        if not recs:
            return None
        corr = np.array([x["iou"] >= 0.5 for x in recs])
        gr = np.array([x["graspable"] for x in recs])
        return (len(recs), corr.mean(), (gr[corr].mean() if corr.any() else np.nan), (corr & gr).mean(),
                gr.mean(), np.mean([x["iou"] for x in recs]))

    keys = ["recall", "miou_oracleK", "miou_kmeans", "miou_theta", "n_seg_theta", "sem_iou", "sem_iou_sel", "sel_acc",
            "sem_iou_sel_sam3", "sem_iou_selmulti_sam3", "sel_acc_sam3", "sem_iou_sam3direct",
            "sel_acc_sam3direct", "sem_iou_sam3direct@0.3", "sel_acc_sam3direct@0.3",
            "acc:sem_iou_sel", "acc:sem_iou_selmulti_sam3", "acc:sem_iou_selrefine_sam3", "acc:sem_iou_sam3direct", "acc:sem_iou_sam3direct@0.3"]
    sweep = []
    for ab in ("full", "geo:none", "geo:jump", "sem only (lam=1)", "sem: PartField", "sem: PartField, geo:none",
               "sem: lifted+PartField", "sem: lifted+PartField, geo:none"):
        if ab in rows:
            sweep.append(f"| {ab} | " + " | ".join(
                f"{np.nanmean([cat_mean(rs, f'miou@{th}') for rs in rows[ab].values()]):.3f} "
                f"({np.nanmean([cat_mean(rs, f'nseg@{th}') for rs in rows[ab].values()]):.1f})" for th in THETAS) + " |")
    lines = ["| Variant | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for ab, by_cat in rows.items():
        vals = [np.nanmean([cat_mean(rs, k) for rs in by_cat.values()]) for k in keys]
        lines.append(f"| {ab} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    n_obj = sum(len(v) for v in rows["full"].values())
    table = f"{n_obj} objects, {len(rows['full'])} categories, theta={theta} (class-mean)\n\n" + "\n".join(lines)
    if meta:
        cats = sorted(rows["full"])
        table += ("\n\nPer-category (full):\n\n| Category | n | sem mIoU (segments) | sem mIoU (SigLIP sel.) "
                  "| sel acc (SigLIP) | sem mIoU (SAM 3 sel.) | sel acc (SAM 3) |\n|---|---|---|---|---|---|---|\n")
        for c in cats:
            rs = rows["full"][c]
            table += (f"| {c} | {len(rs)} | {cat_mean(rs, 'sem_iou'):.3f} | {cat_mean(rs, 'sem_iou_sel'):.3f} "
                      f"| {cat_mean(rs, 'sel_acc'):.3f} | {cat_mean(rs, 'sem_iou_sel_sam3'):.3f} "
                      f"| {cat_mean(rs, 'sel_acc_sam3'):.3f} |\n")
    table += ("\n\nClass-agnostic mIoU (mean #segments) vs merge threshold theta:\n\n| Variant | "
              + " | ".join(str(t) for t in THETAS) + " |\n|" + "---|" * (len(THETAS) + 1) + "\n" + "\n".join(sweep))
    tl = []
    trecs = [t for rs in rows["full"].values() for r in rs for t in r.get("task", [])]
    if trecs:
        for k in ("siglip_raw", "siglip_rewrite", "sam3_raw", "sam3_rewrite"):
            v = [t[k] for t in trecs if k in t]
            if v:
                tl.append(f"| {k} | {len(v)} | {np.mean(np.array(v) >= 0.5):.3f} | {np.mean(v):.3f} |")
        table += ("\n\nTask instructions (full; 'rewrite' = oracle part phrase standing in for LLM rewriting):"
                  "\n\n| Query | n | acc (IoU>=0.5) | mean IoU |\n|---|---|---|---|\n" + "\n".join(tl))
    gl = []
    for ab in ("full", "geo:none", "geo:jump", "baseline: PartField", "sem: PartField",
               "sem: PartField, geo:none", "sem: lifted+PartField", "sem: lifted+PartField, geo:none",
               "sem: lifted+PartField, geo:jump"):
        if ab not in rows:
            continue
        for sfx, eng in (("", "SigLIP"), ("_sam3", "SAM 3"), ("_sam3direct", "SAM 3 lifted, no hierarchy")):
            for mode in ("relevance", "feasible"):
                st = grasp_stats(rows[ab], mode, sfx)
                if st:
                    gl.append(f"| {ab} | {eng} | {mode} | {st[0]} | {st[5]:.3f} | {st[1]:.3f} | {st[4]:.3f} | "
                              f"{st[2]:.3f} | {st[3]:.3f} |")
    if gl:
        table += ("\n\nGraspable-part queries (handles, knobs, shafts, necks, grips; metric scale from "
                  "category priors, G = 8 cm):\n\n| Variant | Engine | Selection | n queries | mean IoU | correct (IoU>=0.5) | "
                  "graspable (any selection) | GPR given correct | correct and graspable |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(gl))
    print(table)
    open(os.path.join(out_dir, "summary.md"), "w").write(table + "\n")


if __name__ == "__main__":
    main()
