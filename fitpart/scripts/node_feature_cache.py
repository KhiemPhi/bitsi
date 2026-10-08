"""Per object: candidate-node features, per-part-name labels (node IoU with the GT part and
with its best single instance), and per-part SAM 3 / SigLIP node scores for fusion.
Writes <obj>.nodefeat.npz next to each cache entry.

usage: python node_feature_cache.py CACHE_DIR META_JSON [workers]
"""
import glob, json, os, pickle, sys
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

META = None


def run(f):
    out = f[:-4] + ".nodefeat.npz"
    if os.path.exists(out):
        return f, "cached"
    from fitpart import lifting
    from fitpart.metrics import node_gt_iou
    from fitpart.node_features import node_features
    from fitpart.pipeline import Config, FitPart
    from fitpart.scales import metric_scale
    from fitpart.selection import CANONICAL, relevance
    from fitpart.sam3_select import union_score
    L = pickle.load(open(f, "rb"))
    S3 = pickle.load(open(f[:-4] + ".sam3.pkl", "rb"))
    P = L["points"].astype(float)
    scale = metric_scale(P, L["cat"]) or 1.0 / float(np.ptp(P, 0).max())
    fp = FitPart(Config())
    res = fp.assemble(L)
    sp, H = res.sp, res.hier
    X, cand = node_features(P, sp, L["pairs"], H, scale)
    names = META[L["cat"]]
    sem, ins = L["sem"], L.get("ins")
    lab_iou, lab_inst, sam, sig = [], [], [], []
    for p, nm in enumerate(names):
        g = (sem == p).astype(int)
        if g.any():
            iou, ids = node_gt_iou(H, sp, g)
            lab_iou.append(iou[cand, list(ids).index(1)])
            if ins is not None:
                inst = np.where(sem == p, ins, -1)
                iou_i, ids_i = node_gt_iou(H, sp, inst)
                cols = [k for k, v in enumerate(ids_i) if v >= 0]
                lab_inst.append(iou_i[cand][:, cols].max(1) if cols else np.zeros(len(cand)))
            else:
                lab_inst.append(lab_iou[-1])
        else:
            lab_iou.append(np.full(len(cand), np.nan))
            lab_inst.append(np.full(len(cand), np.nan))
        sam.append(union_score(H, sp, S3[nm])[cand])
        t = lifting.encode_text([f"the {nm}"] + list(CANONICAL))
        sig.append(relevance(res.z_nodes, t[0], t[1:])[cand])
    np.savez_compressed(out, X=X.astype(np.float32), cand=cand, lab_iou=np.array(lab_iou, np.float32),
                        lab_inst=np.array(lab_inst, np.float32), sam=np.array(sam, np.float32),
                        sig=np.array(sig, np.float32), names=np.array(names), cat=L["cat"],
                        node_sizes=np.array([np.isin(sp, H.members[h]).sum() for h in cand]), n=len(sp))
    return f, "ok"


def _init(meta, text_cache):
    global META
    META = meta
    from fitpart import lifting
    lifting._text_cache.update(text_cache)


def main():
    cache, meta_json = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    meta = json.load(open(meta_json))
    from fitpart import lifting
    from fitpart.selection import CANONICAL
    files = [f for f in sorted(glob.glob(os.path.join(cache, "*.pkl"))) if ".sam3" not in os.path.basename(f)]
    phrases = set(CANONICAL)
    for f in files:
        phrases |= {f"the {p}" for p in meta[os.path.basename(f).split("__")[0]]}
    lifting.precompute_text(sorted(phrases), device="cpu")
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(meta, dict(lifting._text_cache))) as pool:
        for f, st in pool.imap_unordered(run, files):
            print(os.path.basename(f), st, flush=True)


if __name__ == "__main__":
    main()
