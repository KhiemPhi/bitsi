"""Compare the non-learned geometric part detector with SAM 3 selection on PartNet-E part
names that have a geometric definition (handle, knob, button, switch, lid, cap, leg, wheel).

usage: python eval_geometric_detect.py CACHE_DIR META_JSON [workers]
"""
import glob, json, os, pickle, sys
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.geometric_detect import DEFINED, geometric_scores, node_descriptors
from fitpart.pipeline import Config, FitPart
from fitpart.scales import metric_scale
from fitpart.sam3_select import union_score
from fitpart.selection import node_membership, select

META = None


def run(f):
    L = pickle.load(open(f, "rb"))
    S3 = pickle.load(open(f[:-4] + ".sam3.pkl", "rb"))
    names = META[L["cat"]]
    sem = L["sem"]
    present = [p for p in np.unique(sem[sem >= 0]) if names[p] in DEFINED]
    if not present:
        return []
    P = L["points"].astype(float)
    scale = metric_scale(P, L["cat"])
    if scale is None:
        return []
    fp = FitPart(Config())
    res = fp.assemble(L)
    sp, H = res.sp, res.hier
    s = np.bincount(sp).astype(float)
    Hm, sizes = node_membership(H, s)
    desc = node_descriptors(P, sp, res.pairs[: len(L["pairs"])], H, scale=scale)
    n = len(sp)
    out = []
    for p in present:
        g = sem == p
        iou = lambda h: float((Hm[h][sp] & g).sum() / (Hm[h][sp] | g).sum())
        pick = lambda r: iou(select(r, sizes, None, fp.cfg.eps_r, min_size=0.001 * n)[0])
        geo = geometric_scores(names[p], desc, H.n_nodes)
        sam = union_score(H, sp, S3[names[p]])
        empty = sam.max() < 0.05
        rec = dict(cat=L["cat"], part=names[p], empty=empty,
                   sam3=pick(sam), geometric=pick(geo),
                   sam3_else_geometric=pick(geo if empty else sam),
                   product=pick(sam * (0.2 + geo)),
                   blend=pick(sam + 0.5 * geo))
        out.append(rec)
    return out


def main():
    global META
    cache, meta_json = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    META = json.load(open(meta_json))
    files = [f for f in sorted(glob.glob(os.path.join(cache, "*.pkl"))) if ".sam3" not in os.path.basename(f)]
    with Pool(workers) as pool:
        recs = [r for rs in pool.map(run, files, chunksize=1) for r in rs]
    keys = ["sam3", "geometric", "sam3_else_geometric", "blend", "product"]
    e = np.array([r["empty"] for r in recs])
    print(f"{len(recs)} queries with a geometric definition; SAM 3 empty on {e.mean():.2f}")
    print("| score | mean IoU | acc | acc where SAM 3 empty | acc where SAM 3 fires |")
    print("|---|---|---|---|---|")
    for k in keys:
        v = np.array([r[k] for r in recs])
        print(f"| {k} | {v.mean():.3f} | {(v >= 0.5).mean():.3f} | {(v[e] >= 0.5).mean():.3f} | "
              f"{(v[~e] >= 0.5).mean():.3f} |")
    by = defaultdict(list)
    for r in recs:
        by[r["part"]].append(r)
    print("\n| part | n | SAM 3 acc | geometric acc | SAM 3 else geometric acc |\n|---|---|---|---|---|")
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        a = lambda key: np.mean([x[key] >= 0.5 for x in v])
        print(f"| {k} | {len(v)} | {a('sam3'):.2f} | {a('geometric'):.2f} | {a('sam3_else_geometric'):.2f} |")


if __name__ == "__main__":
    main()
