"""Geometry-only FitPart (lambda = 0) on the ShapeNetPart test split.

Compares cut detectors on the same superpoints:
  cheb     - multi-scale Chebyshev inflections, D directions (ours)
  cheb_pnorm - same, slope normalized per profile as in the draft's s(c)
  axis     - same, three PCA axes only
  single   - single Chebyshev degree
  extrema  - Chebyshev extrema instead of inflections
  jump     - ITSI jump threshold delta = 0.2
  none     - no cuts (affinity 1 everywhere: merge order set by graph structure only)

Reports superpoint purity, hierarchy recall (IoU >= 0.5), and class-agnostic mIoU
when the tree is cut at the ground-truth part count (an oracle-K diagnostic that
needs no merge threshold).

usage: python eval_geometry_shapenetpart.py OUT_DIR [per_category] [workers]
"""
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.affinity import detect_all_cuts, geometric_affinity, normal_smoothness
from fitpart.data import load_shapenetpart, shapenetpart_categories, shapenetpart_split
from fitpart.hierarchy import agglomerate
from fitpart.metrics import class_agnostic_miou, hierarchy_recall, superpoint_purity
from fitpart.superpoints import superpoint_adjacency, superpoints

VARIANTS = {
    "cheb": dict(detector="cheb"),
    "cheb_ext": dict(detector="cheb", extent_weight=True),
    "cheb_pnorm": dict(detector="cheb", norm="profile"),
    "axis": dict(detector="cheb", axis_only=True),
    "single": dict(detector="single"),
    "extrema": dict(detector="extrema"),
    "jump": dict(detector="jump", delta=0.2),
    "none": None,
    "cheb+smooth": dict(detector="cheb", smooth=True),
    "smooth": dict(smooth_only=True),
}


def run_one(args):
    sid, oid = args
    P, Nm, gt = load_shapenetpart(sid, oid)
    t0 = time.time()
    sp, g = superpoints(P, Nm)
    pairs = superpoint_adjacency(sp, g)
    t_sp = time.time() - t0
    out = dict(sid=sid, oid=oid, n_gt=int(len(np.unique(gt))), purity=superpoint_purity(sp, gt),
               n_sp=int(sp.max() + 1))
    for name, kw in VARIANTS.items():
        t1 = time.time()
        kw = dict(kw) if kw else None
        smooth = kw.pop("smooth", False) if kw else False
        if kw is None:
            w = np.ones(len(pairs))
        elif kw.pop("smooth_only", False):
            w = normal_smoothness(Nm, sp, pairs)
        else:
            cuts = detect_all_cuts(P, **kw)
            w = geometric_affinity(P, sp, pairs, cuts)
            if smooth:
                w = w * normal_smoothness(Nm, sp, pairs)
        H = agglomerate(sp.max() + 1, pairs, w)
        rec, _ = hierarchy_recall(H, sp, gt)
        lab, _ = H.flat(n_clusters=out["n_gt"], sizes=np.bincount(sp), min_frac=0.02)
        miou, _ = class_agnostic_miou(gt, lab[sp])
        out[name] = dict(recall=rec, miou_oracleK=miou, time=time.time() - t1 + t_sp)
    return out


def main():
    out_dir = sys.argv[1]
    per_cat = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    os.makedirs(out_dir, exist_ok=True)
    cats = shapenetpart_categories()
    taken = defaultdict(int)
    items = []
    for sid, oid in shapenetpart_split("test"):
        if taken[sid] < per_cat:
            taken[sid] += 1
            items.append((sid, oid))
    print(f"{len(items)} objects", flush=True)
    with Pool(workers) as pool:
        results = pool.map(run_one, items, chunksize=2)
    with open(os.path.join(out_dir, "per_object.json"), "w") as f:
        json.dump(results, f)

    # category-averaged then instance-averaged summaries
    by_cat = defaultdict(list)
    for r in results:
        by_cat[cats[r["sid"]]].append(r)
    names = list(VARIANTS)
    lines = ["| Category | n | SP purity | " + " | ".join(f"{n} rec / mIoU" for n in names) + " |",
             "|" + "---|" * (3 + len(names))]
    cat_means = defaultdict(list)
    for c in sorted(by_cat):
        rs = by_cat[c]
        row = [c, str(len(rs)), f"{np.mean([r['purity'] for r in rs]):.3f}"]
        for n in names:
            rec = np.mean([r[n]["recall"] for r in rs])
            mi = np.mean([r[n]["miou_oracleK"] for r in rs])
            cat_means[n].append((rec, mi))
            row.append(f"{rec:.3f} / {mi:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    row = ["**Class mean**", str(len(results)),
           f"{np.mean([np.mean([r['purity'] for r in rs]) for rs in by_cat.values()]):.3f}"]
    for n in names:
        a = np.array(cat_means[n])
        row.append(f"**{a[:, 0].mean():.3f} / {a[:, 1].mean():.3f}**")
    lines.append("| " + " | ".join(row) + " |")
    row = ["Time (s/obj)", "", ""] + [f"{np.mean([r[n]['time'] for r in results]):.2f}" for n in names]
    lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
