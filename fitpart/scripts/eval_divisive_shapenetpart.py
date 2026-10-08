"""Divisive (recursive) geometric hierarchy on ShapeNetPart: Chebyshev cuts vs ITSI jumps.

usage: python eval_divisive_shapenetpart.py OUT_DIR [per_category] [workers]
"""
import json, os, sys, time
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.data import load_shapenetpart, shapenetpart_categories, shapenetpart_split
from fitpart.divisive import divisive_tree, flat_k, node_recall
from fitpart.metrics import class_agnostic_miou
from fitpart.superpoints import knn_graph

VARIANTS = {
    "cheb": dict(detector="cheb"),
    "cheb_persist": dict(detector="cheb", persist_levels=2),
    "cheb_axis": dict(detector="cheb", axis_only=True),
    "single": dict(detector="single"),
    "extrema": dict(detector="extrema"),
    "jump0.2": dict(detector="jump", delta=0.2),
}


def run_one(args):
    sid, oid = args
    P, Nm, gt = load_shapenetpart(sid, oid)
    g = knn_graph(P, Nm)
    K = len(np.unique(gt))
    out = dict(sid=sid, oid=oid, n_gt=K)
    for name, kw in VARIANTS.items():
        t0 = time.time()
        kw = dict(kw)
        pl = kw.pop("persist_levels", None)
        nodes, order = divisive_tree(P, g, detector_kwargs=kw, persist_levels=pl)
        lab = flat_k(nodes, order, K)
        out[name] = dict(recall=node_recall(nodes, gt), miou_oracleK=class_agnostic_miou(gt, lab)[0],
                         n_nodes=len(nodes), time=time.time() - t0)
    return out


def main():
    out_dir = sys.argv[1]
    per_cat = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    os.makedirs(out_dir, exist_ok=True)
    cats = shapenetpart_categories()
    taken = defaultdict(int)
    items = []
    for sid, oid in shapenetpart_split("test"):
        if taken[sid] < per_cat:
            taken[sid] += 1
            items.append((sid, oid))
    with Pool(workers) as pool:
        results = pool.map(run_one, items, chunksize=1)
    json.dump(results, open(os.path.join(out_dir, "per_object.json"), "w"))
    by_cat = defaultdict(list)
    for r in results:
        by_cat[cats[r["sid"]]].append(r)
    names = list(VARIANTS)
    lines = ["| Category | n | " + " | ".join(f"{n} rec / mIoU" for n in names) + " |",
             "|" + "---|" * (2 + len(names))]
    cm = defaultdict(list)
    for c in sorted(by_cat):
        rs = by_cat[c]
        row = [c, str(len(rs))]
        for n in names:
            a = (np.mean([r[n]["recall"] for r in rs]), np.mean([r[n]["miou_oracleK"] for r in rs]))
            cm[n].append(a)
            row.append(f"{a[0]:.3f} / {a[1]:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| **Class mean** | %d | " % len(results) + " | ".join(
        f"**{np.mean([a[0] for a in cm[n]]):.3f} / {np.mean([a[1] for a in cm[n]]):.3f}**" for n in names) + " |")
    lines.append("| Time (s/obj) | | " + " | ".join(f"{np.mean([r[n]['time'] for r in results]):.2f}" for n in names) + " |")
    t = "\n".join(lines)
    print(t)
    open(os.path.join(out_dir, "summary.md"), "w").write(t + "\n")


if __name__ == "__main__":
    main()
