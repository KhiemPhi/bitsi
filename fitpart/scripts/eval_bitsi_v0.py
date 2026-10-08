"""Run the existing BITSI pipeline on the same ShapeNetPart objects; Hungarian mIoU.

usage: python eval_bitsi_v0.py OUT_DIR [per_category] [workers] [dataset=shapenetpart|partnete] [Cat1,Cat2 (partnete)]
On PartNet-E, GT regions are instances plus one "rest" region for unlabeled points.
"""
import contextlib, io, json, os, sys, time
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from fitpart.data import (load_partnete, load_shapenetpart, partnete_objects, shapenetpart_categories,
                          shapenetpart_split)
from fitpart.metrics import class_agnostic_miou


def run(args):
    sid, oid = args
    from fitpart.baselines.bitsi_v0 import segment
    if os.path.isdir(str(oid)):  # PartNet-E object directory
        P, N, _, sem, ins = load_partnete(oid)
        gt = (ins if ins is not None else sem).copy()
        gt[sem < 0] = gt.max() + 1
    else:
        P, N, gt = load_shapenetpart(sid, oid)
    t = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lab = segment(P, N)
        return dict(sid=sid, oid=oid, miou=class_agnostic_miou(gt, lab)[0], n_seg=int(lab.max() + 1),
                    n_gt=int(len(np.unique(gt))), time=time.time() - t)
    except Exception as e:
        return dict(sid=sid, oid=oid, error=repr(e))


if __name__ == "__main__":
    out = sys.argv[1]; per = int(sys.argv[2]); workers = int(sys.argv[3])
    dataset = sys.argv[4] if len(sys.argv) > 4 else "shapenetpart"
    os.makedirs(out, exist_ok=True)
    taken = defaultdict(int); items = []
    if dataset == "partnete":
        cats = {}
        only = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else None
        for c, d in partnete_objects("test"):
            if only is not None and c not in only:
                continue
            if taken[c] < per:
                taken[c] += 1; items.append((c, d)); cats[c] = c
    else:
        cats = shapenetpart_categories()
        for sid, oid in shapenetpart_split("test"):
            if taken[sid] < per:
                taken[sid] += 1; items.append((sid, oid))
    with Pool(workers) as p:
        rs = p.map(run, items, chunksize=1)
    json.dump(rs, open(os.path.join(out, "per_object.json"), "w"))
    ok = [r for r in rs if "miou" in r]
    by = defaultdict(list)
    for r in ok:
        by[cats[r["sid"]]].append(r)
    for c in sorted(by):
        print(f"| {c} | {len(by[c])} | {np.mean([r['miou'] for r in by[c]]):.3f} | {np.mean([r['n_seg'] for r in by[c]]):.1f} |")
    print(f"class-mean Hungarian mIoU {np.mean([np.mean([r['miou'] for r in v]) for v in by.values()]):.3f}; "
          f"failures {len(rs) - len(ok)}; mean time {np.mean([r['time'] for r in ok]):.1f}s")
