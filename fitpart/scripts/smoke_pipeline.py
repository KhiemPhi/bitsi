"""Smoke test: full FitPart on a few ShapeNetPart objects, with stage timings."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import faulthandler; faulthandler.dump_traceback_later(60, repeat=True)
from fitpart.data import load_shapenetpart, shapenetpart_categories, shapenetpart_split
from fitpart.metrics import class_agnostic_miou, hierarchy_recall
from fitpart.pipeline import Config, FitPart
from fitpart.viz import show_segmentations

out = sys.argv[1]
vl = sys.argv[2] if len(sys.argv) > 2 else "clip"
os.makedirs(out, exist_ok=True)
cats = shapenetpart_categories()
queries = {"Mug": ["the handle", "the mug", "the body of the mug"],
           "Knife": ["the handle", "the blade"],
           "Chair": ["the chair back", "the seat", "the legs"]}
for cat, qs in queries.items():
    sid, oid = [x for x in shapenetpart_split("test") if cats[x[0]] == cat][1]
    P, N, gt = load_shapenetpart(sid, oid)
    fp = FitPart(Config(vl_model=vl))
    t = time.time()
    L = fp.lift(P, N)
    res = fp.assemble(L)
    print(cat, "build %.1fs" % (time.time() - t), {k: round(v, 1) for k, v in res.timings.items()},
          "masks/view", L["n_masks"], flush=True)
    K = len(np.unique(gt))
    lab, _ = res.hier.flat(n_clusters=K, sizes=np.bincount(res.sp), min_frac=0.02)
    print("  oracle-K mIoU %.3f  recall %.3f" % (class_agnostic_miou(gt, lab[res.sp])[0],
                                                hierarchy_recall(res.hier, res.sp, gt)[0]), flush=True)
    sels = []
    for q in qs:
        h, st, r = fp.select(res, q, P, N, use_feasibility=False)
        idx = np.isin(res.sp, res.hier.members[h])
        sels.append(idx.astype(float))
        print("  %-22s size %4d  gt-part counts %s  r=%.3f" % (q, idx.sum(),
              np.bincount(gt[idx], minlength=gt.max() + 1)[np.unique(gt)], r[h]), flush=True)
    show_segmentations(P, [gt, lab[res.sp]] + sels, ["GT", "oracle-K"] + qs,
                       os.path.join(out, f"pipeline_{cat}.png"), views=((20, 30),))
