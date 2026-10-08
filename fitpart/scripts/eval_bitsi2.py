"""BITSI v2 ablations on PartNet-E (class-agnostic Hungarian mIoU, instances + one rest region).

usage: python eval_bitsi2.py OUT_JSON Cat1,Cat2,... [max_per_cat=100] [workers=8]
"""
import json, os, sys
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.bitsi2 import BITSI2
from fitpart.data import PARTNETE_ROOT, load_partnete
from fitpart.metrics import class_agnostic_miou

VARIANTS = {
    "hist, children uncut": dict(cut_channels=("hist",), child_cuts=False),
    "hist, children uncut, root auto": dict(cut_channels=("hist",), child_cuts=False, root_spine="auto"),
    "hist, children uncut, order 0": dict(cut_channels=("hist",), child_cuts=False, n_max=0),
    "hist + radius steps, children uncut": dict(cut_channels=("hist", "section0"), child_cuts=False),
    "hist + radius steps z>5, children uncut": dict(cut_channels=("hist", "section0"), child_cuts=False, z_min=5.0),
    "hist only (recursive, children cut)": dict(cut_channels=("hist",)),
}
if os.environ.get("BITSI2_VARIANTS"):
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in os.environ["BITSI2_VARIANTS"].split(";")}


def run(d):
    P, N, C, sem, ins = load_partnete(d)
    gt = (ins if ins is not None else sem).copy()
    gt[sem < 0] = gt.max() + 1
    out = dict(cat=os.path.basename(os.path.dirname(d)), oid=os.path.basename(d))
    for name, kw in VARIANTS.items():
        try:
            lab, root = BITSI2(**kw).fit(P, N, C)
            out[name] = [float(class_agnostic_miou(gt, lab)[0]), int(lab.max() + 1)]
            if name == list(VARIANTS)[0]:
                out["tree"] = root.describe()
                out["loops"] = sum(c.loop for c in root.children)
                out["children"] = len(root.children)
        except Exception as e:
            out[name] = [float("nan"), 0]
            out.setdefault("errors", []).append(f"{name}: {e!r}")
    return out


if __name__ == "__main__":
    out_json, cats = sys.argv[1], sys.argv[2].split(",")
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    items = []
    for c in cats:
        root = os.path.join(PARTNETE_ROOT, "data", "test", c)
        ds = sorted((x for x in os.listdir(root) if os.path.exists(os.path.join(root, x, "label.npy"))), key=int)[:per]
        items += [os.path.join(root, x) for x in ds]
    with Pool(workers) as p:
        rs = p.map(run, items, chunksize=1)
    json.dump(rs, open(out_json, "w"))
    by = defaultdict(list)
    for r in rs:
        by[r["cat"]].append(r)
    errs = [e for r in rs for e in r.get("errors", [])]
    print(f"{len(rs)} objects, {len(errs)} errors", errs[:3])
    print("| Variant | " + " | ".join(cats) + " | class mean |")
    for k in VARIANTS:
        vals = [np.nanmean([r[k][0] for r in by[c]]) for c in cats]
        print(f"| {k} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | **{np.mean(vals):.3f}** |")
