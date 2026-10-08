"""Primitive decomposition vs training-free sweep baselines on PartNet-E (class-agnostic).

GT regions: instances plus one "rest" region for unlabeled points (as eval_bitsi_v0.py).
Methods: primitive (axial Chebyshev cuts on R(s) + residual components), its two halves, HOG z>3
sweep events along the longest PCA axis (best training-free sweep detector so far), and BITSI v0
read from --bitsi JSON (run separately in the pointcloud env).

usage: python eval_primitive.py OUT_JSON Cat1,Cat2,... [max_per_cat=20] [workers=12]
"""
import json, os, sys
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.data import PARTNETE_ROOT, load_partnete
from fitpart.metrics import class_agnostic_miou
from fitpart.primitive import primitive_decomposition
from fitpart.profiles import pca_axes
from fitpart.sweep_events import events_along


def run(d):
    P, N, C, sem, ins = load_partnete(d)
    gt = (ins if ins is not None else sem).copy()
    gt[sem < 0] = gt.max() + 1
    out = dict(cat=os.path.basename(os.path.dirname(d)), oid=os.path.basename(d), n_gt=int(len(np.unique(gt))))
    m = lambda lab: (float(class_agnostic_miou(gt, lab)[0]), int(len(np.unique(lab))))
    lab, info = primitive_decomposition(P, N)
    out["primitive"] = m(lab)
    out["axis_score"] = info["axis_score"]
    out["primitive_axial_only"] = m(primitive_decomposition(P, N, residual=False)[0])
    out["primitive_residual_only"] = m(primitive_decomposition(P, N, axial_cuts=False)[0])
    out["primitive_hog"] = m(primitive_decomposition(P, N, C, axial="hog")[0])
    out["primitive_hog_pca"] = m(primitive_decomposition(P, N, C, axial="hog_pca")[0])
    out["hog_on_symmetry_axis"] = m(primitive_decomposition(P, N, C, axial="hog", residual=False)[0])
    ev, prof = events_along(P, pca_axes(P)[0], N, C, ("hist",))
    out["hog_events"] = m(np.searchsorted(np.sort([c.t for c in ev]), prof["coord"]))
    out["whole_object"] = m(np.zeros(len(P), int))
    return out


if __name__ == "__main__":
    out_json, cats = sys.argv[1], sys.argv[2].split(",")
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    items = []
    for c in cats:
        root = os.path.join(PARTNETE_ROOT, "data", "test", c)
        ds = sorted((x for x in os.listdir(root) if os.path.exists(os.path.join(root, x, "label.npy"))), key=int)[:per]
        items += [os.path.join(root, x) for x in ds]
    with Pool(workers) as p:
        rs = p.map(run, items, chunksize=1)
    json.dump(rs, open(out_json, "w"))
    keys = [k for k in rs[0] if isinstance(rs[0][k], (list, tuple))]
    by = defaultdict(list)
    for r in rs:
        by[r["cat"]].append(r)
    print("| Category | n | " + " | ".join(keys) + " | axis score |")
    for c in cats:
        v = by[c]
        print(f"| {c} | {len(v)} | " + " | ".join(f"{np.mean([r[k][0] for r in v]):.3f} ({np.mean([r[k][1] for r in v]):.1f})" for k in keys)
              + f" | {np.mean([r['axis_score'] for r in v]):.2f} |")
    print("| class mean | | " + " | ".join(f"{np.mean([np.mean([r[k][0] for r in v]) for v in by.values()]):.3f}" for k in keys) + " | |")
