"""Held-out test of the frozen BITSI v2 configuration: PartNet-E categories NOT used in development.

Compares on the same objects: BITSI v2 (defaults), BITSI v1 primitive (residual + HOG on the PCA
axis), HOG sweep events (longest PCA axis), whole object, and BITSI v0 (read from its per-object
JSON). Class-agnostic Hungarian mIoU, instances + one rest region; stratified bootstrap.

usage: python eval_bitsi2_heldout.py OUT_JSON BITSI_V0_JSON EXCLUDE_CATS [per_cat=4] [workers=12]
       python eval_bitsi2_heldout.py OUT_JSON BITSI_V0_JSON include:CATS [per_cat] [workers] [start=20]
         (in-domain held-out: objects start.. of the development categories, never used)
"""
import json, os, sys
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.bitsi2 import BITSI2
from fitpart.data import PARTNETE_ROOT, load_partnete
from fitpart.metrics import class_agnostic_miou
from fitpart.primitive import primitive_decomposition
from fitpart.profiles import pca_axes
from fitpart.sweep_events import events_along


def run(d):
    P, N, C, sem, ins = load_partnete(d)
    gt = (ins if ins is not None else sem).copy()
    gt[sem < 0] = gt.max() + 1
    m = lambda lab: [float(class_agnostic_miou(gt, lab)[0]), int(len(np.unique(lab)))]
    out = dict(cat=os.path.basename(os.path.dirname(d)), oid=os.path.basename(d), dir=d)
    lab, root = BITSI2().fit(P, N, C)
    out["BITSI v2"] = m(lab)
    out["tree"] = root.describe()
    out["BITSI v1 primitive"] = m(primitive_decomposition(P, N, C, axial="hog_pca")[0])
    ev, prof = events_along(P, pca_axes(P)[0], N, C, ("hist",))
    out["HOG events"] = m(np.searchsorted(np.sort([c.t for c in ev]), prof["coord"]))
    out["whole object"] = m(np.zeros(len(P), int))
    return out


if __name__ == "__main__":
    out_json, v0_json = sys.argv[1], sys.argv[2]
    include = sys.argv[3].startswith("include:")
    cat_set = set(sys.argv[3].split(":", 1)[-1].split(","))
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    start = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    base = os.path.join(PARTNETE_ROOT, "data", "test")
    items = []
    for c in sorted(os.listdir(base)):
        if (c in cat_set) != include:
            continue
        ds = sorted((x for x in os.listdir(os.path.join(base, c)) if os.path.exists(os.path.join(base, c, x, "label.npy"))), key=int)[start:start + per]
        items += [os.path.join(base, c, x) for x in ds]
    with Pool(workers) as p:
        rs = p.map(run, items, chunksize=1)
    v0 = {}
    for f in v0_json.split(","):
        v0.update({r["oid"].rstrip("/"): r for r in json.load(open(f)) if "miou" in r})
    for r in rs:
        b = v0.get(r["dir"].rstrip("/"))
        r["BITSI v0"] = [b["miou"], b["n_seg"]] if b else [float("nan"), 0]
    json.dump(rs, open(out_json, "w"))
    keys = ["BITSI v2", "BITSI v1 primitive", "HOG events", "BITSI v0", "whole object"]
    byc = defaultdict(list)
    for r in rs:
        byc[r["cat"]].append(r)
    cats = sorted(byc)
    cm = lambda rows, k: np.nanmean([np.nanmean([r[k][0] for r in rows if r["cat"] == c]) for c in cats])
    print(f"{len(rs)} objects, {len(cats)} categories; BITSI v0 matched {sum(np.isfinite(r['BITSI v0'][0]) for r in rs)}")
    for k in keys:
        print(f"| {k} | {cm(rs, k):.3f} | {np.mean([r[k][1] for r in rs]):.1f} segs |")
    rng = np.random.default_rng(0)
    for b in keys[1:4]:
        bs = []
        for _ in range(3000):
            rows = [x for c in cats for x in [byc[c][i] for i in rng.integers(len(byc[c]), size=len(byc[c]))]]
            bs.append(cm(rows, "BITSI v2") - cm(rows, b))
        wins = sum(np.nanmean([r["BITSI v2"][0] for r in byc[c]]) > np.nanmean([r[b][0] for r in byc[c]]) for c in cats)
        print(f"BITSI v2 - {b}: {cm(rs, 'BITSI v2') - cm(rs, b):+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}], better in {wins}/{len(cats)} categories")
