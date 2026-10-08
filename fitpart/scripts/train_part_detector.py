"""Learned (non-deep) part detector on hierarchy nodes, evaluated leave-one-category-out.

For every part name shared by >= 2 PartNet-E categories, a gradient-boosted tree classifier
(scikit-learn HistGradientBoosting) scores candidate nodes as "is this part". Training uses
the other categories that have the part; testing uses the held-out category, so the
detector never sees the test category. A query (object, part present) selects the
highest-scoring candidate node; it is correct when that node has IoU >= 0.5 with the GT
part (all instances) or with any single instance of it.

usage: python train_part_detector.py CACHE_DIR [OUT_DIR]
"""
import glob, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from fitpart.node_features import FEATURE_NAMES

GROUPS = {
    "geometry": FEATURE_NAMES[:13],
    "geometry+topology": FEATURE_NAMES[:20],
    "geometry+topology+chebyshev": FEATURE_NAMES[:26],
}
TOPO = FEATURE_NAMES[13:20]
CHEB = FEATURE_NAMES[20:26]


def load(cache):
    objs = []
    for f in sorted(glob.glob(os.path.join(cache, "*.nodefeat.npz"))):
        d = np.load(f, allow_pickle=True)
        objs.append(dict(name=os.path.basename(f), cat=str(d["cat"]), X=d["X"], names=list(d["names"]),
                         lab_iou=d["lab_iou"], lab_inst=d["lab_inst"], sam=d["sam"], sig=d["sig"]))
    return objs


def design(o, p, cols):
    idx = [FEATURE_NAMES.index(c) for c in cols if c in FEATURE_NAMES]
    X = o["X"][:, idx]
    extra = []
    if "sam" in cols:
        extra.append(o["sam"][p][:, None])
    if "sig" in cols:
        extra.append(o["sig"][p][:, None])
    return np.hstack([X] + extra) if extra else X


def positives(o, p):
    return (np.nan_to_num(o["lab_iou"][p]) >= 0.5) | (np.nan_to_num(o["lab_inst"][p]) >= 0.5)


def main():
    cache = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    objs = load(cache)
    # part names present (with GT points) per category
    cats_of = defaultdict(set)
    for o in objs:
        for p, nm in enumerate(o["names"]):
            if not np.all(np.isnan(o["lab_iou"][p])):
                cats_of[nm].add(o["cat"])
    shared = sorted(nm for nm, cs in cats_of.items() if len(cs) >= 2)
    feature_sets = dict(GROUPS)
    feature_sets["all + SAM 3 + SigLIP scores (fusion)"] = FEATURE_NAMES + ["sam", "sig"]
    results = defaultdict(list)  # method -> list of (part, correct, iou)
    by_query = defaultdict(dict)
    for nm in shared:
        cats = sorted(cats_of[nm])
        for held in cats:
            train = [(o, o["names"].index(nm)) for o in objs if o["cat"] in cats and o["cat"] != held]
            test = [(o, o["names"].index(nm)) for o in objs if o["cat"] == held
                    and not np.all(np.isnan(o["lab_iou"][o["names"].index(nm)]))]
            if not test:
                continue
            for fs_name, cols in feature_sets.items():
                Xtr = np.vstack([design(o, p, cols) for o, p in train])
                ytr = np.concatenate([positives(o, p) for o, p in train])
                if ytr.sum() < 2:
                    continue
                clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                     class_weight="balanced", random_state=0)
                clf.fit(Xtr, ytr)
                for o, p in test:
                    s = clf.predict_proba(design(o, p, cols))[:, 1]
                    h = int(np.argmax(s))
                    rec = (nm, o["cat"], bool(positives(o, p)[h]), float(np.nan_to_num(o["lab_iou"][p][h])))
                    results[fs_name].append(rec)
                    by_query[(o["name"], nm)][fs_name] = rec
            for o, p in test:  # SAM 3 alone, and SAM 3 with the learned detector as fallback
                h = int(np.argmax(o["sam"][p]))
                rec = (nm, o["cat"], bool(positives(o, p)[h]), float(np.nan_to_num(o["lab_iou"][p][h])))
                results["SAM 3 (union score)"].append(rec)
                by_query[(o["name"], nm)]["SAM 3 (union score)"] = rec
                by_query[(o["name"], nm)]["sam_max"] = float(o["sam"][p].max())
    # SAM 3 when it detects something, else the learned geometry+topology detector
    for tau in (0.05, 0.15, 0.3):
        k = f"SAM 3, else geometry+topology (SAM 3 max < {tau})"
        for q, d in by_query.items():
            if "geometry+topology" in d and "SAM 3 (union score)" in d:
                results[k].append(d["geometry+topology"] if d["sam_max"] < tau else d["SAM 3 (union score)"])
        feature_sets[k] = None
    lines = [f"Leave-one-category-out over {len(shared)} part names shared by >= 2 categories: "
             + ", ".join(shared), "",
             "| Detector | queries | acc (IoU>=0.5, part or instance) | mean IoU with GT part |",
             "|---|---|---|---|"]
    order = ["SAM 3 (union score)"] + list(feature_sets)
    for k in order:
        r = results[k]
        lines.append(f"| {k} | {len(r)} | {np.mean([x[2] for x in r]):.3f} | {np.mean([x[3] for x in r]):.3f} |")
    # per part
    parts = sorted({x[0] for x in results["SAM 3 (union score)"]},
                   key=lambda nm: -sum(x[0] == nm for x in results["SAM 3 (union score)"]))
    lines += ["", "| Part | n | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
    for nm in parts:
        row = [nm, str(sum(x[0] == nm for x in results[order[0]]))]
        for k in order:
            v = [x[2] for x in results[k] if x[0] == nm]
            row.append(f"{np.mean(v):.2f}" if v else "-")
        lines.append("| " + " | ".join(row) + " |")
    txt = "\n".join(lines)
    print(txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "part_detector_loco.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
