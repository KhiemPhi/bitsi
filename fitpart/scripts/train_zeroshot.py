"""Zero-shot geometric-topological part detector, gated with SAM 3 (coloured renders).

Z2 (no 3D labels): per part name, a gradient-boosted classifier on node features (geometry +
topology) is trained on SAM 3 pseudo-labels: positives = nodes with SAM 3 score >= max(0.5,
0.9 x the object's best) in objects where SAM 3 is confident (best >= 0.5); negatives = nodes
scoring < 0.1 in those objects; objects where SAM 3 found nothing give no labels.
Splits: leave-one-category-out (never sees the test category) and leave-one-object-out (other
unlabeled objects of the same category, i.e. test-time adaptation without labels).
Z1 (cross-dataset): trained on ShapeNetPart GT labels (*.snpfeat.npz), part types shared with
PartNet-E; no PartNet-E labels.
Reference: supervised leave-one-category-out on PartNet-E GT labels.
Every detector is used as a fallback: SAM 3's choice when its best node score >= 0.05.

usage: python train_zeroshot.py PNE_CACHE [SNP_CACHE] [OUT_DIR]
"""
import glob, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from fitpart.node_features import FEATURE_NAMES

COLS = list(range(20))  # geometry + topology
GATE = 0.05
SNP_TYPES = ("handle", "leg", "wheel", "seat", "blade", "base", "screen")
# categories whose PartNet-E objects may share ShapeNet meshes with ShapeNetPart training data
SHAPENET_OVERLAP = {"Chair", "Table", "Lamp", "Laptop", "Knife"}


def clf():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          class_weight="balanced", random_state=0)


def load_pne(cache):
    objs = []
    for f in sorted(glob.glob(os.path.join(cache, "*.nodefeat.npz"))):
        d = np.load(f, allow_pickle=True)
        s = np.load(f.replace(".nodefeat.npz", ".samc.npz"))
        objs.append(dict(name=os.path.basename(f), cat=str(d["cat"]), X=d["X"][:, COLS], names=list(d["names"]),
                         lab_iou=d["lab_iou"], lab_inst=d["lab_inst"], sam=s["samc"]))
    return objs


def gt_pos(o, p):
    return (np.nan_to_num(o["lab_iou"][p]) >= 0.5) | (np.nan_to_num(o["lab_inst"][p]) >= 0.5)


def pseudo(o, p):
    s = o["sam"][p]
    m = s.max()
    if m < 0.5:
        return None
    pos = s >= max(0.5, 0.9 * m)
    neg = s < 0.1
    keep = pos | neg
    return o["X"][keep], pos[keep]


def fit(pairs, label_fn):
    X, y = [], []
    for o, p in pairs:
        r = label_fn(o, p)
        if r is None:
            continue
        X.append(r[0]); y.append(r[1])
    if not X:
        return None
    X, y = np.vstack(X), np.concatenate(y)
    if y.sum() < 2 or (~y).sum() < 2:
        return None
    return clf().fit(X, y)


def main():
    pne = load_pne(sys.argv[1])
    snp_cache = sys.argv[2] if len(sys.argv) > 2 else None
    out_dir = sys.argv[3] if len(sys.argv) > 3 else None
    by_name = defaultdict(list)  # part name -> [(obj, index)] where the part has GT points
    for o in pne:
        for p, nm in enumerate(o["names"]):
            if not np.all(np.isnan(o["lab_iou"][p])):
                by_name[nm].append((o, p))
    all_pairs = defaultdict(list)  # part name -> all (obj, index) incl. objects without GT points
    for o in pne:
        for p, nm in enumerate(o["names"]):
            all_pairs[nm].append((o, p))
    # Z1: ShapeNetPart-trained models per shared type
    z1 = {}
    if snp_cache:
        snp = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(snp_cache, "*.snpfeat.npz")))]
        for tp in SNP_TYPES:
            X, y = [], []
            for d in snp:
                names = list(d["types"])
                if tp in names:
                    k = names.index(tp)
                    X.append(d["X"][:, COLS]); y.append(d["pos"][k].astype(bool))
            if X and np.concatenate(y).sum() >= 2:
                z1[tp] = clf().fit(np.vstack(X), np.concatenate(y))
        print("Z1 trained types:", sorted(z1), f"({len(snp)} ShapeNetPart objects)")
    rec = defaultdict(list)  # method -> list of (name, cat, correct, sam_empty)

    def score_pick(o, p, model):
        s = model.predict_proba(o["X"])[:, 1]
        return int(np.argmax(s))

    for nm, pairs in by_name.items():
        cats = sorted({o["cat"] for o, _ in pairs})
        for o, p in pairs:
            empty = o["sam"][p].max() < GATE
            h_sam = int(np.argmax(o["sam"][p]))
            ok = lambda h: bool(gt_pos(o, p)[h])
            rec["SAM 3 alone"].append((nm, o["cat"], ok(h_sam), empty))
            # Z2 leave-one-category-out
            if len(cats) >= 2:
                rec["SAM 3 alone (same 242 queries)"].append((nm, o["cat"], ok(h_sam), empty))
                m = fit([(a, b) for a, b in all_pairs[nm] if a["cat"] != o["cat"]], pseudo)
                h = score_pick(o, p, m) if (m is not None and empty) else h_sam
                rec["SAM 3, else Z2 pseudo-label (leave-category-out)"].append((nm, o["cat"], ok(h), empty))
                # supervised reference (GT labels of other categories)
                ms = fit([(a, b) for a, b in by_name[nm] if a["cat"] != o["cat"]], lambda a, b: (a["X"], gt_pos(a, b)))
                h = score_pick(o, p, ms) if (ms is not None and empty) else h_sam
                rec["SAM 3, else supervised (leave-category-out, reference)"].append((nm, o["cat"], ok(h), empty))
            # Z2 leave-one-object-out (any category, incl. single-category part names)
            m = fit([(a, b) for a, b in all_pairs[nm] if a["name"] != o["name"]], pseudo)
            h = score_pick(o, p, m) if (m is not None and empty) else h_sam
            rec["SAM 3, else Z2 pseudo-label (leave-object-out)"].append((nm, o["cat"], ok(h), empty))
            # Z1 cross-dataset
            if nm in z1:
                h = score_pick(o, p, z1[nm]) if empty else h_sam
                rec["SAM 3, else Z1 ShapeNetPart-trained"].append((nm, o["cat"], ok(h), empty))
                rec["SAM 3 alone (Z1 types)"].append((nm, o["cat"], ok(h_sam), empty))
                if o["cat"] not in SHAPENET_OVERLAP:
                    rec["SAM 3, else Z1 (non-ShapeNet categories)"].append((nm, o["cat"], ok(h), empty))
                    rec["SAM 3 alone (Z1 types, non-ShapeNet categories)"].append((nm, o["cat"], ok(h_sam), empty))
    lines = ["| Method | queries | accuracy | accuracy where SAM 3 is empty | SAM 3 empty share |", "|---|---|---|---|---|"]
    for k, v in rec.items():
        a = np.array([x[2] for x in v]); e = np.array([x[3] for x in v])
        lines.append(f"| {k} | {len(v)} | {a.mean():.3f} | {a[e].mean() if e.any() else float('nan'):.3f} | {e.mean():.2f} |")
    # paired bootstrap over queries: Z1 vs SAM 3 alone on the same queries
    rng = np.random.default_rng(0)
    for a, b in (("SAM 3, else Z1 ShapeNetPart-trained", "SAM 3 alone (Z1 types)"),
                 ("SAM 3, else Z1 (non-ShapeNet categories)", "SAM 3 alone (Z1 types, non-ShapeNet categories)"),
                 ("SAM 3, else supervised (leave-category-out, reference)", "SAM 3 alone (same 242 queries)")):
        if a in rec and b in rec and len(rec[a]) == len(rec[b]):
            d = np.array([x[2] for x in rec[a]], float) - np.array([y[2] for y in rec[b]], float)
            boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
            lines.append(f"\n{a} vs {b}: {d.mean():+.3f} [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]"
                         f" (queries changed: +{int((d > 0).sum())} / -{int((d < 0).sum())})")
    txt = "\n".join(lines)
    print(txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "zeroshot_detector.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
