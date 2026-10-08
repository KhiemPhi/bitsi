"""Us vs baselines: merge every per-object result file of a dataset and tabulate.

Every method is scored on the same objects (those with a result for every method listed).
Class means (categories weighted equally); v4 - method differences with 95% bootstrap intervals
(objects resampled within category).
usage: [FAIL_AS_WHOLE=1] python summarize_all.py DATASET METHODS FILE [FILE ...]
  DATASET = partnete | shapenetpart; METHODS = comma-separated names as stored in the files
"""
import json, os, sys
from collections import defaultdict
import numpy as np

SPLITS = {"partnete": [("in-domain (9 swept categories, held-out)", {"indomain"}),
                       ("out-of-domain (36 other categories, held-out)", {"outdomain", "oodrest"}),
                       ("all PartNet-E test objects", {"dev", "indomain", "ooddev", "outdomain", "oodrest"})],
          "shapenetpart": [("ShapeNetPart test", {"test"})]}
NAMES = {"v4": "BITSI v4 (ours)", "v3": "BITSI v3", "v2": "BITSI v2", "v1": "BITSI v1", "v0": "BITSI v0",
         "hog": "sweep events (HOG)", "whole": "whole object", "find3d": "Find3D (GT part names)",
         "partfield": "PartField (oracle K)", "partfield_v4k": "PartField (K = v4 count)", "coacd": "CoACD", "superquadric": "EMS superquadrics"}


def rename(r):
    """v4 result files store the frozen variant under its grid name."""
    for k in list(r):
        if k.startswith("v4 children=mdl cuts=no pen10"):
            r["v4"] = {kk: vv for kk, vv in r[k].items() if kk not in ("u", "n_children")}
    return r


def fail_as_whole(rows, methods):
    """A method that raised on an object is scored as returning the whole object (1 segment)
    instead of dropping the object (dropping favours the failing method). Needs "whole" rows."""
    n = 0
    for r in rows:
        for m in methods:
            if m in r and "error" in r[m] and "whole" in r and "error" not in r["whole"]:
                r[m] = dict(r["whole"])
                n += 1
    return n


def merge(files):
    rows = {}
    for f in files:
        for l in open(f):
            if l.strip():
                r = rename(json.loads(l))
                cur = rows.setdefault(r["key"], dict(key=r["key"], cat=r["cat"], split=r["split"]))
                for k, v in r.items():
                    if isinstance(v, dict) and k not in cur:
                        cur[k] = v
    return list(rows.values())


def main():
    ds, methods, files = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
    rows = merge(files)
    if os.environ.get("FAIL_AS_WHOLE"):
        print("failures scored as the whole object:", fail_as_whole(rows, methods))
    for title, splits in SPLITS[ds]:
        sub = [r for r in rows if r["split"] in splits and all(m in r and "error" not in r[m] for m in methods)]
        n_all = sum(r["split"] in splits for r in rows)
        if not sub:
            print(f"\n### {title}: no objects with every method yet ({n_all} objects)")
            continue
        cats = sorted({r["cat"] for r in sub})
        print(f"\n### {title}: {len(sub)} objects ({len(cats)} categories) with every method, of {n_all}")
        print("| method | mIoU | sym mIoU | ARI | segments (GT) |\n|---|---|---|---|---|")
        for m in methods:
            per = defaultdict(list)
            for r in sub:
                per[r["cat"]].append([r[m]["miou"], r[m]["sym_miou"], r[m]["ari"], r[m]["n_pred"], r[m]["n_gt"]])
            cm = np.mean([np.mean(v, 0) for v in per.values()], 0)
            print(f"| {NAMES.get(m, m)} | {cm[0]:.3f} | {cm[1]:.3f} | {cm[2]:.3f} | {cm[3]:.1f} ({cm[4]:.1f}) |")
        if "v4" in methods:
            rng = np.random.default_rng(0)
            for m in methods:
                if m == "v4":
                    continue
                txt = []
                for met in ("sym_miou", "ari", "miou"):
                    per = defaultdict(list)
                    for r in sub:
                        per[r["cat"]].append(r["v4"][met] - r[m][met])
                    arr = [np.array(v) for v in per.values()]
                    pt = np.mean([a.mean() for a in arr])
                    bs = [np.mean([a[rng.integers(0, len(a), len(a))].mean() for a in arr]) for _ in range(2000)]
                    txt.append(f"{met} {pt:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
                print(f"- v4 - {NAMES.get(m, m)}: " + "; ".join(txt))


if __name__ == "__main__":
    main()
