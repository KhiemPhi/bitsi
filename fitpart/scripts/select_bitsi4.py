"""Apply the section-20 selection rule to eval_bitsi4.py output, or score a frozen variant.

usage: python select_bitsi4.py DEV_JSONL                       (grid + selected variant)
       python select_bitsi4.py HELDOUT_JSONL VARIANT GATE      (held-out tables + bootstrap CIs)
"""
import json, sys
from collections import defaultdict
import numpy as np

GATES = [np.inf, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
MET = ("miou", "sym_miou", "ari", "n_pred")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def scores(r, variant, gate):
    """Per-object scores of a method name; v4 variants with the gate applied."""
    if variant in ("v3", "hog"):
        return r.get(variant)
    v = r.get(variant)
    if v is None or "error" in v:
        return None
    return r["hog"] if v["u"] > gate and "error" not in r["hog"] else v


def table(rows, variant, gate, splits):
    by = defaultdict(list)
    for r in rows:
        if r["split"] in splits:
            s = scores(r, variant, gate)
            if s is not None and "error" not in s:
                by[r["cat"]].append([s[m] for m in MET])
    cat = {c: np.mean(v, 0) for c, v in by.items()}
    return np.mean(list(cat.values()), 0), cat, by


def boot_diff(rows, a, ga, b, gb, splits, metric, n=3000, seed=0):
    """Category-mean difference a - b; objects resampled within category."""
    i = MET.index(metric)
    by = defaultdict(list)
    for r in rows:
        if r["split"] in splits:
            sa, sb = scores(r, a, ga), scores(r, b, gb)
            if sa and sb and "error" not in sa and "error" not in sb:
                by[r["cat"]].append(sa[metric] - sb[metric])
    rng = np.random.default_rng(seed)
    cats = [np.array(v) for v in by.values()]
    point = np.mean([c.mean() for c in cats])
    bs = [np.mean([c[rng.integers(0, len(c), len(c))].mean() for c in cats]) for _ in range(n)]
    return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    rows = load(sys.argv[1])
    variants = sorted({k for r in rows for k in r if k.startswith("v4 ")})
    if len(sys.argv) == 2:
        v3_dev = table(rows, "v3", None, {"dev"})[0]
        print(f"{len(rows)} objects. v3 dev: miou {v3_dev[0]:.3f} sym {v3_dev[1]:.3f} ari {v3_dev[2]:.3f}")
        print("| variant | gate | dev mIoU | dev sym | dev ARI | ooddev sym | ooddev ARI | objective | fallback rate |")
        print("|---|---|---|---|---|---|---|---|---|")
        best = None
        for name, gates in [("v3", [None]), ("hog", [None])] + [(v, GATES) for v in variants]:
            for g in gates:
                d = table(rows, name, g, {"dev"})[0]
                o = table(rows, name, g, {"ooddev"})[0]
                both = table(rows, name, g, {"dev", "ooddev"})[0]
                obj = 0.5 * (both[1] + both[2])
                fb = np.mean([r[name]["u"] > g for r in rows if name in r and "u" in r[name]]) if g is not None else float("nan")
                ok = d[0] >= v3_dev[0] - 0.03
                print(f"| {name} | {g} | {d[0]:.3f} | {d[1]:.3f} | {d[2]:.3f} | {o[1]:.3f} | {o[2]:.3f} | {obj:.3f}{'' if ok else ' (fails mIoU)'} | {fb:.2f} |")
                if name.startswith("v4") and ok and (best is None or obj > best[0]):
                    best = (obj, name, g)
        print("SELECTED:", best)
        return
    variant, gate = sys.argv[2], float(sys.argv[3])
    for splits in ({"indomain"}, {"outdomain"}, {"oodrest"}, {"outdomain", "oodrest"}):
        print("\n###", "+".join(sorted(splits)))
        print("| method | mIoU | sym mIoU | ARI | segments |\n|---|---|---|---|---|")
        for name, g in ((variant, gate), ("v3", None), ("hog", None)):
            m = table(rows, name, g, splits)[0]
            print(f"| {name if name != variant else 'BITSI v4 (frozen)'} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.1f} |")
        for other in ("v3", "hog"):
            for metric in ("sym_miou", "ari", "miou"):
                p, lo, hi = boot_diff(rows, variant, gate, other, None, splits, metric)
                print(f"v4 - {other} {metric}: {p:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
