"""Tables for the eye-in-hand experiment (RESULTS.md section 21).
usage: python summarize_eyeinhand.py OUT.jsonl [OUT2.jsonl ...]  (one file per robot)"""
import json, sys
from collections import defaultdict
import numpy as np

STRATS = ("orbit", "nbv_cov", "nbv_bitsi")
SPLITS = (("in-domain", {"indomain"}), ("out-of-domain", {"outdomain"}))


def cmean(vals):
    per = defaultdict(list)
    for c, v in vals:
        per[c].append(v)
    return float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")


def get(r, strat, k, m, key):
    for e in r[strat]["evals"]:
        if e["k"] == k:
            if key == "coverage":
                return e["coverage"]
            s = e.get(m)
            return None if s is None or "error" in s else s[key]
    return None


def boot(rows, a, b, k, m, key, n=2000, seed=0):
    per = defaultdict(list)
    for r in rows:
        x, y = get(r, a, k, m, key), get(r, b, k, m, key)
        if x is not None and y is not None:
            per[r["cat"]].append(x - y)
    arr = [np.array(v) for v in per.values()]
    rng = np.random.default_rng(seed)
    pt = np.mean([v.mean() for v in arr])
    bs = [np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in arr]) for _ in range(n)]
    return f"{pt:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "error" not in r]
    robot = rows[0]["robot"] if rows else "?"
    for title, sp in SPLITS:
        sub = [r for r in rows if r["split"] in sp]
        print(f"\n### {robot}, {title}: {len(sub)} objects, mean reachable views {np.mean([r['n_reach'] for r in sub]):.0f}/96")
        print("| strategy | K | coverage | v4 sym (seen) | v4 ARI (seen) | v4 sym (full) | v4 ARI (full) | travel (m) |")
        print("|---|---|---|---|---|---|---|---|")
        for s in STRATS:
            for k in (1, 2, 3, 4, 6, 8):
                cells = [cmean([(r["cat"], get(r, s, k, "v4", "coverage")) for r in sub if get(r, s, k, "v4", "coverage") is not None])]
                for m, key in (("v4", "sym_miou"), ("v4", "ari"), ("v4_full", "sym_miou"), ("v4_full", "ari")):
                    cells.append(cmean([(r["cat"], get(r, s, k, m, key)) for r in sub if get(r, s, k, m, key) is not None]))
                tr = np.mean([r[s]["travel"] for r in sub])
                print(f"| {s} | {k} | " + " | ".join(f"{c:.3f}" for c in cells) + f" | {tr:.2f} (all {s} views) |" if k == 8 else
                      f"| {s} | {k} | " + " | ".join(f"{c:.3f}" for c in cells) + " | |")
        al = [(r["cat"], r["all"]["evals"][0]) for r in sub]
        print(f"| all reachable | {np.mean([r['n_reach'] for r in sub]):.0f} | {cmean([(c, e['coverage']) for c, e in al]):.3f} | "
              f"{cmean([(c, e['v4']['sym_miou']) for c, e in al if 'sym_miou' in e['v4']]):.3f} | {cmean([(c, e['v4']['ari']) for c, e in al if 'ari' in e['v4']]):.3f} | "
              f"{cmean([(c, e['v4_full']['sym_miou']) for c, e in al if 'v4_full' in e]):.3f} | {cmean([(c, e['v4_full']['ari']) for c, e in al if 'v4_full' in e]):.3f} | |")
        print("\nPrimary (K = 3, full object):")
        for other in ("nbv_cov", "orbit"):
            print(f"- nbv_bitsi - {other}: sym {boot(sub, 'nbv_bitsi', other, 3, 'v4_full', 'sym_miou')}; ARI {boot(sub, 'nbv_bitsi', other, 3, 'v4_full', 'ari')}")
        print("\nSegmenters on the same nbv_bitsi clouds (full-object sym / ARI):")
        for k in (1, 3, 8):
            txt = []
            for m in ("v4", "v3", "hog"):
                txt.append(f"{m} {cmean([(r['cat'], get(r, 'nbv_bitsi', k, m + '_full', 'sym_miou')) for r in sub if get(r, 'nbv_bitsi', k, m + '_full', 'sym_miou') is not None]):.3f}"
                           f"/{cmean([(r['cat'], get(r, 'nbv_bitsi', k, m + '_full', 'ari')) for r in sub if get(r, 'nbv_bitsi', k, m + '_full', 'ari') is not None]):.3f}")
            print(f"- K={k}: " + ", ".join(txt))
        st = [r["nbv_bitsi"].get("stop_k") for r in sub]
        print(f"- adaptive stop reached within 8 views on {np.mean([s is not None for s in st]):.2f} of objects, median K {np.median([s for s in st if s]) if any(st) else float('nan')}")
