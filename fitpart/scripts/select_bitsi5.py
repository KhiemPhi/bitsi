"""Apply the section-24 selection rule to dev_bitsi5.py output.
usage: python select_bitsi5.py DEV.jsonl [DEV2.jsonl ...]"""
import json, sys
from collections import defaultdict
import numpy as np


def cmean(pairs):
    per = defaultdict(list)
    for c, v in pairs:
        per[c].append(v)
    return float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")


rows = {}
for f in sys.argv[1:]:  # several files are merged by object (variants run separately)
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            cur = rows.setdefault(r["key"], dict(key=r["key"], cat=r["cat"], split=r["split"], full={}, eih={}))
            cur["full"].update(r["full"])
            cur["eih"].update(r["eih"])
rows = list(rows.values())
names = sorted({k for r in rows for k in r["full"]})
print(f"{len(rows)} objects, {sum(bool(r['eih']) for r in rows)} with eye-in-hand clouds")
print("| variant | full sym | full ARI | full obj | eih K3 sym/ARI | eih K6 sym/ARI | eih obj | objective |")
print("|---|---|---|---|---|---|---|---|")
res = {}
for nm in names:
    fs = cmean([(r["cat"], r["full"][nm]["sym_miou"]) for r in rows if "sym_miou" in r["full"].get(nm, {})])
    fa = cmean([(r["cat"], r["full"][nm]["ari"]) for r in rows if "ari" in r["full"].get(nm, {})])
    e = {}
    for k in (3, 6):
        key = f"{nm}@K{k}"
        e[k] = (cmean([(r["cat"], r["eih"][key]["sym_miou"]) for r in rows if "sym_miou" in r["eih"].get(key, {})]),
                cmean([(r["cat"], r["eih"][key]["ari"]) for r in rows if "ari" in r["eih"].get(key, {})]))
    fo = 0.5 * (fs + fa)
    eo = 0.25 * (e[3][0] + e[3][1] + e[6][0] + e[6][1])
    res[nm] = (fo, eo, 0.5 * (fo + eo))
    print(f"| {nm} | {fs:.3f} | {fa:.3f} | {fo:.3f} | {e[3][0]:.3f}/{e[3][1]:.3f} | {e[6][0]:.3f}/{e[6][1]:.3f} | {eo:.3f} | {res[nm][2]:.3f} |")
base = res["residual/free"][0]
ok = {nm: v for nm, v in res.items() if v[0] >= base - 0.005}
best = max(ok, key=lambda nm: ok[nm][2])
print("v4 (residual/free) full objective", round(base, 4), "| SELECTED:", best, [round(x, 4) for x in res[best]])
