"""Score cached baseline labels on the saved eye-in-hand clouds, next to BITSI v4 from the run file.

Per (object, strategy, K): seen and full-object partition scores, exactly as eval_eyeinhand.score.
usage: python score_eih_baselines.py ROBOT RUN.jsonl METHOD[,METHOD]
"""
import glob, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.scripts.eval_eyeinhand import CLOUDS, load, objects, score


def main():
    robot, run_file, methods = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    runs = {json.loads(l)["key"]: json.loads(l) for l in open(run_file) if l.strip()}
    its = {it["key"]: it for it in objects()}
    res = defaultdict(lambda: defaultdict(list))   # (split, strat, K) -> method -> [(cat, sym, ari)]
    for key, r in runs.items():
        if "error" in r:
            continue
        it = its[key]
        W, Nw, C, gt = load(it)
        for strat in ("orbit", "nbv_cov", "nbv_bitsi"):
            for e in r[strat]["evals"]:
                f = os.path.join(CLOUDS, robot, strat, f"{key.replace('/', '__')}__K{e['k']}.npz")
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                row = {"v4": e.get("v4_full")}
                for m in methods:
                    lf = f[:-4] + f".{m}.npy"
                    if os.path.exists(lf):
                        row[m] = score(np.load(lf).astype(int), z["idx"], z["pts"], gt, W, None)[1]
                if all(row.get(m) for m in ["v4"] + methods):
                    for m, s in row.items():
                        res[(it["split"], strat, e["k"])][m].append((it["cat"], s["sym_miou"], s["ari"]))
    for (sp, strat, k), d in sorted(res.items()):
        cells = []
        for m in ["v4"] + methods:
            per = defaultdict(list)
            for c, s, a in d[m]:
                per[c].append((s, a))
            cm = np.mean([np.mean(v, 0) for v in per.values()], 0)
            cells.append(f"{m} {cm[0]:.3f}/{cm[1]:.3f}")
        print(f"{robot} {sp} {strat} K={k} (n={len(d['v4'])}): full-object sym/ARI " + ", ".join(cells))


if __name__ == "__main__":
    main()
