"""Section-32 dev grid: root-axis rules for the robot setting (gravity known) on eye-in-hand clouds of
dev + tabletop ooddev objects (orbit views, K = 3 and 6), full-object scores.
usage: python dev_axis.py OUT.jsonl SPLITS (dev,ooddev | indomain,outdomain)"""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from fitpart import eyeinhand as E
from fitpart.bitsi4 import BITSI4
from fitpart.scales import CATEGORY_SIZE_M
from fitpart.scripts.eval_all import items
from fitpart.scripts.eval_eyeinhand import load, score, seed_of

UP = (0, 0, 1.0)
VARIANTS = {"v4 (no gravity)": dict(), "gravity as 3rd candidate": dict(up=UP),
            "prefer vertical d0.02": dict(up=UP, up_prior=0.02), "prefer vertical d0.05": dict(up=UP, up_prior=0.05),
            "prefer vertical d0.10": dict(up=UP, up_prior=0.10), "always vertical": dict(up=UP, up_prior=np.inf),
            "cone15 d0.05": dict(up=UP, up_prior=0.05, up_cone=15.0), "cone15 always": dict(up=UP, up_prior=np.inf, up_cone=15.0)}
VIEWS = [(180.0, 35.0), (0.0, 35.0), (90.0, 35.0), (270.0, 35.0), (135.0, 35.0), (315.0, 35.0)]


def run(it):
    warnings.filterwarnings("ignore")
    W, Nw, C, gt = load(it)
    h = E.spacing(W)
    centre, diag = E.object_frame(W)
    dist = E.view_distance(diag)
    F = E.Fused(h, seed=1)
    out = dict(key=it["key"], cat=it["cat"], split=it["split"])
    for k, v in enumerate(VIEWS, 1):
        cam, Rc = E.camera_pose(centre, dist, *v)
        idx, meas = E.observe(W, Nw, cam, Rc, h, np.random.default_rng(seed_of(it["key"], *v)))
        F.add(idx, meas, cam)
        if k in (3, 6):
            fi, pts, n = F.cloud()
            col = C[fi] if C is not None else None
            for nm, kw in VARIANTS.items():
                try:
                    out[f"{nm}@K{k}"] = score(BITSI4(**kw).fit(pts, n, col)[0], fi, pts, gt, W, None)[1]
                except Exception as e:
                    out[f"{nm}@K{k}"] = dict(error=repr(e))
    return out


if __name__ == "__main__":
    splits = set(sys.argv[2].split(","))
    its = [it for it in items("partnete") if it["split"] in splits and (CATEGORY_SIZE_M.get(it["cat"]) or 9) <= 0.6]
    with Pool(int(os.environ.get("WORKERS", 14))) as p, open(sys.argv[1], "w") as f:
        for r in p.imap_unordered(run, its):
            f.write(json.dumps(r) + "\n")
            f.flush()
    print("finished", len(its))
