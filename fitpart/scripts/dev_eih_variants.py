"""Development check on PartNet-E DEV objects (eye-in-hand clouds, orbit views K = 3 and 6):
BITSI v4 vs v4 with the gravity-axis candidate (up = table normal). Full-object scores.
usage: python dev_eih_variants.py OUT.jsonl"""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from fitpart import eyeinhand as E
from fitpart.bitsi4 import BITSI4
from fitpart.scripts.eval_all import items
from fitpart.scripts.eval_eyeinhand import load, score, seed_of

VIEWS = [(180.0, 35.0), (0.0, 35.0), (90.0, 35.0), (270.0, 35.0), (135.0, 35.0), (315.0, 35.0)]


def run(it):
    warnings.filterwarnings("ignore")
    W, Nw, C, gt = load(it)
    h = E.spacing(W)
    centre, diag = E.object_frame(W)
    dist = E.view_distance(diag)
    F = E.Fused(h, seed=1)
    out = dict(key=it["key"], cat=it["cat"])
    for k, v in enumerate(VIEWS, 1):
        cam, Rc = E.camera_pose(centre, dist, *v)
        idx, meas = E.observe(W, Nw, cam, Rc, h, np.random.default_rng(seed_of(it["key"], *v)))
        F.add(idx, meas, cam)
        if k in (3, 6):
            fi, pts, n = F.cloud()
            col = C[fi] if C is not None else None
            for nm, up in (("v4", None), ("v4_up", (0, 0, 1.0))):
                try:
                    out[f"{nm}_K{k}"] = score(BITSI4(up=up).fit(pts, n, col)[0], fi, pts, gt, W, None)[1]
                except Exception as e:
                    out[f"{nm}_K{k}"] = dict(error=repr(e))
    return out


if __name__ == "__main__":
    its = [it for it in items("partnete") if it["split"] == "dev"]
    with Pool(int(os.environ.get("WORKERS", 8))) as p, open(sys.argv[1], "w") as f:
        for r in p.imap_unordered(run, its):
            f.write(json.dumps(r) + "\n")
            f.flush()
    print("finished")
