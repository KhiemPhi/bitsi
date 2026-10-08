"""Section-24 dev grid: BITSI v5 variants on full clouds and on eye-in-hand clouds.

Per dev / ooddev object: every variant on the full cloud (no gravity axis), and, for tabletop
objects, on the orbit-view fused clouds at K = 3 and 6 (gravity axis offered, robot setting),
scored on the full object.
usage: python dev_bitsi5.py OUT.jsonl SPLITS            (dev,ooddev | indomain,outdomain,oodrest)
       VARIANTS env: comma-separated variant names to restrict the grid (default: all)
"""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart import eyeinhand as E
from fitpart.bitsi5 import BITSI5
from fitpart.data import load_partnete
from fitpart.metrics import partition_scores
from fitpart.scales import CATEGORY_SIZE_M, metric_scale
from fitpart.scripts.eval_all import items
from fitpart.scripts.eval_eyeinhand import load, score, seed_of

VIEWS = [(180.0, 35.0), (0.0, 35.0), (90.0, 35.0), (270.0, 35.0), (135.0, 35.0), (315.0, 35.0)]
GRID = {}
for spc in ("free", "cheb"):
    GRID[f"gripper/{spc}"] = dict(axis_rule="gripper", spine_coding=spc, gripper_opening=0.08)
    GRID[f"residual/{spc}"] = dict(axis_rule="residual", spine_coding=spc)
    for tau in (0.15, 0.20, 0.25, 0.30):
        GRID[f"normals{tau:.2f}/{spc}"] = dict(axis_rule="normals", axis_tau=tau, spine_coding=spc)


def fit(kw, P, N, C, up=None, scale=1.0):
    return BITSI5(up=up, scale=scale, **kw).fit(P, N, C)[0]


def run(args):
    it, names = args
    warnings.filterwarnings("ignore")
    P, N, C, sem, ins = load_partnete(it["dir"])
    gt = (ins if ins is not None else sem).copy()
    gt[sem < 0] = gt.max() + 1
    out = dict(key=it["key"], cat=it["cat"], split=it["split"], full={}, eih={})
    for nm in names:
        try:
            out["full"][nm] = partition_scores(gt, fit(GRID[nm], P, N, C, scale=metric_scale(P, it["cat"]) or 1.0))
        except Exception as e:
            out["full"][nm] = dict(error=repr(e))
    if (CATEGORY_SIZE_M.get(it["cat"]) or 9) <= 0.6:
        W, Nw, Cw, gtw = load(it)
        h = E.spacing(W)
        centre, diag = E.object_frame(W)
        dist = E.view_distance(diag)
        F = E.Fused(h, seed=1)
        for k, v in enumerate(VIEWS, 1):
            cam, Rc = E.camera_pose(centre, dist, *v)
            idx, meas = E.observe(W, Nw, cam, Rc, h, np.random.default_rng(seed_of(it["key"], *v)))
            F.add(idx, meas, cam)
            if k in (3, 6):
                fi, pts, n = F.cloud()
                col = Cw[fi] if Cw is not None else None
                for nm in names:
                    try:
                        out["eih"][f"{nm}@K{k}"] = score(fit(GRID[nm], pts, n, col, up=(0, 0, 1.0)), fi, pts, gtw, W, None)[1]
                    except Exception as e:
                        out["eih"][f"{nm}@K{k}"] = dict(error=repr(e))
    return out


if __name__ == "__main__":
    out_jsonl, splits = sys.argv[1], set(sys.argv[2].split(","))
    names = os.environ["VARIANTS"].split(",") if os.environ.get("VARIANTS") else list(GRID)
    done = set()
    if os.path.exists(out_jsonl):
        done = {json.loads(l)["key"] for l in open(out_jsonl) if l.strip()}
    todo = [(it, names) for it in items("partnete") if it["split"] in splits and it["key"] not in done]
    print(len(todo), "to run", flush=True)
    with Pool(int(os.environ.get("WORKERS", 14))) as p, open(out_jsonl, "a") as f:
        for i, r in enumerate(p.imap_unordered(run, todo, chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 25 == 0:
                print(i, "/", len(todo), flush=True)
    print("finished", flush=True)
