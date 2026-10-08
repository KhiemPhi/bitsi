"""Paired development-only v4 / observed-radius v5 experiment.

Fixed first N objects per development category, full scans and orbit K=3,6.
No held-out split is accepted. Output is exclusive-create to avoid stale results.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from fitpart import eyeinhand as E
from fitpart.bitsi4 import BITSI4
from fitpart.bitsi5 import BITSI5
from fitpart.metrics import partition_scores
from fitpart.scales import CATEGORY_SIZE_M
from fitpart.scripts.eval_all import items
from fitpart.scripts.eval_eyeinhand import load, score, seed_of
from fitpart.scripts.dev_bitsi5 import VIEWS


def run(task):
    it, pooling = task
    W, N, C, gt = load(it)
    row = dict(key=it["key"], cat=it["cat"], split=it["split"], pooling=pooling, evaluations={})

    def evaluate(tag, P, normals, colors, ids=None):
        result = {}
        for name, model in [("v4", BITSI4(up=(0, 0, 1))),
                            ("v5_partial", BITSI5(up=(0, 0, 1), axis_rule="residual",
                                                   spine_coding="free", partial_cuts=True,
                                                   partial_bin_pooling=pooling))]:
            start = time.monotonic()
            labels, root = model.fit(P, normals, colors)
            obs, full = (partition_scores(gt, labels), partition_scores(gt, labels)) if ids is None else score(labels, ids, P, gt, W, None)
            result[name] = dict(observed=obs, full=full, children=len(root.children),
                                cuts=len(root.cuts), seconds=time.monotonic()-start)
        row["evaluations"][tag] = result

    evaluate("full", W, N, C)
    h = E.spacing(W)
    centre, diag = E.object_frame(W)
    F = E.Fused(h, seed=1)
    for k, view in enumerate(VIEWS, 1):
        cam, Rc = E.camera_pose(centre, E.view_distance(diag), *view)
        ids, measured = E.observe(W, N, cam, Rc, h, np.random.default_rng(seed_of(it["key"], *view)))
        F.add(ids, measured, cam)
        if k in (3, 6):
            ids, P, normals = F.cloud()
            evaluate(f"K{k}", P, normals, None if C is None else C[ids], ids)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pooling", choices=["mean", "sum"], default="mean")
    parser.add_argument("--splits", choices=["dev", "ooddev", "dev,ooddev"], default="dev,ooddev")
    args = parser.parse_args()
    counts, todo = Counter(), []
    for it in items("partnete"):
        if it["split"] not in args.splits.split(",") or (CATEGORY_SIZE_M.get(it["cat"]) or 9) > .6:
            continue
        if counts[it["cat"]] < args.per_category:
            todo.append(it)
            counts[it["cat"]] += 1
    with open(args.output, "x") as f, Pool(args.workers) as pool:
        print(f"{len(todo)} development objects", flush=True)
        for i, row in enumerate(pool.imap(run, [(it, args.pooling) for it in todo]), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{i}/{len(todo)} {row['key']}", flush=True)
