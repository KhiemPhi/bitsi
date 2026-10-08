"""Partition scores that see over-segmentation (metrics.partition_scores: GT-averaged mIoU,
symmetric mIoU, adjusted Rand index, segment count) for every method on a list of objects.

Run once per environment and merge by object: methods "v2", "v1", "hog", "whole" (sam3 env) and
"v0" (pointcloud env, the existing BITSI pipeline).
usage: python eval_partition.py OUT_JSON METHODS SPLIT
  SPLIT = dev | indomain | outdomain   (dev: first 20 of the 9 symmetric categories; indomain:
          objects 21+ of those; outdomain: first 4 of the other 36 categories)
"""
import contextlib, io, json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.data import PARTNETE_ROOT, load_partnete
from fitpart.metrics import partition_scores

SYM = ["Bottle", "Bucket", "Kettle", "KitchenPot", "Lamp", "Globe", "Pen", "TrashCan", "Dispenser"]


def objects(split):
    base = os.path.join(PARTNETE_ROOT, "data", "test")
    out = []
    for c in sorted(os.listdir(base)):
        if (c in SYM) != (split in ("dev", "indomain")):
            continue
        ds = sorted((x for x in os.listdir(os.path.join(base, c)) if os.path.exists(os.path.join(base, c, x, "label.npy"))), key=int)
        ds = ds[:20] if split == "dev" else ds[20:] if split == "indomain" else ds[:4]
        out += [os.path.join(base, c, x) for x in ds]
    return out


def labels(method, P, N, C):
    if method == "v2":
        from fitpart.bitsi2 import BITSI2
        return BITSI2().fit(P, N, C)[0]
    if method == "v1":
        from fitpart.primitive import primitive_decomposition
        return primitive_decomposition(P, N, C, axial="hog_pca")[0]
    if method == "hog":
        from fitpart.profiles import pca_axes
        from fitpart.sweep_events import events_along
        ev, prof = events_along(P, pca_axes(P)[0], N, C, ("hist",))
        return np.searchsorted(np.sort([c.t for c in ev]), prof["coord"])
    if method == "whole":
        return np.zeros(len(P), int)
    if method == "v0":
        from fitpart.baselines.bitsi_v0 import segment
        with contextlib.redirect_stdout(io.StringIO()):
            return segment(P, N)
    raise ValueError(method)


def run(args):
    d, methods = args
    P, N, C, sem, ins = load_partnete(d)
    gt = (ins if ins is not None else sem).copy()
    gt[sem < 0] = gt.max() + 1
    out = dict(dir=d, cat=os.path.basename(os.path.dirname(d)))
    for m in methods:
        try:
            out[m] = partition_scores(gt, labels(m, P, N, C))
        except Exception as e:
            out[m] = dict(error=repr(e))
    return out


if __name__ == "__main__":
    out_json, methods, split = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    workers = int(os.environ.get("WORKERS", 12))
    with Pool(workers) as p:
        rs = p.map(run, [(d, methods) for d in objects(split)], chunksize=1)
    json.dump(rs, open(out_json, "w"))
    print(len(rs), "objects;", sum("error" in r[m] for r in rs for m in methods), "errors")
