"""BITSI v3 variants (channel groups x MDL cut penalty) with partition scores.

Selection rule (fixed before running): on the dev split, maximise mean(symmetric mIoU, ARI) subject
to GT-averaged mIoU >= BITSI v2's - 0.03; the selected variant is then run once on the held-out
splits (in-domain, out-of-domain).
usage: python eval_bitsi3.py OUT_JSON SPLIT [VARIANTS (semicolon-separated names)]
"""
import json, os, sys, warnings
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.bitsi3 import BITSI3
from fitpart.data import load_partnete
from fitpart.metrics import partition_scores
from fitpart.scripts.eval_partition import objects

VARIANTS = {"v3 (frozen defaults)": dict()}
for ch in (("section",), ("topo",), ("section", "topo"), ("section", "topo", "hist")):
    for pen in (1.0, 3.0, 10.0, 30.0):
        VARIANTS[f"v3 {'+'.join(ch)} pen{pen:g}"] = dict(channels=ch, penalty=pen)


def run(args):
    d, names = args
    warnings.filterwarnings("ignore")
    P, N, C, sem, ins = load_partnete(d)
    gt = (ins if ins is not None else sem).copy()
    gt[sem < 0] = gt.max() + 1
    out = dict(dir=d, cat=os.path.basename(os.path.dirname(d)))
    for nm in names:
        try:
            out[nm] = partition_scores(gt, BITSI3(**VARIANTS[nm]).fit(P, N, C)[0])
        except Exception as e:
            out[nm] = dict(error=repr(e))
    return out


if __name__ == "__main__":
    out_json, split = sys.argv[1], sys.argv[2]
    names = sys.argv[3].split(";") if len(sys.argv) > 3 else list(VARIANTS)
    with Pool(int(os.environ.get("WORKERS", 12))) as p:
        rs = p.map(run, [(d, names) for d in objects(split)], chunksize=1)
    json.dump(rs, open(out_json, "w"))
    print(len(rs), "objects;", sum("error" in r[n] for r in rs for n in names), "errors")
