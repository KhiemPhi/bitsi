"""BITSI v4 development and held-out runs (protocol fixed in RESULTS.md section 20 before running).

Per object: BITSI v3 (frozen), sweep events (the fallback), and the four v4 fits
children in {v3 rule, mdl} x child cuts in {no, yes}, each with its gate statistic u; the gate
threshold is applied afterwards (labels = sweep events when u > gate), so every gate value is
scored from the same fits.
usage: python eval_bitsi4.py OUT_JSONL SPLITS [PENALTIES]
  SPLITS = comma-separated eval_all.py split tags (dev,ooddev | indomain,outdomain,oodrest)
  PENALTIES = comma-separated MDL cut penalties for the v4 fits (default 10)
  GRID = "full" (default: both child rules x both child-cut settings) or "frozen" (only the
         variant selected on dev: children=mdl, cuts=no)
"""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.bitsi3 import BITSI3
from fitpart.bitsi4 import BITSI4, hog_labels
from fitpart.data import load_partnete
from fitpart.metrics import partition_scores
from fitpart.scripts.eval_all import items


def run(args):
    it, pens, grid = args
    warnings.filterwarnings("ignore")
    P, N, C, sem, ins = load_partnete(it["dir"])
    gt = (ins if ins is not None else sem).copy()
    gt[sem < 0] = gt.max() + 1
    out = dict(key=it["key"], cat=it["cat"], split=it["split"])
    for nm, fn in (("v3", lambda: BITSI3().fit(P, N, C)[0]), ("hog", lambda: hog_labels(P, N, C))):
        try:
            out[nm] = partition_scores(gt, fn())
        except Exception as e:
            out[nm] = dict(error=repr(e))
    for pen in pens:
        for ch in (("mdl",) if grid == "frozen" else ("v3", "mdl")):
            for cc in ((False,) if grid == "frozen" else (False, True)):
                nm = f"v4 children={ch} cuts={'yes' if cc else 'no'} pen{pen:g}"
                try:
                    M = BITSI4(children=ch, child_cuts=cc, penalty=pen)
                    lab, root = M.fit(P, N, C)
                    out[nm] = dict(partition_scores(gt, lab), u=M.unexplained, n_children=len(root.children))
                except Exception as e:
                    out[nm] = dict(error=repr(e))
    return out


if __name__ == "__main__":
    out_jsonl, splits = sys.argv[1], set(sys.argv[2].split(","))
    pens = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [10.0]
    grid = sys.argv[4] if len(sys.argv) > 4 else "full"
    done = set()
    if os.path.exists(out_jsonl):
        done = {json.loads(l)["key"] for l in open(out_jsonl) if l.strip()}
    todo = [it for it in items("partnete") if it["split"] in splits and it["key"] not in done]
    print(len(done), "done,", len(todo), "to run", flush=True)
    with Pool(int(os.environ.get("WORKERS", 14))) as p, open(out_jsonl, "a") as f:
        for i, r in enumerate(p.imap_unordered(run, [(it, pens, grid) for it in todo], chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 25 == 0:
                print(i, "/", len(todo), flush=True)
    print("finished", flush=True)
