"""Partition scores for every method on every local object of a dataset (resumable).

One JSON line per object in OUT_JSONL; objects already there are skipped, so the same command
continues an interrupted run, and runs in different envs (v0 needs `pointcloud`) append to their
own files to be merged by object later (`scripts/summarize_all.py`).
usage: [SPLITS=indomain,outdomain] [MAX_PER_CAT=8] python eval_all.py OUT_JSONL METHODS DATASET
  METHODS = comma-separated: v4, v3, v2, v1, hog, whole (sam3 env); v0, coacd, superquadric (pointcloud env)
  DATASET = partnete    (every object under ~/data/PartNetE/data/test, 20k-point subsample,
                          tagged with its split: dev / indomain / outdomain / ooddev / oodrest)
          | shapenetpart (the official test list, 2.5k points, no colour)
"""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.data import PARTNETE_ROOT, load_partnete, load_shapenetpart, shapenetpart_categories, shapenetpart_split
from fitpart.metrics import partition_scores
from fitpart.scripts.eval_partition import SYM, labels


def split_of(cat, rank):
    if cat in SYM:
        return "dev" if rank < 20 else "indomain"
    return "outdomain" if rank < 4 else "ooddev" if rank < 8 else "oodrest"


def items(dataset):
    if dataset == "shapenetpart":
        cats = shapenetpart_categories()
        return [dict(key=f"{s}/{o}", cat=cats[s], split="test", sid=s, oid=o) for s, o in shapenetpart_split("test")]
    base = os.path.join(PARTNETE_ROOT, "data", "test")
    out = []
    for c in sorted(os.listdir(base)):
        ds = sorted((x for x in os.listdir(os.path.join(base, c)) if os.path.exists(os.path.join(base, c, x, "label.npy"))), key=int)
        out += [dict(key=f"{c}/{x}", cat=c, split=split_of(c, i), dir=os.path.join(base, c, x)) for i, x in enumerate(ds)]
    return out


CACHED = ("find3d", "partfield", "partfield_v4k")  # GPU baselines (own envs), labels cached per object


def method_labels(m, P, N, C, it=None):
    if m in CACHED:
        ds = "partnete" if "dir" in it else "shapenetpart"
        lab = np.load(os.path.join(os.path.expanduser("~/.cache/fitpart_baselines"), m, ds, it["key"].replace("/", "__") + ".npy"))
        assert len(lab) == len(P), (m, it["key"], len(lab), len(P))
        return lab
    if m in ("coacd", "superquadric"):
        from fitpart.baselines import geometric
        return getattr(geometric, f"{m}_segment")(P, N)
    if m == "v4":
        from fitpart.bitsi4 import BITSI4
        return BITSI4().fit(P, N, C)[0]
    if m == "v3":
        from fitpart.bitsi3 import BITSI3
        return BITSI3().fit(P, N, C)[0]
    return labels(m, P, N, C)


def run(args):
    it, methods = args
    warnings.filterwarnings("ignore")
    if "dir" in it:
        P, N, C, sem, ins = load_partnete(it["dir"])
        gt = (ins if ins is not None else sem).copy()
        gt[sem < 0] = gt.max() + 1
    else:
        P, N, gt = load_shapenetpart(it["sid"], it["oid"])
        C = None
    out = dict(key=it["key"], cat=it["cat"], split=it["split"], n_points=len(P))
    for m in methods:
        try:
            out[m] = partition_scores(gt, method_labels(m, P, N, C, it))
        except Exception as e:
            out[m] = dict(error=repr(e))
    return out


if __name__ == "__main__":
    out_jsonl, methods, dataset = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    done = set()
    if os.path.exists(out_jsonl):
        done = {json.loads(l)["key"] for l in open(out_jsonl) if l.strip()}
    splits = os.environ.get("SPLITS")  # optional comma-separated split filter
    per_cat = int(os.environ.get("MAX_PER_CAT", 0))  # optional: first N objects of each category
    todo, seen = [], {}
    for it in items(dataset):
        if splits and it["split"] not in splits.split(","):
            continue
        seen[it["cat"]] = seen.get(it["cat"], 0) + 1          # counted within the selected splits
        if per_cat and seen[it["cat"]] > per_cat:
            continue
        if it["key"] not in done:
            todo.append(it)
    print(len(done), "done,", len(todo), "to run", flush=True)
    with Pool(int(os.environ.get("WORKERS", 14))) as p, open(out_jsonl, "a") as f:
        for i, r in enumerate(p.imap_unordered(run, [(it, methods) for it in todo], chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 50 == 0:
                print(i, "/", len(todo), flush=True)
    print("finished", flush=True)
