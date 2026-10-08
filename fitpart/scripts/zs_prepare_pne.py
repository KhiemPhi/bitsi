"""Per object: SAM 3 node scores from coloured masks (.sam3c.pkl) for the candidate nodes of
<obj>.nodefeat.npz (same hierarchy: assemble() is deterministic). Writes <obj>.samc.npz.

usage: python zs_prepare_pne.py CACHE_DIR META_JSON [workers]
"""
import glob, json, os, pickle, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

META = None


def run(f):
    from fitpart.pipeline import Config, FitPart
    from fitpart.sam3_select import union_score
    base = f[:-len(".nodefeat.npz")]
    out = base + ".samc.npz"
    if os.path.exists(out):
        return
    d = np.load(f, allow_pickle=True)
    L = pickle.load(open(base + ".pkl", "rb"))
    S3 = pickle.load(open(base + ".sam3c.pkl", "rb"))
    res = FitPart(Config()).assemble(L)
    cand = d["cand"]
    samc = np.array([union_score(res.hier, res.sp, S3[nm])[cand] for nm in META[L["cat"]]], np.float32)
    np.savez_compressed(out, samc=samc)


if __name__ == "__main__":
    META = json.load(open(sys.argv[2]))
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.nodefeat.npz")))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 10) as p:
        p.map(run, files, chunksize=1)
    print(len(files), "done")
