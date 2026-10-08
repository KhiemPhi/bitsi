"""Table 2's "exists but cannot be grasped" report (Sec. 3 / 4.5).

For every part name present in an object, the GT part is labeled ungraspable if it fails
the box test (smallest min-volume-OBB extent > G, metric scale from category priors).
FitPart selects among feasible nodes with SAM 3 scores and a relevance floor r_min; if
no feasible node reaches r_min while some node does, it reports "infeasible".

Reported per r_min: on ungraspable GT parts, the rate of correct "infeasible" reports
(vs returning some feasible fragment); on graspable GT parts, the false-alarm rate.

usage: python eval_infeasible.py CACHE_DIR META_JSON [workers]
"""
import glob, json, os, pickle, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.feasibility import box_test, feasible
from fitpart.pipeline import Config, FitPart
from fitpart.sam3_select import node_scores
from fitpart.scales import metric_scale
from fitpart.selection import node_membership, select

R_MINS = (0.2, 0.3, 0.4, 0.5)
ALPHAS = (0.5, 0.7, 0.9)
META = None


def run(f):
    L = pickle.load(open(f, "rb"))
    s3p = f[:-4] + ".sam3.pkl"
    if not os.path.exists(s3p):
        return []
    L["sam3"] = pickle.load(open(s3p, "rb"))
    P, N = L["points"].astype(float), L["normals"].astype(float)
    scale = metric_scale(P, L["cat"])
    if scale is None:
        return []
    Pm = P * scale
    fp = FitPart(Config())
    res = fp.assemble(L)
    sp, H, sem = res.sp, res.hier, L["sem"]
    _, sizes = node_membership(H, np.bincount(sp).astype(float))
    names = META[L["cat"]]
    out = []
    feas = {}
    for p in np.unique(sem[sem >= 0]):
        g = sem == p
        if g.sum() < 20:
            continue
        gt_ok = box_test(Pm[g], fp.cfg.gripper_width)
        r = node_scores(H, sp, L["sam3"][names[p]])
        top = np.argsort(-r)[:40]
        for h in top:
            if h not in feas:
                m = np.isin(sp, H.members[h])
                feas[h] = feasible(Pm[m], N[m], fp.cfg.gripper_width)
        fmask = np.zeros(len(r), bool)
        fmask[[h for h in top if feas[h]]] = True
        rec = dict(cat=L["cat"], part=names[p], gt_graspable=bool(gt_ok))
        for rmin in R_MINS:
            _, status = select(r, sizes, fmask, fp.cfg.eps_r, rmin, min_size=0.01 * len(sp))
            rec[rmin] = status
        # relative rule: the most relevant node is infeasible and the best feasible one
        # scores below alpha times it
        big_enough = sizes >= 0.01 * len(sp)
        rr = np.where(big_enough, r, -np.inf)
        h_all = int(np.argmax(rr))
        if h_all not in feas:
            m = np.isin(sp, H.members[h_all])
            feas[h_all] = feasible(Pm[m], N[m], fp.cfg.gripper_width)
        best_feas = rr[fmask].max() if fmask.any() else 0.0
        for alpha in ALPHAS:
            rec[f"rel{alpha}"] = ("infeasible" if (not feas[h_all] and best_feas < alpha * rr[h_all])
                                  else "ok")
        out.append(rec)
    return out


def main():
    global META
    cache, meta_json = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    META = json.load(open(meta_json))
    files = [f for f in sorted(glob.glob(os.path.join(cache, "*.pkl"))) if ".sam3" not in os.path.basename(f)]
    with Pool(workers) as pool:
        recs = [r for rs in pool.map(run, files, chunksize=1) for r in rs]
    big = [r for r in recs if not r["gt_graspable"]]
    small = [r for r in recs if r["gt_graspable"]]
    print(f"{len(recs)} part queries: {len(big)} on ungraspable GT parts, {len(small)} on graspable ones")
    print("| r_min | ungraspable parts reported infeasible | graspable parts falsely reported infeasible |")
    print("|---|---|---|")
    for rmin in R_MINS:
        a = np.mean([r[rmin] == "infeasible" for r in big]) if big else np.nan
        b = np.mean([r[rmin] == "infeasible" for r in small]) if small else np.nan
        print(f"| {rmin} | {a:.3f} | {b:.3f} |")
    for alpha in ALPHAS:
        k = f"rel{alpha}"
        a = np.mean([r[k] == "infeasible" for r in big]) if big else np.nan
        b = np.mean([r[k] == "infeasible" for r in small]) if small else np.nan
        print(f"| relative alpha={alpha} | {a:.3f} | {b:.3f} |")
    print("ungraspable parts by name:", sorted({(r["part"]) for r in big}))


if __name__ == "__main__":
    main()
