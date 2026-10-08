"""Figure: BITSI v4 segments on the robot's partial point clouds (eye-in-hand, section 21), successes and
failures. Per object: (1) GT parts on the fused partial cloud, (2) BITSI v4's segments on the same cloud,
coloured by the GT part each is Hungarian-matched to (the matching of the partition metric); unmatched
extra segments in grey shades, (3) the full object with the surface the camera never saw faint.
The v4 fit is re-run on the saved cloud (frozen v4, as in the evaluation) and its score must equal the
logged one.
usage (sam3 env): python fig_partial_segments.py PICKS.json OUT_PREFIX [robot=panda] [strategy=nbv_bitsi] [K=3]
"""
import json, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from fitpart.bitsi4 import BITSI4
from fitpart.metrics import partition_scores
from fitpart.scripts.eval_eyeinhand import CLOUDS, load, objects

PART = ["#2a78d6", "#eb6834", "#1baf7a", "#7a5af5", "#eda100", "#e87ba4", "#00a3c4", "#8c6d46"]
EXTRA = ["#9a9992", "#bebdb6", "#6f6e69", "#d6d5cf"]


def matched_colours(gt, pred):
    g_ids, g = np.unique(gt, return_inverse=True)
    p_ids, p = np.unique(pred, return_inverse=True)
    inter = np.zeros((len(g_ids), len(p_ids)))
    np.add.at(inter, (g, p), 1)
    iou = inter / (inter.sum(1, keepdims=True) + inter.sum(0, keepdims=True) - inter)
    r, c = linear_sum_assignment(-iou)
    col_p = {}
    for gi, pi in zip(r, c):
        col_p[pi] = PART[gi % len(PART)]
    k = 0
    for pi in range(len(p_ids)):
        if pi not in col_p:
            col_p[pi] = EXTRA[k % len(EXTRA)]
            k += 1
    return np.array([PART[i % len(PART)] for i in g]), np.array([col_p[i] for i in p]), len(p_ids) - len(r)


def panel(ax, P, cols, title, view, faint=None, s=5):
    if faint is not None and len(faint):
        ax.scatter(*faint.T, c="#e7e6e0", s=1, linewidths=0, depthshade=False)
    ax.scatter(*P.T, c=cols, s=s, linewidths=0, depthshade=False)
    allp = P if faint is None else np.vstack([P, faint])
    c = 0.5 * (allp.min(0) + allp.max(0))
    r = 0.42 * np.ptp(allp, 0).max()
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(*view)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def main(picks, prefix, robot="panda", strat="nbv_bitsi", K=3):
    warnings.filterwarnings("ignore")
    P_ = json.load(open(picks))
    its = {it["key"]: it for it in objects()}
    logged = {json.loads(l)["key"]: json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "eih", f"{robot}.jsonl"))}
    groups = [("Successes", P_["good"]), ("Failures", P_["bad"])]
    n = sum(len(v) for _, v in groups)
    fig = plt.figure(figsize=(13.5, 2.9 * n + 1.2), dpi=110)
    row = 0
    for gname, keys in groups:
        for key in keys:
            it = its[key]
            W, Nw, C, gt = load(it)
            z = np.load(os.path.join(CLOUDS, robot, strat, f"{key.replace('/', '__')}__K{K}.npz"))
            idx, pts, nrm = z["idx"], z["pts"], z["n"].astype(float)
            M = BITSI4()
            lab, root = M.fit(pts, nrm, C[idx] if C is not None else None)
            a = root.spine.tangents[0] / np.linalg.norm(root.spine.tangents[0])
            tilt = float(np.degrees(np.arccos(min(1.0, abs(a[2])))))      # spine vs vertical (objects stand upright)
            sc = partition_scores(gt[idx], lab)
            log = [e for e in logged[key][strat]["evals"] if e["k"] == K][0]["v4"]
            # the saved cloud is float32 points / float16 normals (the run used float64): tiny differences
            assert abs(sc["sym_miou"] - log["sym_miou"]) < 0.01, (key, sc, log)
            gcol, pcol, n_extra = matched_colours(gt[idx], lab)
            seen = cKDTree(pts).query(W)[0] < 2 * np.median(cKDTree(W).query(W, k=2)[0][:, 1])
            view = (22, -60)
            ax = fig.add_subplot(n, 3, 3 * row + 1, projection="3d")
            label = "SUCCESS" if gname == "Successes" else "FAILURE"
            panel(ax, pts, gcol, f"{label}: {key}\nGT parts on the partial cloud ({len(pts)} pts, {K} views)", view)
            ax = fig.add_subplot(n, 3, 3 * row + 2, projection="3d")
            panel(ax, pts, pcol, f"BITSI v4: {sc['n_pred']} segments (GT {sc['n_gt']}), spine {tilt:.0f} deg from vertical\n"
                  f"sym mIoU {sc['sym_miou']:.2f}, ARI {sc['ari']:.2f}" + (f", {n_extra} extra (grey)" if n_extra else ""), view)
            ax = fig.add_subplot(n, 3, 3 * row + 3, projection="3d")
            g_ids, g = np.unique(gt, return_inverse=True)
            panel(ax, W[seen], np.array([PART[i % len(PART)] for i in g[seen]]),
                  f"full object: {seen.mean():.0%} of the surface seen\n(faint = never observed)", view, faint=W[~seen], s=2)
            row += 1
            print(gname, key, sc, flush=True)
    top = 1 - 1.2 / (2.9 * n + 1.2)
    fig.text(0.01, 1 - 0.25 / (2.9 * n + 1.2), f"BITSI v4 on the robot's partial clouds ({robot}, {strat} views, K = {K})", fontsize=12, va="top", fontweight="bold")
    fig.text(0.01, 1 - 0.6 / (2.9 * n + 1.2), "Colours: each BITSI segment takes the colour of the GT part it is matched to "
             "(Hungarian, as in the metric); grey = extra segments (over-segmentation).", fontsize=10, va="top")
    fig.subplots_adjust(left=0.0, right=1.0, top=top, bottom=0.0, hspace=0.18, wspace=0.0)
    fig.savefig(prefix + ".png")
    plt.close(fig)
    print("saved", prefix + ".png")


if __name__ == "__main__":
    main(*sys.argv[1:3], *(sys.argv[3:4] or []), *(sys.argv[4:5] or []), *([int(sys.argv[5])] if len(sys.argv) > 5 else []))
