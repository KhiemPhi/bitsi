"""ShapeNetPart training data for the cross-dataset (Z1) detector: candidate-node features of
the FitPart hierarchy + per-type positives (node IoU >= 0.5 with the GT part or one of its
connected pieces). Label ids verified geometrically (see RESULTS). Writes <obj>.snpfeat.npz.

usage: python snp_node_features.py SNP_CACHE [workers]
"""
import glob, os, pickle, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

# ShapeNetPart global label id -> PartNet-E part type (only types shared with PartNet-E)
LABEL_TYPE = {4: "handle", 23: "handle", 33: "handle", 36: "handle", 39: "handle",
              14: "leg", 48: "leg", 10: "wheel", 32: "wheel", 44: "wheel",
              13: "seat", 31: "seat", 22: "blade", 24: "base", 29: "screen"}
SIZE_FALLBACK = 0.5  # metres, for categories without a size prior


def run(f):
    import traceback
    try:
        return _run(f)
    except Exception:
        print("FAILED", os.path.basename(f), traceback.format_exc().splitlines()[-1], flush=True)


def _run(f):
    from scipy.sparse.csgraph import connected_components
    from fitpart.data import load_shapenetpart, shapenetpart_categories
    from fitpart.metrics import node_gt_iou
    from fitpart.node_features import node_features
    from fitpart.pipeline import Config, FitPart
    from fitpart.scales import metric_scale
    from fitpart.superpoints import knn_graph
    out = f[:-4] + ".snpfeat.npz"
    if os.path.exists(out):
        return
    L = pickle.load(open(f, "rb"))
    if "points" not in L:
        sid = {v: k for k, v in shapenetpart_categories().items()}[L["cat"]]
        P, N, _ = load_shapenetpart(sid, L["oid"])
    else:
        P, N = L["points"].astype(float), L["normals"].astype(float)
    gt = L["sem"]
    types = sorted({LABEL_TYPE[l] for l in np.unique(gt) if l in LABEL_TYPE})
    if not types:
        return
    scale = metric_scale(P, L["cat"]) or SIZE_FALLBACK / float(np.ptp(P, 0).max())
    res = FitPart(Config(up=(0.0, 1.0, 0.0))).assemble(L)
    X, cand = node_features(P, res.sp, L["pairs"], res.hier, scale)
    g = knn_graph(P, N, k=8)
    pos = []
    for tp in types:
        m = np.isin(gt, [l for l, t in LABEL_TYPE.items() if t == tp])
        iou, ids = node_gt_iou(res.hier, res.sp, m.astype(int))
        best = iou[cand, list(ids).index(1)] >= 0.5
        # instances: connected pieces of the part (e.g. single legs, single wheels)
        idx = np.where(m)[0]
        ncc, cc = connected_components(g[idx][:, idx], directed=False)
        inst = -np.ones(len(gt), int)
        inst[idx] = cc
        iou_i, ids_i = node_gt_iou(res.hier, res.sp, inst)
        cols = [k for k, v in enumerate(ids_i) if v >= 0]
        best |= iou_i[cand][:, cols].max(1) >= 0.5
        pos.append(best)
    np.savez_compressed(out, X=X.astype(np.float32), pos=np.array(pos), types=np.array(types), cat=L["cat"])


if __name__ == "__main__":
    files = [f for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.pkl"))) if ".sam3" not in os.path.basename(f)]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 10) as p:
        p.map(run, files, chunksize=1)
    print(len(glob.glob(os.path.join(sys.argv[1], "*.snpfeat.npz"))), "objects with features")
