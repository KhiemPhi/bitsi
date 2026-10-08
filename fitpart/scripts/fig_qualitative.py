"""Qualitative grid: for each (object, query) the GT part and each method's selection,
rendered from one viewpoint (selected points in color, rest in grey).

usage: python fig_qualitative.py CACHE_DIR OUT_PNG META_JSON "Cat:oid:query" [...]
       (oid may be '*' for the first cached object of that category)
"""
import glob, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fitpart import lifting
from fitpart.pipeline import Config, FitPart, _up_rotation
from fitpart.render import render_views
from fitpart.sam3_select import node_scores, superpoint_scores
from fitpart.selection import CANONICAL, node_membership, relevance, select

SEL = np.array([0.92, 0.41, 0.20])   # orange #eb6834-ish
GT = np.array([0.16, 0.47, 0.84])    # blue #2a78d6
GREY = np.array([0.78, 0.78, 0.76])


def view(P, N, mask, color, az_index=0):
    cols = np.where(mask[:, None], color[None, :], GREY[None, :])
    R = _up_rotation((0.0, 1.0, 0.0))
    r = render_views(P @ R.T, cols, N @ R.T, k=8, size=384, device="cpu")
    # choose the view that shows the most of the mask
    vis = [(r["pix2pt"][v][r["pix2pt"][v] >= 0][:, None] == np.where(mask)[0][None, :]).any(1).sum()
           if mask.any() else 0 for v in range(8)]
    return r["images"][int(np.argmax(vis))]


def main():
    cache, out, meta_json = sys.argv[1], sys.argv[2], sys.argv[3]
    specs = sys.argv[4:]
    meta = json.load(open(meta_json))
    fp = FitPart(Config())
    rows = []
    for spec in specs:
        cat, oid, query = spec.split(":", 2)
        f = sorted(glob.glob(os.path.join(cache, f"{cat}__{oid}.pkl")))
        f = [x for x in f if not x.endswith(".sam3.pkl")][0]
        L = pickle.load(open(f, "rb"))
        s3 = pickle.load(open(f[:-4] + ".sam3.pkl", "rb")) if os.path.exists(f[:-4] + ".sam3.pkl") else {}
        res = fp.assemble(L)
        sp, H = res.sp, res.hier
        P, N = L["points"].astype(float), L["normals"].astype(float)
        _, sizes = node_membership(H, np.bincount(sp).astype(float))
        names = meta[cat]
        part = query if query in names else None
        gt = (L["sem"] == names.index(part)) if part else np.zeros(len(P), bool)
        t = lifting.encode_text([f"the {query}"] + list(CANONICAL))
        cols = [("ground truth", gt, GT)]
        h, _ = select(relevance(res.z_nodes, t[0], t[1:]), sizes, None, fp.cfg.eps_r, min_size=0.01 * len(sp))
        cols.append(("FitPart + SigLIP", np.isin(sp, H.members[h]), SEL))
        if query in s3:
            h, _ = select(node_scores(H, sp, s3[query]), sizes, None, fp.cfg.eps_r, min_size=0.01 * len(sp))
            cols.append(("FitPart + SAM 3", np.isin(sp, H.members[h]), SEL))
            cols.append(("SAM 3 lifted, no hierarchy", (superpoint_scores(sp, s3[query]) >= 0.5)[sp], SEL))
        rows.append((f"{cat}: “{query}”", [(n, view(P, N, m, c)) for n, m, c in cols]))
    ncol = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.2 * ncol, 2.3 * len(rows)), squeeze=False)
    for i, (label, panels) in enumerate(rows):
        for j in range(ncol):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(panels):
                ax.imshow(panels[j][1])
                if i == 0:
                    ax.set_title(panels[j][0], fontsize=8)
        axes[i, 0].text(-0.05, 0.5, label, transform=axes[i, 0].transAxes, rotation=90, va="center",
                        ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
