"""Fig. 4: semantic part quality vs. graspability, one point per method, from eval_cache's
rows.pkl. x = semantic mIoU (part-name queries, category-accumulated), y = fraction of graspable-part queries
whose selection is both correct (IoU >= 0.5) and graspable.

usage: python fig4_scatter.py ROWS_PKL OUT_PREFIX
"""
import os, pickle, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
OURS, BASE = "#2a78d6", "#a8a79f"

rows = pickle.load(open(sys.argv[1], "rb"))
out = sys.argv[2]


def cat_mean_iou(by_cat, key):
    """Class mean of category-accumulated part IoU (sum I / sum U per part)."""
    vals = []
    for rs in by_cat.values():
        iu = [r[key]["iu"] for r in rs if key in r]
        if not iu:
            continue
        I = np.sum([[x[0] for x in a] for a in iu], 0)
        U = np.sum([[x[1] for x in a] for a in iu], 0)
        if (U > 0).any():
            vals.append(np.mean(I[U > 0] / U[U > 0]))
    return float(np.mean(vals)) if vals else np.nan


def grasp_rate(by_cat, key, mode="feasible"):
    recs = [g[mode] for rs in by_cat.values() for r in rs for g in r.get(key, [])]
    if not recs:
        return np.nan, 0
    return float(np.mean([x["iou"] >= 0.5 and x["graspable"] for x in recs])), len(recs)


methods = [  # label, row, semantic key, grasp key, ours?
    ("FitPart + SAM 3", "full", "sem_iou_selmulti_sam3", "grasp_sam3", True),
    ("FitPart + SigLIP", "full", "sem_iou_sel", "grasp", True),
    ("FitPart, no geometry + SAM 3", "geo:none", "sem_iou_selmulti_sam3", "grasp_sam3", True),
    ("ITSI-jump geometry + SAM 3", "geo:jump", "sem_iou_selmulti_sam3", "grasp_sam3", False),
    ("PartField + SAM 3", "baseline: PartField", "sem_iou_selmulti_sam3", "grasp_sam3", False),
    ("SAM 3 lifted, no hierarchy", "full", "sem_iou_sam3direct", "grasp_sam3direct", False),
]
pts = []
for label, row, sk, gk, ours in methods:
    if row not in rows:
        continue
    x = cat_mean_iou(rows[row], sk)
    y, n = grasp_rate(rows[row], gk)
    if np.isfinite(x) and np.isfinite(y):
        pts.append((label, x, y, n, ours))
        print(f"{label:34s} sem mIoU {x:.3f}  correct&graspable {y:.3f} (n={n})")

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(3.4, 2.8))
placed = []
for label, x, y, n, ours in pts:
    ax.scatter(x, y, s=40, color=OURS if ours else BASE, edgecolor="white", linewidth=1.5, zorder=3)
    # stack labels of (near-)coincident points instead of overprinting them
    k = sum(1 for (px, py) in placed if abs(px - x) < 0.04 and abs(py - y) < 0.01)
    placed.append((x, y))
    ax.annotate(label, (x, y), xytext=(6, 3 - 9 * k), textcoords="offset points", fontsize=6.5, color=INK)
ax.set_xlabel("semantic part mIoU (part-name queries)", color=INK)
ax.set_ylabel("graspable-part queries:\ncorrect and graspable", color=INK)
ax.grid(color=GRID, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
lo = min(p[1] for p in pts) if pts else 0
ax.set_xlim(max(0, lo - 0.05), max(p[1] for p in pts) + 0.15 if pts else 1)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
