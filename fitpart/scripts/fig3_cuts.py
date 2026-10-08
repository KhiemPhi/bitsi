"""Fig. 3: one BITC profile, Chebyshev fits at three degrees with inflection cuts (a),
versus ITSI-style thresholded jumps on the same profile (b).

usage: python fig3_cuts.py OUT_PREFIX [category=Knife] [index=1] [delta=0.2]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import chebyshev as C

from fitpart.cheb import jump_threshold_cuts, multiscale_cuts
from fitpart.data import load_shapenetpart, shapenetpart_categories, shapenetpart_split
from fitpart.profiles import bitc_profile, pca_axes

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal blue: coarse -> fine
CUT = "#eb6834"                            # categorical slot 2

out = sys.argv[1]
cat = sys.argv[2] if len(sys.argv) > 2 else "Knife"
k = int(sys.argv[3]) if len(sys.argv) > 3 else 1
delta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
cats = shapenetpart_categories()
sid, oid = [x for x in shapenetpart_split("test") if cats[x[0]] == cat][k]
P, N, gt = load_shapenetpart(sid, oid)
prof = bitc_profile(P, pca_axes(P)[0])
t, y = prof["t"], prof["values"]
cuts, info = multiscale_cuts(t, y)
degs, coeffs = info["degrees"], info["coeffs"]
pick = [0, len(degs) // 2, len(degs) - 1] if len(degs) >= 3 else list(range(len(degs)))
yn0, yr = info["y0"], info["yr"]
x = np.linspace(-1, 1, 600)
tt = info["lo"] + (x + 1) / 2 * (info["hi"] - info["lo"])
# ground-truth part boundary along the sweep (for reference): where the majority label flips
coord = prof["coord"]
order = np.argsort(coord)
lab = gt[order]
flip = coord[order][np.where(lab[1:] != lab[:-1])[0]]
gt_b = np.median(flip) if len(flip) else None

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
for ax in axes:
    ax.scatter(t, y, s=6, color="#a8a79f", lw=0, label="BITC profile", zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if gt_b is not None:
        ax.axvline(gt_b, color=INK, lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.set_xlabel("position along sweep direction")
axes[0].set_ylabel("BITC")

ax = axes[0]
for j, i in enumerate(pick):
    ax.plot(tt, yn0 + yr * C.chebval(x, coeffs[i]), color=RAMP[j], lw=1.5, label=f"Chebyshev, n = {degs[i]}")
strong = sorted(cuts, key=lambda c: -c.strength)
for c in strong:
    ax.axvline(c.t, color=CUT, lw=0.5 + 2.0 * c.strength, alpha=0.25 + 0.75 * c.strength, zorder=2)
ax.set_title(f"(a) inflection cuts, width ~ strength s(c)  [{len(cuts)} candidates]", fontsize=8, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7, loc="upper left")

ax = axes[1]
jumps = jump_threshold_cuts(t, y, delta)
for c in jumps:
    ax.axvline(c.t, color=CUT, lw=1.0, zorder=2)
ax.set_title(f"(b) thresholded jumps, δ = {delta}  [{len(jumps)} cuts]", fontsize=8, color=INK, loc="left")
fig.text(0.99, 0.01, "dashed: ground-truth part boundary", ha="right", fontsize=6.5, color=MUTED)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
print(f"{cat} {oid}: degrees {degs}, {len(cuts)} inflection candidates, {len(jumps)} jumps; "
      f"strongest cut at {strong[0].t:.3f} (s={strong[0].strength:.2f}), GT boundary {gt_b}")
