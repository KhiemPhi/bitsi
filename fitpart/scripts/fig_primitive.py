"""Qualitative primitive decomposition on PartNet-E: GT | HOG sweep events | primitive (axis drawn,
residual parts in strong colours) | the fitted radius profile R(s).
usage: python fig_primitive.py OUT.png Cat/oid [Cat/oid ...]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fitpart.data import PARTNETE_ROOT, load_partnete
from fitpart.metrics import class_agnostic_miou
from fitpart.primitive import primitive_decomposition
from fitpart.profiles import pca_axes
from fitpart.sweep_events import events_along

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5af5', '#8c6d46', '#00a3c4', '#c4003a']
BODY = ['#c9c8c0', '#a8a79f', '#dedcd3', '#8f8e87', '#b8b7ae']


def cols(lab):
    _, inv = np.unique(lab, return_inverse=True)
    return np.array([PAL[i % len(PAL)] for i in inv])


def main():
    out, objs = sys.argv[1], sys.argv[2:]
    fig = plt.figure(figsize=(16, 4.2 * len(objs)))
    for row, co in enumerate(objs):
        P, N, C, sem, ins = load_partnete(os.path.join(PARTNETE_ROOT, "data", "test", co))
        gt = (ins if ins is not None else sem).copy(); gt[sem < 0] = gt.max() + 1
        ev, prof = events_along(P, pca_axes(P)[0], N, C, ("hist",))
        hog = np.searchsorted(np.sort([c.t for c in ev]), prof["coord"])
        lab, info = primitive_decomposition(P, N, C, axial="hog_pca")
        res = info["residual"]
        pc = np.where(res, np.array([PAL[(l % len(PAL))] for l in lab]), np.array([BODY[l % len(BODY)] for l in lab]))
        a, c = info["axis"], info["centre"]
        s = (P - c) @ a
        # view: axis up
        z = a; x = np.cross(z, [1, 0, 0] if abs(z[0]) < .9 else [0, 1, 0]); x /= np.linalg.norm(x); y = np.cross(z, x)
        V = (P - c) @ np.stack([x, y, z]).T
        panels = [(f"{co}: ground truth ({len(np.unique(gt))} parts)", cols(gt)),
                  (f"HOG sweep events: mIoU {class_agnostic_miou(gt, hog)[0]:.2f}, {len(np.unique(hog))} segs", cols(hog)),
                  (f"Primitive: mIoU {class_agnostic_miou(gt, lab)[0]:.2f}; off-axis parts coloured, body grey", pc)]
        for j, (t, cl) in enumerate(panels):
            ax = fig.add_subplot(len(objs), 4, row * 4 + j + 1, projection="3d")
            ax.scatter(*V[::2].T, c=cl[::2], s=6, linewidths=0, depthshade=False)
            if j == 2:
                ax.plot([0, 0], [0, 0], [s.min() - .05 * np.ptp(s), s.max() + .05 * np.ptp(s)], color="#1f1f1e", lw=2)
            r = np.ptp(V, 0).max() * .55; m = V.mean(0)
            ax.set_xlim(m[0] - r, m[0] + r); ax.set_ylim(m[1] - r, m[1] + r); ax.set_zlim(m[2] - r, m[2] + r)
            ax.view_init(15, 35); ax.set_axis_off(); ax.set_title(t, fontsize=9)
        ax = fig.add_subplot(len(objs), 4, row * 4 + 4)
        Q = P - c; rr = np.linalg.norm(Q - np.outer(s, a), axis=1)
        ax.scatter(s[~res][::3], rr[~res][::3], s=3, color="#a8a79f", linewidths=0, label="body points")
        ax.scatter(s[res][::2], rr[res][::2], s=4, color="#eb6834", linewidths=0, label="residual (off-axis parts)")
        ax.plot(info["t"], info["R"], color="#2a78d6", lw=2, label="fitted body radius R(s)")
        ax.set_xlabel("position along symmetry axis s"); ax.set_ylabel("distance from axis r")
        ax.set_title(f"primitive profile (axis score {info['axis_score']:.2f})", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=100)


if __name__ == "__main__":
    main()
