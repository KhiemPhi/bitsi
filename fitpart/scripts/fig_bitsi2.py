"""BITSI v2 qualitative: GT | HOG events | BITSI v2 segments | BITSI v2 primitive tree (root body
grey by coaxial segment, children coloured; legend lists each child's attachment count).
usage: python fig_bitsi2.py OUT.png Cat/oid [Cat/oid ...]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fitpart.bitsi2 import BITSI2
from fitpart.data import PARTNETE_ROOT, load_partnete
from fitpart.metrics import class_agnostic_miou
from fitpart.profiles import pca_axes
from fitpart.sweep_events import events_along

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5af5', '#8c6d46', '#00a3c4', '#c4003a']
GREYS = ['#c9c8c0', '#a8a79f', '#dedcd3', '#8f8e87', '#b8b7ae']


def cols(lab):
    _, inv = np.unique(lab, return_inverse=True)
    return np.array([PAL[i % len(PAL)] for i in inv])


def main():
    out, objs = sys.argv[1], sys.argv[2:]
    fig = plt.figure(figsize=(16, 4.3 * len(objs)))
    for row, co in enumerate(objs):
        P, N, C, sem, ins = load_partnete(os.path.join(PARTNETE_ROOT, "data", "test", co))
        gt = (ins if ins is not None else sem).copy(); gt[sem < 0] = gt.max() + 1
        ev, prof = events_along(P, pca_axes(P)[0], N, C, ("hist",))
        hog = np.searchsorted(np.sort([c.t for c in ev]), prof["coord"])
        lab, root = BITSI2().fit(P, N, C)
        tc = np.array([GREYS[0]] * len(P), dtype=object)
        _, inv = np.unique(root.seg_labels, return_inverse=True)
        tc[root.body] = [GREYS[i % len(GREYS)] for i in inv]
        leg = []
        for j, ch in enumerate(root.children):
            tc[ch.idx] = PAL[j % len(PAL)]
            leg.append((PAL[j % len(PAL)], f"child {j + 1}: {ch.n_attach} attachment patch(es)" + (" = LOOP" if ch.loop else "")))
        sp = root.spine
        a = sp.tangents[0]
        z = a; x = np.cross(z, [1, 0, 0] if abs(z[0]) < .9 else [0, 1, 0]); x /= np.linalg.norm(x); y = np.cross(z, x)
        c0 = P.mean(0)
        V = (P - c0) @ np.stack([x, y, z]).T
        panels = [(f"{co}: ground truth ({len(np.unique(gt))} parts)", cols(gt)),
                  (f"HOG sweep events: mIoU {class_agnostic_miou(gt, hog)[0]:.2f}", cols(hog)),
                  (f"BITSI v2 segments: mIoU {class_agnostic_miou(gt, lab)[0]:.2f}", cols(lab)),
                  (f"BITSI v2 tree: {sp.kind} root spine, {len(root.cuts) + 1} coaxial segments", tc)]
        for j, (t, cl) in enumerate(panels):
            ax = fig.add_subplot(len(objs), 4, row * 4 + j + 1, projection="3d")
            ax.scatter(*V[::2].T, c=list(cl[::2]), s=6, linewidths=0, depthshade=False)
            if j == 3:
                Cn = (sp.centres - c0) @ np.stack([x, y, z]).T
                ax.plot(*Cn.T, color="#1f1f1e", lw=2)
                for col, txt in leg[:4]:
                    ax.plot([], [], "o", color=col, label=txt)
                if leg:
                    ax.legend(fontsize=7, frameon=False, loc="lower left")
            r = np.ptp(V, 0).max() * .55; m = V.mean(0)
            ax.set_xlim(m[0] - r, m[0] + r); ax.set_ylim(m[1] - r, m[1] + r); ax.set_zlim(m[2] - r, m[2] + r)
            ax.view_init(15, 35); ax.set_axis_off(); ax.set_title(t, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=100)


if __name__ == "__main__":
    main()
