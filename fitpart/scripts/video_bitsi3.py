"""Explainer video of BITSI v3: the point cloud and the BITSI v3 function side by side.

Both panels share the vertical direction: the cloud is rotated so the spine is vertical, and the BITSI v3
function is plotted with the spine coordinate s running upward (heights match up to perspective). So a cut is a horizontal line in
the plot AND a horizontal disk in the cloud at the same height, both labelled "cut k" in the same
colour. Each coaxial segment has one colour in both panels (its points, its Chebyshev piece, its
band in the plot). A red ring (cloud) and red line (plot) show the slice being swept.

Right panels: a0(s) = mean radius of the primitive's cross-section, |c1|(s) = its first harmonic
(off-centre bulge), per slice (dots) and the Chebyshev piece MDL chose for each segment (lines).
Third panel: the cloud unrolled around the spine (angle phi across, s up), each point coloured by how
far it sticks out of the body's fitted envelope, (rho - rho_hat)/h. This is where off-axis parts come
from: points more than margin_h*h outside the envelope, grouped into connected components; a
component is a child primitive if it has >= 1% of the points and covers <= 50% of the angle (a full
ring would be coaxial and is left to the cuts). Its contact patches with the body give the topology:
2+ = loop (handle), 1 = free end (spout).
Final frames: the child primitives (off-axis parts) and their attachment topology.
usage: python video_bitsi3.py Cat/oid OUT.mp4
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.polynomial import chebyshev as Ch

from fitpart.bitsi3 import BITSI3, piece_cost, standardize
from fitpart.data import PARTNETE_ROOT, load_partnete
from fitpart.metrics import partition_scores

SEG = ['#2a78d6', '#1baf7a', '#eda100', '#7a5af5', '#00a3c4', '#8c6d46']
CHILD = ['#eb6834', '#e87ba4', '#c4003a', '#008300']
CUT = '#1f1f1e'
SWEEP = '#d62728'


def main():
    warnings.filterwarnings("ignore")
    co, out = sys.argv[1], sys.argv[2]
    P, N, C, sem, ins = load_partnete(os.path.join(PARTNETE_ROOT, "data", "test", co))
    gt = (ins if ins is not None else sem).copy()
    gt[sem < 0] = gt.max() + 1
    render(P, N, C, out, co, gt=gt)


def render(P, N, C, out, co, gt=None, M=None, figsize=(16, 9), version="v3", n_sweep=110, n_end=60):
    """The explainer for any cloud (e.g. the robot's own object): M = a fitted-later BITSI model
    (default BITSI3(); BITSI4(up=...) for the robot setting); gt optional (only for the score)."""
    warnings.filterwarnings("ignore")
    M = M if M is not None else BITSI3()
    lab, root = M.fit(P, N, C)
    sc = partition_scores(gt, lab) if gt is not None else None
    tr = M.trace[0]
    sp = root.spine
    Y = tr["Y"]
    S = len(Y)
    t = 0.5 * (sp.edges[:-1] + sp.edges[1:])
    cuts = list(tr["cuts"])
    bounds = [0] + [c + 1 for c in cuts] + [S]
    Ys, _ = standardize(Y)
    pieces = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        _, ks = piece_cost(Ys, a, b)
        x = np.linspace(-1, 1, b - a)
        pieces.append((a, b, [Ch.chebval(x, Ch.chebfit(x, Y[a:b, j], int(min(ks[j], b - a - 1)))) for j in (0, 1)],
                       [int(ks[j]) for j in (0, 1)]))
    s_cut = [sp.edges[c + 1] for c in cuts]
    # cloud in the spine frame: s vertical
    a = sp.tangents[0]
    xax = np.cross(a, [1, 0, 0] if abs(a[0]) < .9 else [0, 1, 0]); xax /= np.linalg.norm(xax)
    yax = np.cross(a, xax)
    c0 = sp.centres[0] - sp.edges[0] * a if sp.kind != "geodesic" else P.mean(0)
    Rf = np.stack([xax, yax, a])
    V = (P - c0) @ Rf.T                      # V[:, 2] = s along the spine
    Cn = (sp.centres - c0) @ Rf.T
    body = np.zeros(len(P), bool)
    body[root.body] = True
    seg = np.searchsorted(np.asarray(cuts, int), sp.slab, side="left")
    child_of = np.full(len(P), -1)
    for j, ch in enumerate(root.children):
        child_of[ch.idx] = j
    se = root.section                        # root covers every point, so per-point arrays align with P
    excess = (se.rho - se.rho_hat) / M.h
    outside = excess > M.margin_h
    phi_deg = np.degrees(se.phi)
    child_cover = [np.count_nonzero(np.bincount(((se.phi[ch.idx] + np.pi) / (2 * np.pi) * 36).astype(int) % 36,
                                                minlength=36)) / 36 for ch in root.children]
    sub = np.arange(0, len(P), 2)
    lim = np.ptp(V, 0).max() * 0.55
    mid = 0.5 * (V.min(0) + V.max(0))
    fig = plt.figure(figsize=figsize, dpi=100)
    frames = []
    for f in range(n_sweep + n_end):
        fig.clf()
        sweeping = f < n_sweep
        k = min(S - 1, int(f / n_sweep * S)) if sweeping else S - 1
        s_now = t[k]
        # ---------------- left: point cloud
        ax = fig.add_axes([0.0, 0.04, 0.5, 0.86], projection="3d")
        cols = np.full(len(P), "#e3e2dc", dtype=object)
        seen = sp.slab <= k
        m = body & seen
        cols[m] = [SEG[i % len(SEG)] for i in seg[m]]
        if not sweeping:
            for j in range(len(root.children)):
                cols[child_of == j] = CHILD[j % len(CHILD)]
        ax.scatter(*V[sub].T, c=list(cols[sub]), s=6, linewidths=0, depthshade=False, alpha=0.9)
        ax.plot(*Cn.T, color=CUT, lw=1.5, alpha=0.7)
        th = np.linspace(0, 2 * np.pi, 48)
        for i, (c, s) in enumerate(zip(cuts, s_cut)):
            if s > s_now + 1e-9:
                continue
            ctr = 0.5 * (Cn[c] + Cn[c + 1])
            r = 1.3 * max(Y[max(0, c - 2):c + 3, 0].max(), 1e-3)
            disk = np.stack([ctr[0] + r * np.cos(th), ctr[1] + r * np.sin(th), np.full_like(th, s)], 1)
            ax.add_collection3d(Poly3DCollection([disk], facecolor=CUT, alpha=0.18, edgecolor=CUT, linewidth=1.5))
            ax.text(ctr[0] + r * 1.05, ctr[1], s, f"cut {i + 1}", color=CUT, fontsize=13, fontweight="bold")
        if sweeping:
            r = max(Y[k, 0], 1e-3)
            ax.plot(Cn[k, 0] + r * np.cos(th), Cn[k, 1] + r * np.sin(th), np.full_like(th, s_now), color=SWEEP, lw=3)
        ax.set_xlim(mid[0] - lim, mid[0] + lim); ax.set_ylim(mid[1] - lim, mid[1] + lim)
        ax.set_zlim(t[0], t[-1])
        ax.set_box_aspect((1, 1, (t[-1] - t[0]) / (2 * lim)))
        ax.view_init(10, 30 + 0.8 * f)
        ax.set_axis_off()
        # ---------------- right: the BITSI v3 function, s vertical
        for j, (nm, xl) in enumerate((("a0(s): mean radius", "a0"),
                                      ("|c1|(s): off-centre bulge", "|c1|"))):
            bx = fig.add_axes([0.52 + 0.15 * j, 0.22, 0.12, 0.66])
            lo, hi = Y[:, j].min(), Y[:, j].max()
            pad = 0.08 * (hi - lo) + 1e-4
            for i, (a_, b_, fv, ks) in enumerate(pieces):
                bx.axhspan(sp.edges[a_], sp.edges[b_], color=SEG[i % len(SEG)], alpha=0.08 if sp.edges[a_] <= s_now else 0.0, lw=0)
                if t[a_] <= s_now:
                    hi_ = min(b_, k + 1)
                    bx.plot(fv[j][:hi_ - a_], t[a_:hi_], color=SEG[i % len(SEG)], lw=3,
                            label=f"segment {i + 1}: Chebyshev degree {ks[j]}" if j == 0 else None)
            bx.plot(Y[:k + 1, j], t[:k + 1], "o", color="#55544e", ms=3, label="per-slice value" if j == 0 else None)
            for i, s in enumerate(s_cut):
                if s <= s_now + 1e-9:
                    bx.axhline(s, color=CUT, lw=2, ls="--")
                    bx.text(hi + pad * 0.6, s, f" cut {i + 1}", color=CUT, fontsize=12, fontweight="bold", va="center",
                            ha="right" if j == 0 else "right")
            if sweeping:
                bx.axhline(s_now, color=SWEEP, lw=2)
            bx.set_ylim(t[0], t[-1]); bx.set_xlim(lo - pad, hi + 3 * pad)
            bx.set_title(nm.replace(": ", ":\n"), fontsize=11)
            bx.set_xlabel(xl, fontsize=11)
            if j == 0:
                bx.set_ylabel("position along the spine s (vertical in the cloud)", fontsize=11)
                bx.legend(fontsize=10, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
            else:
                bx.set_yticklabels([])
            bx.spines[["top", "right"]].set_visible(False)
        # ---------------- right: the residual map that finds the off-axis parts
        rx = fig.add_axes([0.825, 0.22, 0.16, 0.66])
        m = seen[sub]
        ii = sub[m]
        inn, out_ = ii[~outside[ii]], ii[outside[ii]]
        rx.scatter(phi_deg[inn], V[inn, 2], s=2, c="#d9d8d2", linewidths=0, rasterized=True)
        if sweeping:
            rx.scatter(phi_deg[out_], V[out_, 2], s=4, c=np.clip(excess[out_], M.margin_h, 15), cmap="Oranges",
                       vmin=0, vmax=15, linewidths=0, rasterized=True)
        else:
            rest = out_[child_of[out_] < 0]
            rx.scatter(phi_deg[rest], V[rest, 2], s=4, c="#8a8984", linewidths=0, rasterized=True)
            for j, ch in enumerate(root.children):
                cj = out_[child_of[out_] == j]
                rx.scatter(phi_deg[cj], V[cj, 2], s=5, c=CHILD[j % len(CHILD)], linewidths=0, rasterized=True)
                ph, zz = phi_deg[ch.idx], V[ch.idx, 2]
                rx.text(np.median(ph), zz.min() - 0.02 * (t[-1] - t[0]),
                        f"child {j + 1}\n{child_cover[j]:.0%} of angle\n{ch.n_attach} contact{'s' if ch.n_attach != 1 else ''}\n"
                        + ("= loop" if ch.loop else "= free end"), color=CHILD[j % len(CHILD)],
                        fontsize=10, fontweight="bold", ha="center", va="top")
        for i, s_ in enumerate(s_cut):
            if s_ <= s_now + 1e-9:
                rx.axhline(s_, color=CUT, lw=1.5, ls="--")
        if sweeping:
            rx.axhline(s_now, color=SWEEP, lw=2)
        rx.set_xlim(-180, 180); rx.set_xticks([-180, -90, 0, 90, 180])
        rx.set_ylim(t[0], t[-1]); rx.set_yticklabels([])
        rx.set_title(f"sticks out of the body envelope\n(grey: inside; colour: > {M.margin_h:g}h outside)", fontsize=11)
        rx.set_xlabel("angle phi around the spine (deg)", fontsize=11)
        rx.spines[["top", "right"]].set_visible(False)
        fig.text(0.825, 0.13, "Off-axis parts = points outside the fitted\ncross-section envelope, grouped into\n"
                 "connected pieces; a piece is a child if it\nhas >= 1% of points and covers <= 50% of\n"
                 "the angle (a full ring is a coaxial part).", fontsize=9, color="#55544e", va="top")
        # ---------------- header and footer
        head = (f"BITSI {version} on {co}: sweeping the spine, a cut wherever two Chebyshev pieces are cheaper than one"
                if sweeping else f"BITSI {version} on {co}: {len(cuts) + 1} coaxial segments + {len(root.children)} child primitive(s)"
                + (f"   (GT {sc['n_gt']} parts; symmetric mIoU {sc['sym_miou']:.2f})" if sc else ""))
        fig.text(0.02, 0.95, head, fontsize=14, fontweight="bold", color="#1f1f1e")
        if not sweeping:
            for j, ch in enumerate(root.children[:4]):
                fig.text(0.02, 0.06 + 0.035 * j, f"child {j + 1}: {ch.n_attach} attachment patch(es) "
                         + ("= LOOP (graspable handle)" if ch.loop else "= free end (spout / knob)"),
                         color=CHILD[j % len(CHILD)], fontsize=12, fontweight="bold")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    import imageio.v3 as iio
    iio.imwrite(out, np.stack(frames), fps=20, codec="libx264")
    print("saved", out, len(frames), "frames;", len(cuts), "cuts at s =", np.round(s_cut, 3), ";",
          len(root.children), "children;", sc)


if __name__ == "__main__":
    main()
