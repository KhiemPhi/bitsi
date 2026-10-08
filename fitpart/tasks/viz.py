"""Visualising a task screw: the grasp heatmap inset (as in the pour videos) and the screw in the scene.

task_inset: (left) the object cloud with the screw's grasp part in purple; (right) per surface point, the
best tau of any candidate grasp with a contact there (green = low -> red = best), the task screw (axis
omega through q, curl = + rotation; a straight double arrow for a translation), and the chosen grasp
(blue). tau = ITSI's SOCP screw metric (sim/screw_metric.py) under this task's screw.
screw_overlay: segments for PandaScene.add_overlay (rotation: axis + curl; translation: an arrow along
the motion).
"""
import numpy as np


def task_inset(W, gmask, gs, taus, chosen, ts, title, size=(800, 330)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from scipy.spatial import cKDTree
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    sub = slice(None, None, max(1, len(W) // 6000))
    ctr = W.mean(0)
    r = np.ptp(W, 0).max() * 0.40
    heat = np.full(len(W), np.nan)
    ok = [k for k, tv in enumerate(taus) if np.isfinite(tv) and tv > -1e8]
    if ok:
        tree = cKDTree(W)
        for k in ok:
            for cpt in gs[k][:2]:
                for j in tree.query_ball_point(cpt, 0.006):
                    heat[j] = taus[k] if np.isnan(heat[j]) else max(heat[j], taus[k])
    axis = ts.axis / np.linalg.norm(ts.axis)
    trans = not np.isfinite(ts.pitch)
    view = np.degrees(np.arctan2(*(np.cross([0, 0, 1.0], axis)[[1, 0]]))) + 35 if not trans else -60
    for pnl in (0, 1):
        ax = fig.add_subplot(1, 2, pnl + 1, projection="3d")
        if pnl == 0:
            ax.scatter(*W[sub].T, c=np.where(gmask, "#7a5af5", "#c9c8c0")[sub], s=9, linewidths=0, depthshade=False)
            ax.set_title(title, fontsize=10)
        else:
            has = ~np.isnan(heat)
            ax.scatter(*W[~has][sub].T, c="#dedcd3", s=9, linewidths=0, depthshade=False)
            if has.any():
                cmap = LinearSegmentedColormap.from_list("tau", ["#1a9850", "#fee08b", "#fc8d59", "#d73027"])
                hv = heat[has]
                norm = Normalize(vmin=float(hv.min()), vmax=float(max(hv.max(), hv.min() + 1e-6)))
                o = np.argsort(hv)
                ax.scatter(*W[has][o].T, c=cmap(norm(hv[o])), s=22, linewidths=0, depthshade=False)
                cax = fig.add_axes([0.875, 0.14, 0.016, 0.6])
                cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                cb.set_label("τ under the task screw" + (" (N)" if trans else " (Nm)"), fontsize=8)
                cb.ax.tick_params(labelsize=7)
                cb.outline.set_visible(False)
            q = np.asarray(ts.point, float)
            L = 0.8 * r
            if trans:   # translation: arrow along the motion through q
                tail, tip = q - axis * 0.3 * L, q + axis * L
                ax.plot(*np.stack([tail, tip]).T, color="#c4003a", lw=2.5)
                ax.quiver(*(tip - axis * 0.3 * L), *axis, color="#c4003a", length=0.3 * L, normalize=True,
                          arrow_length_ratio=0.8, lw=3.5)
                ax.text(*(tip + axis * 0.08 * L), "v", color="#c4003a", fontsize=14, fontweight="bold")
            else:
                sgn = np.sign(ts.theta) or 1.0
                w = axis * sgn          # draw the commanded rotation sense
                tail, tip = q - w * L, q + w * L
                ax.plot(*np.stack([tail, tip]).T, color="#c4003a", lw=2.5)
                ax.quiver(*(tip - w * 0.3 * L), *w, color="#c4003a", length=0.3 * L, normalize=True,
                          arrow_length_ratio=0.8, lw=3.5)
                ax.text(*(tip + w * 0.08 * L), "ω", color="#c4003a", fontsize=15, fontweight="bold")
                u = np.cross(w, [0, 0, 1.0]) if abs(w[2]) < 0.95 else np.cross(w, [1.0, 0, 0])
                u /= np.linalg.norm(u)
                v2 = np.cross(w, u)
                th = np.linspace(0, 1.4 * np.pi, 30)
                arc = q + w * L * 0.45 + 0.25 * L * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v2))
                ax.plot(*arc.T, color="#c4003a", lw=1.8)
                ax.quiver(*arc[-1], *(arc[-1] - arc[-3]), color="#c4003a", length=0.12 * L, normalize=True,
                          arrow_length_ratio=1.0, lw=1.8)
            ax.scatter(*q, color="#c4003a", s=30)
            ax.text(*(q + np.array([0, 0, -0.02 * r / 0.1])), "q", color="#c4003a", fontsize=11)
            if chosen is not None:
                c1, c2 = chosen[0], chosen[1]
                ax.plot(*np.stack([c1, c2]).T, color="#2a78d6", lw=3.5)
                ax.scatter(*np.stack([c1, c2]).T, color="#2a78d6", s=110, edgecolors="white", linewidths=1.5, depthshade=False, zorder=10)
            ax.set_title(f"grasp heatmap (red = best τ), chosen grasp (blue)\n{ts.task} screw: "
                         + ("translation along v" if trans else "axis ω through q, curl = + rotation"), fontsize=9)
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        ax.view_init(28, view)
        ax.set_axis_off()
    fig.subplots_adjust(left=0, right=0.87, bottom=0, top=0.84, wspace=0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def screw_overlay(ts, start, L=0.12, r_arc=0.035):
    """Overlay segments (PandaScene.add_overlay format) for the commanded screw, drawn where it acts
    (after the lift): rotation = axis through q with a curl in the commanded sense; translation = an
    arrow from the grasp along the motion."""
    a = ts.axis / np.linalg.norm(ts.axis)
    if not np.isfinite(ts.pitch):
        d = a * np.sign(ts.theta or 1.0)
        return [("arrow", start, start + d * max(L * 0.8, 0.08), 0.004)]
    w = a * (np.sign(ts.theta) or 1.0)
    q = np.asarray(ts.point, float) + np.array([0, 0, ts.lift])
    segs = [("arrow", q - w * L, q + w * L, 0.004)]
    u = np.cross(w, [0, 0, 1.0]) if abs(w[2]) < 0.95 else np.cross(w, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    c = q + w * L * 0.7
    th = np.linspace(0, 1.5 * np.pi, 14)
    pts = [c + r_arc * (np.cos(t) * u + np.sin(t) * v) for t in th]
    for k in range(len(pts) - 1):
        segs.append(("arrow" if k == len(pts) - 2 else "capsule", pts[k], pts[k + 1], 0.002))
    return segs
