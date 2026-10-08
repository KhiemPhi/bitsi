"""Figure: what the robot saw and what each method did, for eye-in-hand pour cases (eih_pour.py --dump).

One row per object: (1) the view map (reachable viewpoints, the views taken in order; the robot
stands at az 180), (2) the wrist depth images of those views (object pixels in colour), (3) the fused
partial cloud with the GT handle (red; handle surface never observed shown faint), (4-6) BITSI v4,
Find3D and whole-object: the part each selected (blue), the executed grasp (black) and the outcome.
usage: python fig_eih_cases.py DUMP_DIR OUT_PREFIX cat/oid [cat/oid ...]
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

INK, MUTED, GRID = "#1f1f1e", "#8a8984", "#d9d8d2"
HANDLE, PART, GRASP = "#d62728", "#2a78d6", "#1f1f1e"
OK, BAD = "#1b8a5a", "#c4003a"
NAMES = {"bitsi4": "BITSI v4 (ours)", "find3d": "Find3D", "whole": "Whole object"}


def outcome(r):
    if r.get("task_success"):
        return "SUCCESS: poured away from the handle", OK
    if r.get("success"):
        return f"FAILED: poured toward the handle ({r.get('pour_side_vs_gt_handle_deg', 0):.0f} deg off)", BAD
    if r.get("planned"):
        return "FAILED: dropped / wrong pose", BAD
    return "FAILED: no feasible grasp + plan", BAD


def view_map(ax, d):
    reach, views = d["reach"], d["views"]
    th = np.deg2rad(reach[:, 0])
    ax.scatter(th, 90 - reach[:, 1], s=10, c=GRID, label="reachable")
    vt = np.deg2rad(views[:, 0])
    vr = 90 - views[:, 1]
    ax.plot(vt, vr, "-", color=PART, lw=1.2, alpha=0.7)
    for k, (t, r) in enumerate(zip(vt, vr)):
        ax.scatter([t], [r], s=120, c=PART, zorder=3)
        ax.text(t, r, str(k + 1), color="white", ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
    ax.annotate("robot", xy=(np.pi, 88), xytext=(np.pi, 100), ha="center", fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED), annotation_clip=False)
    ax.set_ylim(0, 90)
    ax.set_yticks([15, 35, 55, 75])
    ax.set_yticklabels(["75°", "55°", "35°", "15°"], fontsize=6, color=MUTED)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=7)
    ax.set_title(f"views taken ({len(views)}) of {len(reach)} reachable\n(radius = 90° - elevation)", fontsize=9)


def depth_mosaic(ax, d):
    D, M = d["depth"], d["mask"]
    n = len(D)
    cols = 3
    rows = int(np.ceil(n / cols))
    h, w = D.shape[1:]
    canvas = np.ones((rows * h + (rows - 1) * 6, cols * w + (cols - 1) * 6, 3))
    for k in range(n):
        r, c = divmod(k, cols)
        dep = D[k].copy()
        valid = (dep > 0) & (dep < 1.5)
        g = np.ones((h, w, 3)) * 0.92
        if valid.any():
            lo, hi = np.percentile(dep[valid], [2, 98])
            t = np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1)
            g[valid] = (0.55 + 0.35 * t[valid])[:, None]
        obj = M[k]
        if obj.any():
            lo, hi = dep[obj].min(), dep[obj].max()
            t = np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1)
            g[obj] = plt.cm.viridis(1 - t[obj])[:, :3]
        y0, x0 = r * (h + 6), c * (w + 6)
        canvas[y0:y0 + h, x0:x0 + w] = g
    ax.imshow(canvas)
    for k in range(n):
        r, c = divmod(k, cols)
        ax.text(c * (w + 6) + 6, r * (h + 6) + 22, str(k + 1), color=INK, fontsize=9, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1))
    ax.set_axis_off()
    ax.set_title("wrist depth per view (object pixels coloured)", fontsize=9)


def cloud_ax(ax, W, colours, sizes, title, view, extra=None, grasp=None):
    o = np.argsort(sizes)
    ax.scatter(*W[o].T, c=[colours[i] for i in o], s=sizes[o], linewidths=0, depthshade=False)
    if extra is not None and len(extra):
        ax.scatter(*extra.T, c="#f4b6b6", s=1.5, linewidths=0, depthshade=False, alpha=0.6)
    if grasp is not None:
        g = np.asarray(grasp)
        ax.plot(*g.T, "-", color=GRASP, lw=2.5)
        ax.scatter(*g.T, c=GRASP, s=40, marker="X", depthshade=False)
    lim = np.ptp(W, 0).max() * 0.55
    mid = 0.5 * (W.min(0) + W.max(0))
    ax.set_xlim(mid[0] - lim, mid[0] + lim)
    ax.set_ylim(mid[1] - lim, mid[1] + lim)
    ax.set_zlim(mid[2] - lim, mid[2] + lim)
    ax.view_init(*view)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def row(fig, gs_row, d, res, cat, oid):
    W, gt = d["W"], d["gt_handle"]
    Wf, hf = d["W_full"], d["handle_full"]
    seen = cKDTree(W).query(Wf[hf])[0] < 0.004
    hdir = Wf[hf].mean(0) - Wf.mean(0)
    view = (22, np.degrees(np.arctan2(hdir[1], hdir[0])) - 55)
    ax = fig.add_subplot(gs_row[0], projection="polar")
    ax.set_theta_zero_location("E")
    view_map(ax, d)
    depth_mosaic(fig.add_subplot(gs_row[1]), d)
    col = np.where(gt, HANDLE, "#bdbcb6")
    siz = np.where(gt, 3.0, 1.2)
    cloud_ax(fig.add_subplot(gs_row[2], projection="3d"), W, col, siz,
             f"fused cloud: {len(W)} pts\nhandle seen {seen.mean():.0%} (faint = never seen)", view,
             extra=Wf[hf][~seen])
    for j, m in enumerate(("bitsi4", "find3d", "whole")):
        r = res[m]
        part = d.get(f"part_{m}", np.ones(len(W), bool)) if m != "whole" else np.ones(len(W), bool)
        c = np.where(part, PART, "#d9d8d2")
        s = np.where(part, 2.2, 1.0)
        txt, tc = outcome(r)
        iou = r.get("part_iou_with_handle")
        sub = f"part vs handle IoU {iou:.2f}" if iou is not None else "part = whole object"
        a = fig.add_subplot(gs_row[3 + j], projection="3d")
        cloud_ax(a, W, c, s, f"{NAMES[m]}\n{sub}", view, grasp=d.get(f"grasp_{m}"))
        a.text2D(0.5, -0.02, txt, transform=a.transAxes, ha="center", va="top", fontsize=8.5, color=tc, fontweight="bold")
    fig.text(0.004, gs_row[0].get_position(fig).y0 + 0.5 * gs_row[0].get_position(fig).height,
             f"{cat.replace('_', ' ')}\n{oid}", rotation=90, va="center", ha="left", fontsize=11, fontweight="bold", color=INK)


def main():
    dump, prefix, cases = sys.argv[1], sys.argv[2], sys.argv[3:]
    figs = []
    for co in cases:
        cat, oid = co.split("/")
        f = f"{dump}/{cat}_{oid}"
        if not os.path.exists(f + ".npz"):
            print("missing", f)
            continue
        d = dict(np.load(f + ".npz"))
        res = json.load(open(f + ".json"))
        fig = plt.figure(figsize=(22, 3.9), dpi=110)
        gs = fig.add_gridspec(1, 6, left=0.02, right=0.995, top=0.82, bottom=0.08, wspace=0.08,
                              width_ratios=[0.8, 1.25, 1, 1, 1, 1])
        row(fig, [gs[0, i] for i in range(6)], d, res, cat, oid)
        out = f"{prefix}_{cat}_{oid}.png"
        fig.savefig(out)
        plt.close(fig)
        figs.append(out)
        print("saved", out)
    if figs:  # stack all rows into one overview image
        from PIL import Image
        ims = [Image.open(p) for p in figs]
        W_ = max(i.width for i in ims)
        canvas = Image.new("RGB", (W_, sum(i.height for i in ims)), "white")
        y = 0
        for i in ims:
            canvas.paste(i, (0, y))
            y += i.height
        canvas.save(f"{prefix}_all.png")
        print("saved", f"{prefix}_all.png")


if __name__ == "__main__":
    main()
