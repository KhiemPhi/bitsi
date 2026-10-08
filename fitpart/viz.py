"""Quick matplotlib views of segmentations (debugging and qualitative figures)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def show_segmentations(points, labelings, titles, path, views=((20, 30), (20, 120))):
    n = len(labelings)
    fig = plt.figure(figsize=(3.2 * n, 3.0 * len(views)))
    cmap = plt.get_cmap("tab20")
    for r, (el, az) in enumerate(views):
        for c, (lab, title) in enumerate(zip(labelings, titles)):
            ax = fig.add_subplot(len(views), n, r * n + c + 1, projection="3d")
            if lab.dtype.kind == "f":
                col = plt.get_cmap("viridis")(lab)
            else:
                _, inv = np.unique(lab, return_inverse=True)
                col = cmap(inv % 20)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=col, s=2)
            ax.view_init(el, az)
            ax.set_axis_off()
            m = np.ptp(points, 0).max() / 2
            ctr = points.mean(0)
            ax.set_xlim(ctr[0] - m, ctr[0] + m); ax.set_ylim(ctr[1] - m, ctr[1] + m); ax.set_zlim(ctr[2] - m, ctr[2] + m)
            if r == 0:
                ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
