"""Diagnostic figures (env: graspsim): coloured decompositions + suggested grasps + Reeb slicing.

For a prepared HANDAL mug (pour_prepare.py), in its upright simulator pose:
  fig 1  four panels (two views each): GT handle/body; BITSI v0 segments + its top-3 tau grasps;
         Reeb arcs (loop arcs outlined) ; FitPart + SAM 3 "handle" + its top-3 tau grasps.
         Grasps are drawn as gripper glyphs coloured by their simulated pour outcome.
  fig 2  Reeb diagnostic: pieces per slab for each PCA sweep direction, and the slice stack
         (sweep coordinate vs cross-section coordinate, coloured by piece/arc) for each direction.
usage: python viz_decomposition.py POUR_DIR OID OUT_PREFIX
"""
import os, sys
sys.path.insert(0, '/home/khiem/bitsi')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fitpart.profiles import pca_axes, sweep_frame
from fitpart.reeb import arcs_and_loops, reeb_graph, reeb_segmentation, slice_pieces
from fitpart.sim.mujoco_grasp import GraspSim, convex_parts
from fitpart.sim.pour_task import pour_screw, try_pour, upright_pose_revolution
from fitpart.sim.screw_metric import screw_tau

INK, MUTED = '#1f1f1e', '#6b6a63'
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5af5', '#8c6d46', '#00a3c4', '#c4003a']


def colors_for(lab):
    _, inv = np.unique(lab, return_inverse=True)
    return np.array([PAL[i % len(PAL)] for i in inv])


def glyph(ax, c1, c2, a, color):
    x = (c2 - c1) / np.linalg.norm(c2 - c1)
    z = a / np.linalg.norm(a)
    z = z - (z @ x) * x
    z /= np.linalg.norm(z)
    o = 0.006
    f1a, f2a = c1 - x * o, c2 + x * o            # pad faces just outside the contacts
    f1b, f2b = f1a - z * 0.045, f2a - z * 0.045   # back of the fingers
    pts = np.array([f1a, f1b, f2b, f2a])
    ax.plot(*pts.T, color=color, lw=2.2)
    ax.plot(*np.array([0.5 * (f1b + f2b), 0.5 * (f1b + f2b) - z * 0.03]).T, color=color, lw=2.2)


def panel(fig, pos, W, cols, title, view, grasps=(), outcomes=(), mask_alpha=None):
    ax = fig.add_subplot(*pos, projection='3d')
    ax.scatter(*W[::2].T, c=cols[::2], s=7, alpha=0.9 if mask_alpha is None else mask_alpha[::2], linewidths=0, depthshade=False)
    for (c1, c2, a), ok in zip(grasps, outcomes):
        glyph(ax, c1, c2, a, '#008300' if ok else '#c4003a')
    ctr = W.mean(0); r = np.ptp(W, 0).max() * 0.6
    ax.set_xlim(ctr[0] - r, ctr[0] + r); ax.set_ylim(ctr[1] - r, ctr[1] + r); ax.set_zlim(ctr[2] - r * 0.6, ctr[2] + r)
    ax.view_init(*view); ax.set_axis_off()
    ax.set_title(title, fontsize=9, color=INK)
    return ax


def top_grasps(sim, Wp, Wn, mask, com, axis, point, mass, k=3, per_segment=None, across=False):
    from fitpart.grasping import sample_grasps
    regions = [mask] if per_segment is None else [per_segment == l for l in np.unique(per_segment)]
    gs = []
    for m in regions:
        if m.sum() < 10:
            continue
        g = sample_grasps(Wp[m], Wn[m], Wp, n_contacts=2000 if across else 640,
                          max_grasps=80 if per_segment is None else 30, seed=0, across_part=across)
        gs += [x for x in g if sim.clearance_ok(x[0], x[1], x[2] / np.linalg.norm(x[2]))]
    taus = [screw_tau(g[0], g[1], com, mass, axis, point) for g in gs]
    order = np.argsort(-np.nan_to_num(taus, nan=-1e9))
    return [gs[i] for i in order[:k]], [taus[i] for i in order[:k]]


def main():
    D, oid, out = sys.argv[1], sys.argv[2], sys.argv[3]
    z = np.load(f'{D}/{oid}.npz', allow_pickle=True)
    P, N, handle = z['P'].astype(float), z['N'].astype(float), z['handle']
    sel = np.load(f'{D}/{oid}.sel.npz'); bitsi = np.load(f'{D}/{oid}.bitsi.npy')
    sim = GraspSim(convex_parts(str(z['mesh'])), mass=0.35)
    R0, (t, R), _ = upright_pose_revolution(sim, P, N)  # label-free: symmetry axis, closed end down
    W, Wn = P @ R.T + t, N @ R.T
    axis, point = pour_screw(W, handle)
    com = W.mean(0)
    # Reeb segmentation on the upright cloud (same settings as the mug test)
    from fitpart.axes import revolution_axis, closed_end_sign
    ax_b, c_b, _ = revolution_axis(W, Wn)
    ax_b = ax_b * closed_end_sign(W, ax_b, c_b)
    reeb_lab, info = reeb_segmentation(W, d=ax_b, auto_scale=True)
    rr = f'{D}/{oid}.revres.npy'
    revres = np.load(rr) if os.path.exists(rr) else None
    rows = []
    cands = [('BITSI v0 segments', np.ones(len(W), bool), bitsi, False),
             ('FitPart + SAM 3 "handle"', sel['fitpart'], None, True)]
    if revres is not None:
        cands.append(('revolution residual', revres, None, True))
    for name, mask, seg, across in cands:
        gs, taus = top_grasps(sim, W, Wn, mask, com, axis, point, 0.35, per_segment=seg, across=across)
        oks = []
        for c1, c2, a in gs:
            sim.reset_and_settle(R0)
            oks.append(try_pour(sim, c1, c2, a, axis, point)[0])
        rows.append((name, gs, oks, taus))
    views = [(20, 45), (20, 225)]
    fig = plt.figure(figsize=(16, 8.5))
    gt_cols = np.where(handle, '#eb6834', '#a8a79f')
    bitsi_cols = colors_for(bitsi)
    loop = np.isin(info['arc_of_point'], info['loop_arcs'])
    reeb_cols = colors_for(reeb_lab)
    fit_cols = np.where(sel['fitpart'], '#2a78d6', '#d6d5ce')
    specs = [('Ground truth: handle (orange) / body', gt_cols, (), (), None),
             (f'Your BITSI v0: {bitsi.max()+1} segments + top-3 tau grasps', bitsi_cols, rows[0][1], rows[0][2], None),
             (('Surface-of-revolution residual (purple), axis in black: '
               f'handle IoU {(revres & handle).sum() / max((revres | handle).sum(), 1):.2f}') if revres is not None else
              f'Reeb arcs: {reeb_lab.max()+1} parts, {len(info["loops"])} loops', 
              np.where(revres, '#7a5af5', '#d6d5ce') if revres is not None else reeb_cols,
              rows[2][1] if revres is not None else (), rows[2][2] if revres is not None else (), None),
             ('FitPart + SAM 3 "handle" (blue) + top-3 tau grasps', fit_cols, rows[1][1], rows[1][2], None)]
    for j, (title, cols, gs, oks, alpha) in enumerate(specs):
        for i, v in enumerate(views):
            pa = panel(fig, (2, 4, i * 4 + j + 1), W, cols, title if i == 0 else '', v, gs, oks, alpha)
            if j == 2:
                s_ = (W - c_b) @ ax_b
                pa.plot(*np.array([c_b + ax_b * (s_.min() - 0.02), c_b + ax_b * (s_.max() + 0.02)]).T, color=INK, lw=2)
    fig.suptitle(f'HANDAL mug {oid}, label-free upright (symmetry axis, closed end down). Grasp glyphs: green = pour succeeded in simulation, '
                 f'red = failed. Pour axis through the rim opposite the handle.', fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out + '_decomposition.png', dpi=110)
    plt.close(fig)
    # Reeb diagnostic
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
    for k, d in enumerate([ax_b] + list(pca_axes(W))[:2]):
        piece_of, pieces, slab = slice_pieces(W, d, **__import__('fitpart.reeb', fromlist=['x']).choose_scale(W, d))
        edges = reeb_graph(pieces)
        arc, na, loops, G = arcs_and_loops(pieces, edges)
        S = slab.max() + 1
        per = np.bincount([p['slab'] for p in pieces], minlength=S)
        ax = axes[0, k]
        ax.plot(np.arange(S), per, color='#2a78d6', lw=1.5)
        hs = np.bincount(slab[handle], minlength=S) > 0
        ax.fill_between(np.arange(S), 0, per.max() + 0.5, where=hs, color='#eb6834', alpha=0.12, label='slabs containing GT handle')
        tilt = np.degrees(np.arccos(abs(d @ np.array([0, 0, 1.0]))))
        ax.set_title(f'{"symmetry axis" if k == 0 else f"PCA axis {k}"} ({tilt:.0f} deg from vertical): pieces per slab, {len(loops)} raw loops',
                     fontsize=9, color=INK)
        ax.set_xlabel('slab along sweep', color=MUTED); ax.set_ylabel('pieces', color=MUTED)
        ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=7, frameon=False)
        Rf = sweep_frame(W, d)
        q = (W - W.mean(0)) @ Rf.T
        ax = axes[1, k]
        a_of = np.where(piece_of >= 0, arc[np.maximum(piece_of, 0)], -1)
        cols = np.where(a_of >= 0, np.array([PAL[i % len(PAL)] for i in np.maximum(a_of, 0)]), '#d6d5ce')
        ax.scatter(q[:, 0], q[:, 1], c=cols, s=5, linewidths=0)
        ax.scatter(q[handle][::3, 0], q[handle][::3, 1], s=14, facecolors='none', edgecolors=INK, linewidths=0.3,
                   label='GT handle points')
        ax.set_title('slice stack: sweep coordinate vs cross-section coordinate, colour = Reeb arc', fontsize=9, color=INK)
        ax.set_xlabel('sweep coordinate (m)', color=MUTED); ax.set_ylabel('cross-section u (m)', color=MUTED)
        ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=7, frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(out + '_reeb.png', dpi=110)
    print('saved', out + '_decomposition.png', out + '_reeb.png',
          '| BITSI top-3 outcomes', rows[0][2], 'taus', np.round(rows[0][3], 2),
          '| FitPart top-3 outcomes', rows[1][2], 'taus', np.round(rows[1][3], 2), '| residual', rows[-1][2], np.round(rows[-1][3], 2))


if __name__ == '__main__':
    main()
