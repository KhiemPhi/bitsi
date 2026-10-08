"""Panda pours using ONLY what its wrist camera saw: look around (BITSI next-best-view), segment the
fused partial cloud, grasp, pour.

1. Scene as in panda_pour.py (HANDAL mesh, CoACD collision pieces, upright on the table). The robot
   starts at HOME with nothing observed.
2. Views: camera poses on the eye-in-hand grid (eyeinhand.py) that have a collision-free IK solution
   with the camera looking at the object; the arm drives there along an RRT-Connect path. At each
   view, MuJoCo renders depth + segmentation from the wrist camera; object pixels (segmentation
   stands in for table-plane and robot removal) are back-projected, given the same depth noise and
   registration error as eyeinhand.observe, and fused (3 mm voxel averaging). The next view is the
   reachable view that sees the most unobserved cells of the surface predicted by BITSI v4 (with the
   gravity-axis candidate, which a robot can use; checked on PartNet-E dev objects, RESULTS.md 21)
   (eyeinhand.bitsi_gain); stop when < 5% of predicted cells are unobserved or after MAX_VIEWS.
3. Every method segments the SAME fused cloud: BITSI v4 (loop child), Find3D ("handle of a <cat>"),
   whole object. Then panda_pour.run plans and executes the grasp + pour for each.
Outputs: per-object JSON line, and with --video a traversal clip plus one clip per method.
usage (graspsim env for MuJoCo; Find3D is called in its own env):
  python eih_pour.py DATA_DIR OID OUT.jsonl [--video VIDEO_DIR]
"""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart import eyeinhand as E
from fitpart.axes import closed_end_sign, revolution_axis, rotation_to_up
from fitpart.scripts.panda_pour import run as pour_run
from fitpart.sim.mujoco_grasp import HANDAL_MASS, convex_parts
from fitpart.sim.panda_scene import PandaScene, TCP_OFFSET, WRIST_BEHIND_TCP

MAX_VIEWS = 6
METHODS = os.environ.get("METHODS", "bitsi4,find3d,whole").split(",")
ROBOT = os.environ.get("ROBOT", "panda")
CAM_OFFSET = np.array([0.06, 0.0, TCP_OFFSET - WRIST_BEHIND_TCP])   # wrist camera in the hand frame
FIND3D_PY = os.path.expanduser("~/miniconda3/envs/find3d/bin/python")


def hand_targets(cam_pos, forward):
    """Hand-frame candidates (tcp position, rotation) putting the wrist camera at cam_pos looking along
    `forward`, for 4 rolls."""
    from fitpart.sim.view_reach import hand_frames
    out = []
    for R in hand_frames(forward):
        tcp = cam_pos + R @ (np.array([0, 0, TCP_OFFSET]) - CAM_OFFSET)
        out.append((tcp, R))
    return out


class Wrist:
    def __init__(self, sc, w=424, h=240):
        import mujoco
        self.sc = sc
        self.rd = mujoco.Renderer(sc.model, h, w)
        self.rd.enable_depth_rendering()
        self.rs = mujoco.Renderer(sc.model, h, w)
        self.rs.enable_segmentation_rendering()
        m = sc.model
        self.cam = m.camera("wrist").id
        self.obj_geoms = np.flatnonzero(m.geom_bodyid == sc.obj)
        self.f = h / 2 / np.tan(np.deg2rad(m.cam_fovy[self.cam]) / 2)
        self.w, self.h = w, h

    def capture(self, rng):
        d = self.sc.data
        self.rd.update_scene(d, camera="wrist")
        depth = self.rd.render().copy()
        self.rs.update_scene(d, camera="wrist")
        seg = self.rs.render()[..., 0].copy()
        mask = np.isin(seg, self.obj_geoms) & (depth > E.MIN_RANGE) & (depth < E.MAX_RANGE)
        v, u = np.nonzero(mask)
        z = depth[v, u]
        z = z + rng.normal(0, 0.0005 + 0.002 * z ** 2)
        x = (u + 0.5 - self.w / 2) / self.f * z
        y = -(v + 0.5 - self.h / 2) / self.f * z
        pos = d.cam_xpos[self.cam].copy()
        Rm = d.cam_xmat[self.cam].reshape(3, 3).copy()
        pts = pos + np.stack([x, y, -z], 1) @ Rm.T
        ang = rng.normal(0, np.deg2rad(0.3), 3)
        K = np.array([[0, -ang[2], ang[1]], [ang[2], 0, -ang[0]], [-ang[1], ang[0], 0]])
        c0 = pts.mean(0) if len(pts) else pos
        pts = (pts - c0) @ (np.eye(3) + K).T + c0 + rng.normal(0, 0.001, 3)
        return pts, pos, depth, mask


def gather(sc, W_full, rng, renderer=None, inset_fn=None, log=print):
    """Drive the wrist camera through BITSI next-best views. Returns fused (W, Wn), the view list,
    and per-step info."""
    from fitpart.bitsi4 import BITSI4
    centre, diag = E.object_frame(W_full)
    dist = E.view_distance(diag)
    wrist = Wrist(sc)
    # reachable views (collision-free IK with the object present)
    reach = {}
    for az in E.AZ:
        for el in E.EL:
            cam, Rc = E.camera_pose(centre, dist, az, el)
            for tcp, R in hand_targets(cam, Rc[2]):
                q = sc.ik(tcp, R, restarts=4, iters=150)
                if q is not None and not sc.in_collision(q):
                    reach[(float(az), float(el))] = q
                    break
    log(f"{len(reach)} reachable views")
    if not reach:
        return None, None, [], []
    F = E.Fused(0.002, voxel=0.003, seed=1)
    v = E.nearest_reachable(*E.HOME, list(reach))
    seq, steps = [], []
    for k in range(1, MAX_VIEWS + 1):
        path = sc.plan_rrt(sc.data.qpos[:7].copy(), reach[v], seed=k)
        if path is None:
            path = [sc.data.qpos[:7].copy(), reach[v]]
        sc.execute(path, vmax=0.6, renderer=renderer, cam="side", every=17)
        pts, cam_pos, depth, mask = wrist.capture(rng)
        F.add(np.zeros(len(pts), int), pts, cam_pos)
        seq.append(v)
        _, Wf, Nf = F.cloud()
        lab, root = BITSI4(up=(0, 0, 1.0)).fit(Wf, Nf)
        pp, pn, obs = E.predicted_surface(root, max(float(np.median(np.linalg.norm(Wf[1:] - Wf[:-1], axis=1))), 0.002))
        unobs = obs == 0
        steps.append(dict(view=v, n_points=len(Wf), unobserved=float(unobs.mean()), depth=depth, mask=mask,
                          W=Wf, labels=lab, pred=pp[unobs], reach=sorted(reach)))
        if inset_fn is not None:
            inset_fn(steps[-1])
        log(f"view {k} {v}: {len(Wf)} fused points, {unobs.mean():.2f} of predicted surface unobserved")
        rest = [r for r in reach if r not in seq]
        if unobs.mean() < 0.05 or not rest:
            break
        gains = [E.bitsi_gain(pp, pn, unobs, *E.camera_pose(centre, dist, *r), Wf, 0.003) for r in rest]
        v = rest[int(np.argmax(gains))]
    _, Wf, Nf = F.cloud()
    return Wf, Nf, seq, steps


def find3d_mask(W, Wn, cat):
    name = cat.replace("_", " ").rstrip("s") if cat != "pots_pans" else "pan"
    with tempfile.TemporaryDirectory() as td:
        np.savez(f"{td}/c.npz", P=W.astype(np.float32), N=Wn.astype(np.float32))
        env = dict(os.environ, PYTHONPATH=os.path.expanduser("~/bitsi/ext/Find3D") + ":" + os.path.expanduser("~/bitsi/ext/shims"))
        subprocess.run([FIND3D_PY, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baselines", "find3d_cloud.py"),
                        f"{td}/c.npz", f"{td}/l.npy", f"handle of a {name};body of a {name}"],
                       check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return np.load(f"{td}/l.npy") == 1


def main():
    D, oid, out = sys.argv[1], sys.argv[2], sys.argv[3]
    vdir = sys.argv[sys.argv.index("--video") + 1] if "--video" in sys.argv else None
    dump = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None
    z = np.load(f"{D}/{oid}.npz", allow_pickle=True)
    P, N, handle = z["P"].astype(float), z["N"].astype(float), z["handle"]
    cat = str(z["cat"]) if "cat" in z.files else "mug"
    from fitpart.scripts.panda_pour import make_scene
    sc = make_scene(ROBOT, convex_parts(str(z["mesh"])), HANDAL_MASS.get(cat, 0.35))
    a, c, _ = revolution_axis(P, N)
    a = a * closed_end_sign(P, a, c)
    t, R = sc.reset(rotation_to_up(a))
    W_full = P @ R.T + t
    rng = np.random.default_rng(int(oid.split("_")[-1]))
    renderer = None
    frames_insets = []
    if vdir:
        import mujoco
        os.makedirs(vdir, exist_ok=True)
        renderer = mujoco.Renderer(sc.model, 720, 960)
        sc.frames = []
    W, Wn, seq, steps = gather(sc, W_full, rng, renderer=renderer,
                               inset_fn=(lambda st: frames_insets.append((len(sc.frames), st))) if vdir else None)
    res = dict(oid=oid, cat=cat, views=seq, n_views=len(seq),
               unobserved=[s["unobserved"] for s in steps], n_points=0 if W is None else len(W))
    if W is None:
        res["error"] = "no reachable view"
    else:
        from scipy.spatial import cKDTree
        gt_handle = handle[cKDTree(W_full).query(W)[1]]          # diagnostics only
        res["handle_seen_frac"] = float(np.mean(cKDTree(W).query(W_full[handle])[0] < 0.004))
        if vdir:
            write_traversal(sc.frames, frames_insets, f"{vdir}/{cat}_{oid}_traverse.mp4", cat, oid)
        sel = {"find3d": find3d_mask(W, Wn, cat)}
        for m in METHODS:
            r = pour_run(D, oid, m, video=f"{vdir}/{cat}_{oid}_{m}.mp4" if vdir else None, cloud=(W, Wn), sel=sel, robot=ROBOT)
            r.pop("video", None)
            if m != "whole":
                from fitpart.scripts.panda_pour import part_mask
                msk = part_mask(m, W, Wn, sel)[0]
                inter = (msk & gt_handle).sum()
                r["part_iou_with_handle"] = float(inter / max((msk | gt_handle).sum(), 1))
            res[m] = r
    if dump and W is not None:
        from fitpart.scripts.panda_pour import part_mask
        from scipy.spatial import cKDTree
        os.makedirs(dump, exist_ok=True)
        d = {"W": W, "Wn": Wn, "W_full": W_full, "handle_full": handle,
             "gt_handle": handle[cKDTree(W_full).query(W)[1]], "views": np.array(seq),
             "depth": np.stack([s_["depth"] for s_ in steps]), "mask": np.stack([s_["mask"] for s_ in steps]),
             "unobserved": np.array([s_["unobserved"] for s_ in steps]), "reach": np.array(steps[0]["reach"]),
             "n_points_per_view": np.array([s_["n_points"] for s_ in steps])}
        for m in [x for x in METHODS if x != "whole"]:
            d[f"part_{m}"] = part_mask(m, W, Wn, sel)[0]
        for m in METHODS:
            if "grasp" in res[m]:
                d[f"grasp_{m}"] = np.array(res[m]["grasp"])
        np.savez_compressed(f"{dump}/{cat}_{oid}.npz", **d)
        json.dump(res, open(f"{dump}/{cat}_{oid}.json", "w"), default=float)
    with open(out, "a") as f:
        f.write(json.dumps(res, default=float) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("success", "task_success", "planned") if kk in v})
                      for k, v in res.items()}, default=float))


def write_traversal(frames, insets, path, cat, oid):
    """Side-camera frames of the look-around with, after each view, an inset of the fused cloud
    (coloured by BITSI v4 segment) and the predicted-but-unobserved surface (red)."""
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    imgs = []
    for k, (_, st) in enumerate(insets):
        fig = plt.figure(figsize=(3.6, 3.1), dpi=100)
        ax = fig.add_subplot(projection="3d")
        Wf, lab = st["W"], st["labels"]
        ax.scatter(*Wf[::2].T, c=lab[::2] % 10, cmap="tab10", s=1, vmin=0, vmax=9)
        if len(st["pred"]):
            ax.scatter(*st["pred"].T, c="red", s=3)
        ax.set_axis_off()
        ax.set_box_aspect(np.ptp(Wf, 0) + 1e-3)
        ax.set_title(f"view {k + 1}: {len(Wf)} points\nred = predicted, not yet seen ({st['unobserved']:.0%})", fontsize=8)
        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    deps = []
    for k, (_, st) in enumerate(insets):  # what the wrist camera saw at view k (object pixels coloured)
        dep, msk = st["depth"], st["mask"]
        g = np.full(dep.shape + (3,), 235, np.uint8)
        valid = (dep > 0) & (dep < 1.5)
        if valid.any():
            lo, hi = np.percentile(dep[valid], [2, 98])
            t = np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1)
            g[valid] = (140 + 90 * t[valid])[:, None].astype(np.uint8)
        if msk.any():
            lo, hi = dep[msk].min(), dep[msk].max()
            t = np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1)
            g[msk] = (255 * plt.cm.viridis(1 - t[msk])[:, :3]).astype(np.uint8)
        im = Image.fromarray(g).resize((340, int(340 * dep.shape[0] / dep.shape[1])))
        lab = Image.new("RGB", (im.width, im.height + 22), (255, 255, 255))
        lab.paste(im, (0, 22))
        az, el = st["view"]
        ImageDraw.Draw(lab).text((4, 4), f"wrist camera, view {k + 1} (az {az:.0f}, el {el:.0f})", fill=(20, 20, 20))
        deps.append(lab)
    out = []
    starts = [i for i, _ in insets]
    for i, fr in enumerate(frames):
        im = Image.fromarray(fr)
        j = int(np.searchsorted(starts, i, side="right")) - 1
        if j >= 0:
            im.paste(Image.fromarray(imgs[j]), (im.width - 370, 10))
            im.paste(deps[j], (im.width - deps[j].width - 10, im.height - deps[j].height - 10))
        ImageDraw.Draw(im).text((12, 12), f"{cat} {oid}: looking around with the wrist camera (BITSI v4 next-best-view)",
                                fill=(20, 20, 20))
        out.append(np.asarray(im))
    for _ in range(45):
        out.append(out[-1])
    imageio.mimsave(path, out, fps=30, macro_block_size=1)


if __name__ == "__main__":
    main()
