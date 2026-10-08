"""Franka Panda pours from a HANDAL mug: segment -> choose grasp on the part -> plan -> execute.

Task: lift the mug 10 cm and rotate it 90 deg about the rim axis opposite the handle. No labels are
used: each method's screw comes from its own output (pour_screw_from: opposite its part, or opposite
its grasp for the whole-object baseline); the GT handle only feeds a diagnostic angle in the log.
Methods differ only in the part they grasp:
  bitsi2   BITSI v2 child primitive that closes a loop (largest; else largest child), no labels
  fitpart  FitPart + SAM 3 "handle" selection (precomputed in OID.sel.npz)
  sam3     SAM 3 lifted directly to points (OID.sel.npz)
  whole    any point of the object
Every method gets the same two samplers on its part (across-the-part grasps and plain antipodal
grasps); candidates are ranked by the ITSI screw metric tau under that method's screw and the first one with a collision-free IK solution and plan
is executed:
  home --RRT-Connect--> pre-grasp (10 cm back along the approach) --straight line--> grasp
  --close--> lift 10 cm (straight line) --screw path--> pour 90 deg --hold 1 s
Success: final object pose within 20 deg / 3 cm of the commanded screw motion and held by both
fingers. Optionally writes a video (side camera + segmentation inset + phase captions).

usage: python panda_pour.py POUR_DIR OID METHODS(comma) [VIDEO.mp4]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.axes import closed_end_sign, revolution_axis, rotation_to_up
from fitpart.bitsi2 import BITSI2
from fitpart.grasping import sample_grasps
from fitpart.sim.mujoco_grasp import axis_angle, convex_parts
from fitpart.sim.panda_scene import HOME, PandaScene, grasp_frame
from fitpart.sim.pour_task import pour_screw_from
from fitpart.sim.screw_metric import screw_tau

LABEL = {"bitsi5g": "BITSI v5, gripper-wrap axis (ours)", "bitsi4": "BITSI v4 (ours): loop child primitive", "find3d": 'Find3D "handle of a <category>"',
         "bitsi3": "BITSI v3 (ours): loop child primitive", "bitsi2": "BITSI v2 (ours): loop child primitive", "fitpart": 'FitPart + SAM 3 "handle"',
         "sam3": 'SAM 3 direct "handle"', "whole": "Whole object (no segmentation)"}


def part_mask(method, W, Wn, sel):
    if method == "whole":
        return np.ones(len(W), bool), None
    if method in ("fitpart", "sam3"):
        return sel["fitpart" if method == "fitpart" else "sam3_direct"].astype(bool), None
    if method == "find3d":
        return np.asarray(sel["find3d"]).astype(bool), None
    if method == "bitsi5g":  # gripper-wrap root axis (RESULTS.md 25); part rule fixed before running
        from fitpart.bitsi5 import BITSI5
        from fitpart.feasibility import obb_extents
        lab, root = BITSI5(axis_rule="gripper", spine_coding="free", gripper_opening=0.08, up=(0, 0, 1.0)).fit(W, Wn)
        loops = [c for c in root.children if c.loop]
        m = np.zeros(len(W), bool)
        if loops:
            m[max(loops, key=lambda c: len(c.idx)).idx] = True
            return m, root
        segs, cnt = np.unique(lab, return_counts=True)
        cands = [s_ for s_, n in zip(segs, cnt) if s_ != segs[np.argmax(cnt)] and n >= 0.02 * len(W)]
        if not cands:
            return np.ones(len(W), bool), root
        best = min(cands, key=lambda s_: obb_extents(W[lab == s_])[2])
        return lab == best, root
    if method == "bitsi4":   # robot setting: gravity (table normal) offered as a root-axis candidate
        from fitpart.bitsi4 import BITSI4
        lab, root = BITSI4(up=(0, 0, 1.0)).fit(W, Wn)
    elif method == "bitsi3":
        from fitpart.bitsi3 import BITSI3
        lab, root = BITSI3().fit(W, Wn)
    else:
        lab, root = BITSI2().fit(W, Wn)
    if not root.children:
        return np.ones(len(W), bool), root
    loops = [c for c in root.children if c.loop]
    ch = max(loops or root.children, key=lambda c: len(c.idx))
    m = np.zeros(len(W), bool)
    m[ch.idx] = True
    return m, root


def inset(W, mask, grasp, title, size=(360, 310)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    sub = slice(None, None, 3)
    cols = np.where(mask, "#7a5af5", "#c9c8c0")
    ax.scatter(*W[sub].T, c=cols[sub], s=3, linewidths=0, depthshade=False)
    if grasp is not None:
        c1, c2, _ = grasp
        ax.plot(*np.stack([c1, c2]).T, color="#c4003a", lw=3)
        ax.scatter(*np.stack([c1, c2]).T, color="#c4003a", s=25)
    ctr = W.mean(0); r = np.ptp(W, 0).max() * 0.6
    ax.set_xlim(ctr[0] - r, ctr[0] + r); ax.set_ylim(ctr[1] - r, ctr[1] + r); ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.view_init(20, -60); ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def inset2(W, mask, gs, taus, chosen, screw, title, size=(800, 330), save=None):
    """Two panels: (left) the segmented cloud, the method's part in purple; (right) the grasp
    heatmap = per surface point, the best tau of any candidate grasp with a contact there (one hue,
    light = low, dark = high; grey = no candidate), the pour screw axis with its rotation sense,
    and the chosen grasp (black contacts + jaw line)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.spatial import cKDTree
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    sub = slice(None, None, 3)
    ctr = W.mean(0)
    r = np.ptp(W, 0).max() * 0.42
    heat = np.full(len(W), np.nan)
    ok = [k for k, tv in enumerate(taus) if tv > -1e8] if len(gs) else []
    if ok:
        tree = cKDTree(W)
        for k in ok:
            for cpt in gs[k][:2]:
                for j in tree.query_ball_point(cpt, 0.006):
                    heat[j] = taus[k] if np.isnan(heat[j]) else max(heat[j], taus[k])
    for pnl in (0, 1):
        ax = fig.add_subplot(1, 2, pnl + 1, projection="3d")
        if pnl == 0:
            cols = np.where(mask, "#7a5af5", "#c9c8c0")
            ax.scatter(*W[sub].T, c=cols[sub], s=6, linewidths=0, depthshade=False)
            ax.set_title(title, fontsize=11)
        else:
            has = ~np.isnan(heat)
            ax.scatter(*W[~has][sub].T, c="#dedcd3", s=6, linewidths=0, depthshade=False)
            if has.any():
                from matplotlib.colors import LinearSegmentedColormap, Normalize
                heat_cmap = LinearSegmentedColormap.from_list(
                    "tau_heat", ["#1a9850", "#fee08b", "#fc8d59", "#d73027"])  # green -> yellow -> orange -> red (best)
                hv = heat[has]
                norm = Normalize(vmin=float(hv.min()), vmax=float(hv.max()) if hv.max() > hv.min() else float(hv.min()) + 1e-6)
                o = np.argsort(hv)                                      # best drawn last, on top
                ax.scatter(*W[has][o].T, c=heat_cmap(norm(hv[o])), s=12, linewidths=0, depthshade=False)
                cax = fig.add_axes([0.865, 0.14, 0.018, 0.6])
                cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=heat_cmap), cax=cax)
                cb.set_label("\u03c4 (grasp quality\nunder the pour screw)", fontsize=9)
                cb.ax.tick_params(labelsize=8)
                cb.outline.set_visible(False)
            if screw is not None:
                # drawn where it acts relative to the mug: at the rim (the commanded screw is the
                # same axis after the 10 cm lift)
                axis, point = screw[0] / np.linalg.norm(screw[0]), screw[1] - np.array([0, 0, 0.10])
                L = 0.75 * r                      # fits inside the zoomed panel
                tail, tip = point - axis * L, point + axis * L
                ax.plot(*np.stack([tail, tip]).T, color="#c4003a", lw=2.5)
                ax.quiver(*(tip - axis * 0.3 * L), *axis, color="#c4003a", length=0.3 * L, normalize=True,
                          arrow_length_ratio=0.8, lw=3.5)
                ax.text(*(tip + axis * 0.08 * L), "\u03c9", color="#c4003a", fontsize=15, fontweight="bold")
                ax.scatter(*point, color="#c4003a", s=30)
                ax.text(*(point + np.array([0, 0, -0.018])), "q", color="#c4003a", fontsize=11)
                u = np.cross(axis, [0, 0, 1.0]); u = u / (np.linalg.norm(u) + 1e-12); v2 = np.cross(axis, u)
                th = np.linspace(0, 1.4 * np.pi, 30)
                arc = point + axis * L * 0.45 + 0.25 * L * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v2))
                ax.plot(*arc.T, color="#c4003a", lw=1.8)
                d = arc[-1] - arc[-3]
                ax.quiver(*arc[-1], *d, color="#c4003a", length=0.12 * L, normalize=True, arrow_length_ratio=1.0, lw=1.8)
            if chosen is not None:
                c1, c2 = chosen[0], chosen[1]
                ax.plot(*np.stack([c1, c2]).T, color="#2a78d6", lw=3.5)
                ax.scatter(*np.stack([c1, c2]).T, color="#2a78d6", s=80, edgecolors="white", linewidths=1.5)
            ax.set_title("grasp heatmap (red = best \u03c4), chosen grasp (blue)\npour screw: axis \u03c9 through q, curl = + rotation"
                         if ok else "no grasp candidates on this part", fontsize=10)
        ax.set_xlim(ctr[0] - r, ctr[0] + r); ax.set_ylim(ctr[1] - r, ctr[1] + r); ax.set_zlim(ctr[2] - r, ctr[2] + r)
        if screw is not None:   # omega side-on, handle visible: view from the handle side, from above
            wv = screw[0] / np.linalg.norm(screw[0])
            hv = np.cross([0, 0, 1.0], wv)
            ax.view_init(28, np.degrees(np.arctan2(hv[1], hv[0])) + 35)
        else:
            ax.view_init(20, -60)
        ax.set_axis_off()
    fig.subplots_adjust(left=0, right=0.86, bottom=0, top=0.84, wspace=0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    if save:
        fig.savefig(save, dpi=200)
    plt.close(fig)
    return img


def make_scene(robot, parts, mass):
    if robot == "kinova":
        from fitpart.sim.kinova_scene import KinovaScene
        return KinovaScene(parts, mass=mass)
    return PandaScene(parts, mass=mass)


def run(D, oid, method, video=None, max_candidates=120, seed=0, force=False, replay=False, cloud=None, sel=None,
        robot="panda"):
    """cloud = (W, Wn) in the world frame of this (deterministic) scene, e.g. fused from the robot's
    own wrist-camera views (scripts/eih_pour.py); default: the full mesh samples. sel: per-point
    masks of `cloud` for precomputed methods."""
    import mujoco
    z = np.load(f"{D}/{oid}.npz", allow_pickle=True)
    P, N, handle = z["P"].astype(float), z["N"].astype(float), z["handle"]
    if sel is None:
        sel = np.load(f"{D}/{oid}.sel.npz") if os.path.exists(f"{D}/{oid}.sel.npz") else None
    from fitpart.sim.mujoco_grasp import HANDAL_MASS
    cat = str(z["cat"]) if "cat" in z.files else "mug"
    mass = HANDAL_MASS.get(cat, 0.35)
    sc = make_scene(robot, convex_parts(str(z["mesh"])), mass)
    a, c, _ = revolution_axis(P, N)
    a = a * closed_end_sign(P, a, c)
    R0 = rotation_to_up(a)
    t, R = sc.reset(R0)
    W_full = P @ R.T + t
    W, Wn = (W_full, N @ R.T) if cloud is None else (np.asarray(cloud[0], float), np.asarray(cloud[1], float))
    mask, root = part_mask(method, W, Wn, sel)
    com = W.mean(0)
    # Every method gets BOTH samplers: across-the-part grasps and plain antipodal grasps on its part
    gs = []
    if mask.sum() >= 10:
        for across in (True, False):
            gs += sample_grasps(W[mask], Wn[mask], W, n_contacts=2000 if across else 640, max_grasps=120,
                                seed=seed, across_part=across)
    # The task screw comes from the method's OWN output (no labels): pour over the rim opposite the
    # method's part (its handle estimate); a method with no part pours opposite its grasp.
    screws = []
    for g in gs:
        if method == "whole" or mask.all():
            screws.append(pour_screw_from(W, 0.5 * (g[0] + g[1])))
        else:
            screws.append(pour_screw_from(W, W[mask].mean(0), ~mask))
    taus = np.array([np.nan_to_num(screw_tau(g[0], g[1], com, mass, sw[0], sw[1]), nan=-1e9) for g, sw in zip(gs, screws)])
    order = np.argsort(-taus)
    # diagnostic only (not used by any method): angle between the pour side and the GT handle side
    hg = W_full[handle].mean(0) - W_full[~handle].mean(0)
    hg[2] = 0
    hg /= np.linalg.norm(hg)
    renderer = mujoco.Renderer(sc.model, 720, 960) if video else None
    caption = {"text": ""}
    def make_plan(i, strict=True):
        """Plan grasp candidate i. strict=False (forced mode, for showing a method that has NO
        feasible plan): collision checks are skipped and straight joint moves replace failed
        planning, so the robot executes the method's best grasp and the failure becomes visible."""
        c1, c2, app = gs[i]
        axis, point, hside = screws[i]
        Rg = grasp_frame(c1, c2, app)
        mid = 0.5 * (c1 + c2)
        zax = Rg[:, 2]
        q_g = sc.ik(mid, Rg, restarts=6, seed=seed)
        if q_g is None or (strict and sc.in_collision(q_g)):  # fingers open at the grasp pose: no contact yet
            return None
        q_pre = sc.ik(mid - zax * 0.10, Rg, q0=q_g, restarts=0)
        if q_pre is None or (strict and sc.in_collision(q_pre)):
            return None
        approach = sc.cartesian(q_pre, [(mid - zax * 0.10 * (1 - s), Rg) for s in np.linspace(0.1, 1, 10)])
        if approach is None or (strict and not sc.path_free(approach)):
            if strict:
                return None
            approach = [q_pre, q_g]
        lift = sc.cartesian(q_g, [(mid + np.array([0, 0, 0.10 * s]), Rg) for s in np.linspace(0.1, 1, 10)])
        if lift is None:
            if strict:
                return None
            lift = [q_g]
        g1 = mid + np.array([0, 0, 0.10])
        pour = sc.cartesian(lift[-1], [(axis_angle(axis, th) @ (g1 - point) + point, axis_angle(axis, th) @ Rg)
                                       for th in np.linspace(0, np.pi / 2, 31)[1:]])
        if pour is None:
            if strict:
                return None
            pour = [lift[-1]]
        to_pre = sc.plan_rrt(sc.data.qpos[:7].copy(), q_pre, seed=seed)
        if to_pre is None:
            if strict:
                return None
            to_pre = [sc.data.qpos[:7].copy(), q_pre]
        return dict(i=int(i), grasp=gs[i], tau=float(taus[i]), to_pre=to_pre, approach=approach, lift=lift, pour=pour,
                    axis=axis, point=point, pour_vs_gt_handle_deg=float(np.degrees(np.arccos(np.clip(hside @ hg, -1, 1)))),
                    forced=not strict)

    plan = None
    for i in order[:max_candidates]:
        plan = make_plan(i)
        if plan is not None:
            break
    planned = plan is not None
    if plan is None and replay and gs and video:
        return replay_until_failure(sc, W, mask, gs, order, screws, method, video, renderer, len(order[:max_candidates]),
                                    dict(oid=oid, cat=cat, method=method, part_pts=int(mask.sum()), n_grasps=len(gs),
                                         planned=False, success=False), seed, taus)
    if plan is None and force:
        for i in order[:max_candidates]:
            plan = make_plan(i, strict=False)
            if plan is not None:
                break
    result = dict(oid=oid, cat=cat, method=method, part_pts=int(mask.sum()), n_grasps=len(gs), planned=planned,
                  forced=bool(plan is not None and plan["forced"]))
    sc.frames = []
    if plan is None:
        result.update(success=False, reason="no feasible grasp + plan" if gs else "no grasp candidates on the part")
        if video:  # nothing to execute: hold on the scene, the part inset and the reason
            hdr = inset2(W, mask, gs, taus, None, None, LABEL[method],
                         save=video.replace(".mp4", "_inset.png"))
            caption["text"] = (f"FAILED: {'no grasp candidates on this part' if not gs else 'no reachable grasp on this part'}"
                               f" ({int(mask.sum())} points) - the robot cannot start")
            for _ in range(90):
                renderer.update_scene(sc.data, camera="side")
                sc.frames.append(renderer.render().copy())
            _caption_frames(sc, caption, hdr)
    else:
        hdr = inset2(W, mask, gs, taus, plan["grasp"], (plan["axis"], plan["point"]), LABEL[method],
                     save=video.replace(".mp4", "_inset.png")) if video else None
        def rec(txt):
            caption["text"] = txt
        sc._caption = caption
        sc.set_screw_overlay(plan["axis"], plan["point"])
        cam = sc.camera_for(plan["axis"])
        pre = "[no collision-free plan exists; executing its best grasp anyway] " if plan["forced"] else ""
        rec(pre + "1. move to pre-grasp" + ("" if plan["forced"] else " (RRT-Connect)"))
        sc.execute(plan["to_pre"], renderer=renderer, every=17, cam=cam)
        _caption_frames(sc, caption, hdr)
        rec(pre + "2. approach along the grasp axis")
        sc.execute(plan["approach"], vmax=0.4, renderer=renderer, cam=cam)
        _caption_frames(sc, caption, hdr)
        if os.environ.get("POUR_DEBUG"):
            p_now, _ = sc.fk(sc.data.qpos[:7].copy())
            print("DEBUG tcp after approach", np.round(p_now, 4), "target", np.round(0.5 * (plan["grasp"][0] + plan["grasp"][1]), 4),
                  "err mm", round(1000 * float(np.linalg.norm(p_now - 0.5 * (plan["grasp"][0] + plan["grasp"][1]))), 1),
                  "qpos-ctrl max", round(float(np.abs(sc.data.qpos[:7] - sc.data.ctrl[:7]).max()), 4), flush=True)
            m_, d_ = sc.model, sc.data
            print("DEBUG qpos-ctrl per joint", np.round(d_.qpos[:7] - d_.ctrl[:7], 3), "act force", np.round(d_.actuator_force[:7], 1))
            print("DEBUG contacts", sorted({(m_.body(m_.geom_bodyid[c.geom1]).name, m_.body(m_.geom_bodyid[c.geom2]).name) for c in d_.contact[:d_.ncon]}))
            print("DEBUG plan to_pre end vs q_pre", np.round(plan["to_pre"][-1], 3), "approach end", np.round(plan["approach"][-1], 3))
        rec(pre + "3. close the gripper")
        sc.set_gripper(0, renderer=renderer, cam=cam)
        _caption_frames(sc, caption, hdr)
        if os.environ.get("POUR_DEBUG"):
            print("DEBUG after close: grip ctrl", sc.data.ctrl[sc.grip_ctrl], "touch", sc.fingers_touch_object(),
                  "contacts with obj", sum(1 for i in range(sc.data.ncon) if sc.obj in (sc.model.geom_bodyid[sc.data.contact[i].geom1], sc.model.geom_bodyid[sc.data.contact[i].geom2])), flush=True)
        t0, Rob0 = sc.obj_pose()
        result_close = dict(held_after_close=bool(sc.fingers_touch_object()),
                            moved_before_close_cm=float(100 * np.linalg.norm(t0 - t)))
        rec(pre + "4. lift 10 cm")
        sc.execute(plan["lift"], gripper=0, vmax=0.4, renderer=renderer, cam=cam)
        _caption_frames(sc, caption, hdr)
        rec(pre + "5. pour: rotate 90 deg about the rim screw axis")
        sc.execute(plan["pour"], gripper=0, vmax=0.5, settle=1.0, renderer=renderer, cam=cam)
        _caption_frames(sc, caption, hdr)
        t1, R1 = sc.obj_pose()
        axis, point = plan["axis"], plan["point"]
        Rs = axis_angle(axis, np.pi / 2)
        R_exp = Rs @ Rob0
        t_exp = Rs @ (t0 + np.array([0, 0, 0.10]) - point) + point
        ang = np.degrees(np.arccos(np.clip((np.trace(R_exp.T @ R1) - 1) / 2, -1, 1)))
        dpos = float(np.linalg.norm(t1 - t_exp))
        held = sc.fingers_touch_object()
        ok = ang <= 20 and dpos <= 0.03 and held and not plan["forced"]
        off = plan["pour_vs_gt_handle_deg"]
        task_ok = ok and off <= 30
        if task_ok:
            verdict = "SUCCESS: poured away from the handle"
        elif not ok:
            verdict = f"FAILED: {'dropped / not held' if not held else 'wrong pose'} ({ang:.0f} deg / {100 * dpos:.1f} cm off)"
        else:
            verdict = f"FAILED: motion done but poured {off:.0f} deg off (toward the side / handle)"
        rec(verdict)
        sc.set_gripper(0, seconds=1.5, renderer=renderer, cam=cam)
        _caption_frames(sc, caption, hdr)
        result.update(grasp=[np.asarray(plan["grasp"][0]).tolist(), np.asarray(plan["grasp"][1]).tolist()],
                      pour_axis=np.asarray(axis).tolist(), pour_point=np.asarray(point).tolist())
        result.update(success=bool(ok), task_success=bool(task_ok), ang=float(ang), dpos=dpos, held=bool(held),
                      tau=plan["tau"], pour_side_vs_gt_handle_deg=off, **result_close)
    if video and sc.frames:
        import imageio
        imageio.mimsave(video, sc.frames, fps=30, macro_block_size=1)
        result["video"] = video
    if renderer is not None:
        renderer.close()
    return result


def replay_until_failure(sc, W, mask, gs, order, screws, method, video, renderer, n_tried, result, seed, taus_all=()):
    """For a method with NO feasible plan: replay (kinematically, no physics) the motion to its best
    grasp that has an IK solution, collision-checking every configuration as the planner does, and
    stop at the first one that fails; the contacts are marked red and named in the caption."""
    import mujoco
    m = sc.model
    # the best-ranked grasp whose pre-grasp pose is reachable AND collision-free (so the planner can
    # get there and the failure, if any, shows during the approach); else the best reachable one
    best, first_reachable = None, None
    for i in order:
        c1, c2, app = gs[i]
        Rg = grasp_frame(c1, c2, app)
        mid = 0.5 * (c1 + c2)
        q_g = sc.ik(mid, Rg, restarts=6, seed=seed)
        if q_g is None:
            continue
        q_pre = sc.ik(mid - Rg[:, 2] * 0.10, Rg, q0=q_g, restarts=0)
        if q_pre is None:
            continue
        cand = (i, mid, Rg, q_g, q_pre)
        first_reachable = first_reachable or cand
        if not sc.contacts(q_pre):
            best = cand
            break
    best = best or first_reachable
    cam = "side"
    if best:
        sc.set_screw_overlay(*screws[best[0]][:2])
        cam = sc.camera_for(screws[best[0]][0])
    hdr = inset2(W, mask, gs, [taus_all[k] for k in range(len(gs))], gs[best[0]] if best else None,
                 screws[best[0]][:2] if best else None, LABEL[method], save=video.replace(".mp4", "_inset.png"))
    caption = {"text": ""}
    sc.frames = []
    q0 = sc.data.qpos[:7].copy()
    d = sc.data
    obj_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.obj]
    rgba0 = m.geom_rgba[obj_geoms].copy()

    def show(q, marks=(), n=1):
        d.qpos[:7] = q
        d.qpos[7:9] = 0.04
        mujoco.mj_forward(m, d)
        for _ in range(n):
            renderer.update_scene(d, camera=cam)
            scn = renderer.scene
            sc.add_overlay(scn)
            for p in marks:
                if scn.ngeom < scn.maxgeom:
                    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.012, 0, 0]),
                                        np.asarray(p, float), np.eye(3).ravel(), np.array([1, 0, 0, 0.9], np.float32))
                    scn.ngeom += 1
            sc.frames.append(renderer.render().copy())

    if best is None:
        caption["text"] = f"FAILED: none of the {n_tried} grasps on this part is reachable (no IK) - the robot cannot start"
        show(q0, n=90)
        _caption_frames(sc, caption, hdr)
        result.update(reason="no IK for any grasp")
    else:
        i, mid, Rg, q_g, q_pre = best
        approach = sc.cartesian(q_pre, [(mid - Rg[:, 2] * 0.10 * (1 - s_), Rg) for s_ in np.linspace(0.1, 1, 10)]) or [q_pre, q_g]
        # to the pre-grasp: the planner's collision-free path when the pre-grasp pose is itself valid
        # (only a colliding pre-grasp pose leaves the straight line, which then stops at the contact)
        to_pre = sc.plan_rrt(q0, q_pre, seed=seed) if not sc.contacts(q_pre) else None
        phases = [("move to pre-grasp", to_pre or [q0, q_pre]), ("approach along the grasp axis", approach)]
        stopped = None
        for name, path in phases:
            for qa, qb in zip(path[:-1], path[1:]):
                n = max(2, int(np.ceil(np.abs(qb - qa).max() / 0.01)))
                for k, tt in enumerate(np.linspace(0, 1, n + 1)[1:]):
                    q = qa + (qb - qa) * tt
                    hit = sc.contacts(q)
                    if hit:
                        stopped = (name, q, hit)
                        break
                    if k % 3 == 0:
                        caption["text"] = f"collision-checking every step: {name}"
                        show(q)
                if stopped:
                    break
            if stopped:
                break
        _caption_frames(sc, caption, hdr)
        if stopped:
            name, q, hit = stopped
            rb, other, depth, _ = max(hit, key=lambda h: h[2])
            m.geom_rgba[obj_geoms] = [0.9, 0.25, 0.25, 1.0] if other == "mug" else rgba0
            short = "approach" if name.startswith("approach") else name
            caption["text"] = (f"COLLISION CHECK FAILED on {short}: {rb} hits the {other} ({1000 * depth:.0f} mm) - STOPPED"
                               f"\nshown: best-ranked grasp with a valid pre-grasp; all {n_tried} candidates fail a check")
            show(q, [h[3] for h in hit], n=110)
            m.geom_rgba[obj_geoms] = rgba0
            result.update(reason=f"collision during {name}: {rb} vs {other}, {1000 * depth:.0f} mm")
        else:
            caption["text"] = (f"FAILED: the approach is collision-free, but the arm cannot reach the lift / pour poses (no IK)"
                               f"\nshown: best-ranked grasp with a valid pre-grasp; all {n_tried} candidates fail a check")
            show(approach[-1], n=110)
            result.update(reason="no IK for lift/pour")
        _caption_frames(sc, caption, hdr)
    import imageio
    imageio.mimsave(video, sc.frames, fps=30, macro_block_size=1)
    result["video"] = video
    renderer.close()
    return result


def _caption_frames(sc, caption, hdr):
    """Burn the current caption + inset into frames recorded since the last call."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=19)
    except TypeError:
        font = small = ImageFont.load_default()
    start = getattr(sc, "_done", 0)
    for k in range(start, len(sc.frames)):
        im = Image.fromarray(sc.frames[k])
        if hdr is not None:
            im.paste(Image.fromarray(hdr).resize((620, 256)), (10, 10))
        dr = ImageDraw.Draw(im)
        lines = caption["text"].split("\n")
        hbar = 12 + 32 * len(lines)
        dr.rectangle([0, im.height - hbar, im.width, im.height], fill=(0, 0, 0))
        for j, ln in enumerate(lines):
            dr.text((14, im.height - hbar + 8 + 32 * j), ln, fill=(255, 255, 255), font=font if j == 0 else small)
        sc.frames[k] = np.asarray(im)
    sc._done = len(sc.frames)


if __name__ == "__main__":
    D, oid, methods = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    video = sys.argv[4] if len(sys.argv) > 4 else None
    force = os.environ.get("FORCE_FAILED_PLANS") == "1"
    replay = os.environ.get("REPLAY_FAILED_PLANS") == "1"
    for m in methods:
        v = None if video is None else video.replace(".mp4", f"_{m}.mp4")
        print(run(D, oid, m, v, force=force, replay=replay), flush=True)
