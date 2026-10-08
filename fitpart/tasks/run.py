"""Execute a BITSI-inferred task screw with the Kinova Gen3 + Robotiq 2F-85 and check success.

One trial: build the scene (procedural articulated object, or a HANDAL tool lying on the table), take
the object's surface samples (full scan), infer the task screw with tasks.screws (no labels), then:
  grasp candidates on the screw's grasp part (antipodal sampler, across-the-bar and plain), first
  candidate with a collision-free IK solution, approach path and a feasible Cartesian screw path wins;
  RRT-Connect home -> pre-grasp, straight approach, close, optional lift, follow the screw, settle.
  press: no grasp; the closed fingertips push along the screw axis.
Success is measured on the ground truth (joint displacement of the articulated object, or the tool's
pose and the GT handle for rigid tools) -- see SUCCESS.
usage (graspsim env): python -m fitpart.tasks.run TASK SEED_OR_OID OUT.jsonl [--video PATH]
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.grasping import sample_grasps
from fitpart.sim.kinova_scene import KinovaScene, build_kinova
from fitpart.sim.mujoco_grasp import HANDAL_MASS, axis_angle, convex_parts
from fitpart.sim.panda_scene import grasp_frame
from fitpart.tasks import articulated as AR
from fitpart.tasks.screws import UP, RECEIVER, infer

ART_TASK = dict(turn="bottle", hinge="lidbox", slide="drawer", press="button", squeeze="tongs", pivot="pivotbox",
                cut="knife")
TOOL_TASK = dict(lift="power_drill", swing="hammer", scoop="ladle", stir="whisk", handover="screwdriver")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_pour")
PINCH_TO_TIP = 0.018     # the closed 2F-85 fingertips reach this far beyond the pinch point
PLANNER = os.environ.get("PLANNER", "ompl")    # default: OMPL RRT-Connect + self-collision + segment checks; "rrt": the original planner
CHECK_SEGMENTS = PLANNER == "ompl"


class ArtScene(KinovaScene):
    """Kinova scene with a procedural articulated object (the object bodies count as obstacles)."""

    def __init__(self, kind, seed):
        box = {}
        import mujoco
        self.mj = mujoco
        self.model = build_kinova(extra=lambda spec: box.setdefault("a", AR.build(spec, kind, seed)))
        self.art = box["a"]
        KinovaScene._init_from_model(self)
        m = self.model
        art_root = m.body("art").id
        self.art_bodies = {b for b in range(m.nbody) if self._is_desc(b, art_root)}
        if kind in ("pivotbox", "knife"):        # free rigid objects: they are "the object"
            self.obj = art_root
            self.obj_q = m.jnt_qposadr[m.joint("art_free").id]

    def joint(self):
        j = self.model.joint(self.art.joint).id
        return float(self.data.qpos[self.model.jnt_qposadr[j]])

    def in_collision(self, q, allow_obj_fingers=False, held=None, gripper=None, tol=0.002):
        mj, m, d = self.mj, self.model, self.chk
        d.qpos[:] = self.data.qpos
        d.qpos[:7] = q
        mj.mj_forward(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -tol:
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if r1 and r2:
                allowed = getattr(self, "allowed_self", None)
                if allowed is None or (min(b1, b2), max(b1, b2)) in allowed:
                    continue
                return True
            if r1 == r2:
                continue
            other = b2 if r1 else b1
            me = b1 if r1 else b2
            if other in self.art_bodies and allow_obj_fingers and me in self.hand_bodies:
                continue
            return True
        return False


def lying_pose(P):
    """Rotation that lays a tool flat (smallest principal axis vertical)."""
    c = P - P.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    R = np.stack([vt[0], vt[1], np.cross(vt[0], vt[1])])
    return R


START_OPEN = dict(hinge=1.2, slide=0.12, squeeze=-0.42)     # joint value for direction -1 commands


def setup(task, key, direction=1):
    if task in ART_TASK:
        sc = ArtScene(ART_TASK[task], int(key))
        sc.reset(settle=0.0)
        if direction < 0 and task in START_OPEN:
            m = sc.model
            j = m.joint(sc.art.joint).id
            sc.data.qpos[m.jnt_qposadr[j]] = START_OPEN[task]
            if task == "squeeze":   # hold the tongs closed with the spring reference
                m.qpos_spring[m.jnt_qposadr[j]] = START_OPEN[task]
        for _ in range(250):
            sc.mj.mj_step(sc.model, sc.data)
        W, Wn, L = AR.sample_surface(sc.model, sc.data, sc.art, seed=int(key))
        return sc, W, Wn, dict(labels=L, W0=W.copy())
    cat = TOOL_TASK[task]
    z = np.load(f"{DATA}/{cat}/{key}.npz", allow_pickle=True)
    P, N, handle = z["P"].astype(float), z["N"].astype(float), z["handle"]
    sc = KinovaScene(convex_parts(str(z["mesh"])), mass=HANDAL_MASS.get(cat, 0.3))
    t, R = sc.reset(lying_pose(P))
    return sc, P @ R.T + t, N @ R.T, dict(P=P, handle=handle, t0=t, R0=R)


def screw_poses(ts, p0, R0, n=24):
    out = []
    for k in range(1, n + 1):
        f = k / n
        if not np.isfinite(ts.pitch):
            out.append((p0 + ts.axis * ts.theta * f, R0))
        else:
            Rk = axis_angle(ts.axis, ts.theta * f)
            q = ts.point + (UP * ts.lift if ts.lift else 0)
            out.append((Rk @ (p0 - q) + q + ts.pitch * ts.theta * f * ts.axis, Rk @ R0))
    return out


def plan(sc, W, Wn, ts, seed=0, max_candidates=200, tau_of=None):
    """tau_of: optional (com, mass); when given, tau under the task screw is computed for every
    candidate (for the heatmap only -- the candidate order is unchanged)."""
    if ts.push:
        top = ts.point
        for roll in np.linspace(0, np.pi, 4, endpoint=False):
            z = ts.axis
            x = np.cross(z, [1, 0, 0] if abs(z[0]) < 0.9 else [0, 1, 0])
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            R = np.stack([np.cos(roll) * x + np.sin(roll) * y, -np.sin(roll) * x + np.cos(roll) * y, z], 1)
            touch = top - z * PINCH_TO_TIP
            q_pre = sc.ik(touch - z * 0.08, R, restarts=6, seed=seed)
            if q_pre is None or sc.in_collision(q_pre):
                continue
            path = sc.cartesian(q_pre, [(touch - z * 0.08 * (1 - s), R) for s in np.linspace(0.1, 1, 8)] +
                                [(touch + z * ts.theta * s, R) for s in np.linspace(0.25, 1, 4)])
            if path is None:
                continue
            to_pre = sc.plan_rrt(sc.data.qpos[:7].copy(), q_pre, seed=seed)
            if to_pre is None:
                continue
            return dict(push=True, to_pre=to_pre, approach=path, grasp=None, gs=[], taus=[], start=top - z * 0.08)
        return None
    g = ts.grasp
    gs = []
    for across in ((True, False) if ts.across else (False, True)):
        gs += sample_grasps(W[g], Wn[g], W, n_contacts=2000 if across else 1000, max_grasps=120, seed=seed,
                            across_part=across)
    taus = None
    if tau_of is not None:
        from fitpart.sim.screw_metric import screw_tau
        com, mass = tau_of
        taus = []
        for c1, c2, _ in gs:
            try:
                taus.append(float(screw_tau(c1, c2, com, mass, ts.axis, ts.point, translation=not np.isfinite(ts.pitch))))
            except Exception:
                taus.append(-1e9)
    rng = np.random.default_rng(seed)
    for i in rng.permutation(len(gs))[:max_candidates]:
        c1, c2, app = gs[i]
        Rg = grasp_frame(c1, c2, app)
        mid = 0.5 * (c1 + c2)
        z = Rg[:, 2]
        q_g = sc.ik(mid, Rg, restarts=4, seed=seed)
        if q_g is None or sc.in_collision(q_g):
            continue
        q_pre = sc.ik(mid - z * 0.10, Rg, q0=q_g, restarts=0)
        if q_pre is None or sc.in_collision(q_pre):
            continue
        approach = sc.cartesian(q_pre, [(mid - z * 0.10 * (1 - s), Rg) for s in np.linspace(0.1, 1, 10)])
        if approach is None or (CHECK_SEGMENTS and not sc.path_free(approach)):
            continue
        q_last, p_last = approach[-1], mid
        lift = []
        if ts.lift:
            lift = sc.cartesian(q_last, [(mid + UP * ts.lift * s, Rg) for s in np.linspace(0.1, 1, 10)])
            if lift is None or (CHECK_SEGMENTS and not sc.path_free(lift, allow_obj_fingers=True)):
                continue
            q_last, p_last = lift[-1], mid + UP * ts.lift
        tsx = ts
        if ts.info.get("pivot") == "grasp":   # baseline: the grasp (after the lift) is the pivot
            import copy
            tsx = copy.copy(ts)
            tsx.point = mid.copy()
        motion = sc.cartesian(q_last, screw_poses(tsx, p_last, Rg))
        if motion is None or (CHECK_SEGMENTS and not sc.path_free(motion, allow_obj_fingers=True)):
            continue
        to_pre = sc.plan_rrt(sc.data.qpos[:7].copy(), q_pre, seed=seed)
        if to_pre is None:
            continue
        return dict(push=False, to_pre=to_pre, approach=approach, lift=lift, motion=motion, grasp=(c1, c2, app),
                    n_candidates=len(gs), gs=gs, taus=taus, start=mid + UP * ts.lift)
    return None


def obj_rot_deg(R0, R1):
    return float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1))))


def execute(sc, pl, renderer=None, cam=None):
    kw = dict(renderer=renderer, cam=cam) if renderer is not None else {}
    if pl["push"]:
        sc.set_gripper(0, seconds=0.6, **kw)                       # fingers closed, then push
        sc.execute(pl["to_pre"], gripper=0, **kw)
        sc.execute(pl["approach"], gripper=0, vmax=0.2, settle=0.6, **kw)
        return
    sc.execute(pl["to_pre"], **kw)
    sc.execute(pl["approach"], vmax=0.3, **kw)
    sc.set_gripper(0, seconds=1.0, **kw)
    if pl["lift"]:
        sc.execute(pl["lift"], gripper=0, vmax=0.3, **kw)
    sc.execute(pl["motion"], gripper=0, vmax=0.3, settle=0.8, **kw)


def grasp_on_handle(extra, before, grasp):
    """Fraction of the two executed contacts that lie on the GT handle (within 1 cm)."""
    from scipy.spatial import cKDTree
    t0, R0 = before["pose"]
    Wt = extra["P"] @ (R0 @ R0.T @ extra["R0"]).T + extra["t0"]
    tr = cKDTree(Wt[extra["handle"]])
    return float(np.mean([tr.query(c)[0] < 0.01 for c in grasp[:2]]))


def success(task, sc, ts, extra, before, grasp=None):
    """Ground-truth checks (the inferred screw is NOT used to judge itself). Tool-use tasks must hold the
    tool by its GT handle; hand-over must hold it away from the handle."""
    if "handle" in extra and grasp is not None and task in ("swing", "scoop", "stir", "handover"):
        on = grasp_on_handle(extra, before, grasp)
        if (task != "handover" and on < 0.5) or (task == "handover" and on > 0.5):
            return False, dict(grasp_on_handle=on, reason="wrong grasp part")
    if task in ("turn", "hinge", "slide", "press", "squeeze"):
        dq = (sc.joint() - before["joint"]) * before.get("direction", 1)
        need = dict(turn=np.deg2rad(120), hinge=np.deg2rad(45), slide=0.08, press=0.004, squeeze=np.deg2rad(14))[task]
        if task == "turn":
            ok, val = abs(dq) >= need, np.degrees(dq)
        elif task == "squeeze":
            ok, val = -dq >= need, np.degrees(dq)
        else:
            ok, val = dq >= need, (np.degrees(dq) if task == "hinge" else dq)
        return bool(ok), dict(joint_change=float(val))
    t1, R1 = sc.obj_pose()
    t0, R0 = before["pose"]
    held = sc.fingers_touch_object()
    rot = obj_rot_deg(R0, R1)
    if task == "pivot":
        return bool(rot >= 35), dict(rot=rot)
    if task == "lift":   # off the table: the object's lowest point rose >= 10 cm (a heavy tool swings in the grasp)
        zmin = min(sc._geom_zmin(g) for g in range(sc.model.ngeom) if sc.model.geom_bodyid[g] == sc.obj)
        return bool(held and zmin - before["zmin"] >= 0.10), dict(rise=float(zmin - before["zmin"]), held=bool(held))
    if task in ("swing", "scoop"):
        return bool(held and rot >= 25), dict(rot=rot, held=bool(held))
    if task == "stir":
        yaw = np.degrees(np.arccos(np.clip(((R1 @ R0.T)[:2, :2].trace()) / 2, -1, 1)))
        return bool(held and yaw >= 90), dict(yaw=float(yaw), held=bool(held))
    if task == "cut":   # along the TRUE blade direction (GT labels), not the inferred one
        L = extra["labels"]
        bd = extra["W0"][L == "blade"].mean(0) - extra["W0"][L == "handle"].mean(0)
        bd[2] = 0
        bd /= np.linalg.norm(bd)
        d = float((t1 - t0) @ bd)
        return bool(held and d >= 0.07), dict(stroke=d, held=bool(held))
    if task == "handover":
        P, handle = extra["P"], extra["handle"]
        Wt = P @ (R1 @ R0.T @ extra["R0"]).T
        hd = Wt[handle].mean(0) - Wt[~handle].mean(0)
        hd[2] = 0
        hd /= np.linalg.norm(hd) + 1e-12
        err = float(np.degrees(np.arccos(np.clip(hd @ RECEIVER, -1, 1))))
        return bool(held and err <= 30), dict(handle_err_deg=err, held=bool(held))
    raise ValueError(task)


def apply_spec(ts, spec):
    """Language slots -> screw magnitude and sense. BITSI fixes the axis, point and the positive sense
    (open / loosen / pull out / close the tongs); direction -1 reverses it."""
    mag = spec.get("magnitude")
    if mag is not None and spec.get("unit") in ("deg", "cm"):
        val = np.deg2rad(mag) if spec["unit"] == "deg" else mag / 100.0
        if not np.isfinite(ts.pitch) or spec["unit"] == "deg":
            ts.theta = np.sign(ts.theta or 1.0) * val
    ts.theta *= spec.get("direction", 1)
    return ts


def trial(task, key, video=None, seed=0, spec=None):
    direction = (spec or {}).get("direction", 1)
    sc, W, Wn, extra = setup(task, key, direction)
    res = dict(task=task, key=str(key), spec=spec)
    ts, A = infer(task, W, Wn)
    if ts is None:
        res.update(success=False, stage="no screw inferred")
        return res
    if spec:
        ts = apply_spec(ts, spec)
    # screw baselines (same grasp part and axis direction as BITSI, different pivot point):
    #   grasp    -- rotate about the grasp point (an end-effector rotation, no object-centric pivot)
    #   centroid -- rotate about the object's centroid
    base = os.environ.get("SCREW_BASELINE")
    if base == "centroid":
        ts.point = W.mean(0)
    elif base == "grasp":
        ts.info["pivot"] = "grasp"
    res["screw_baseline"] = base or "bitsi"
    res["screw"] = dict(axis=ts.axis.tolist(), point=ts.point.tolist(), pitch=float(ts.pitch), theta=float(ts.theta),
                        lift=ts.lift, grasp_points=int(ts.grasp.sum()), **{k: v for k, v in ts.info.items() if k != "handle_dir"})
    if task in ART_TASK and sc.art.gt_axis is not None:
        ang = float(np.degrees(np.arccos(np.clip(abs(ts.axis @ sc.art.gt_axis), 0, 1))))
        res["axis_err_deg"] = ang
        if np.isfinite(sc.art.gt_pitch) and np.isfinite(ts.pitch):
            v = ts.point - sc.art.gt_point
            res["line_dist_cm"] = float(100 * np.linalg.norm(v - (v @ sc.art.gt_axis) * sc.art.gt_axis))
    if "labels" in extra:
        u, c = np.unique(extra["labels"][ts.grasp], return_counts=True)
        res["grasp_part_gt"] = str(u[np.argmax(c)]) if len(u) else None
    elif "handle" in extra:
        res["grasp_handle_frac"] = float(extra["handle"][ts.grasp].mean())
    if PLANNER == "ompl":
        from fitpart.sim.ompl_planner import OMPLPlanner
        sc.ompl = OMPLPlanner(sc)
    res["planner"] = PLANNER
    tau_of = None
    if video:
        if task in TOOL_TASK:
            tau_of = (W.mean(0), HANDAL_MASS.get(TOOL_TASK[task], 0.3))
        elif task in ("pivot", "cut"):
            tau_of = (W.mean(0), 0.25 if task == "pivot" else 0.10)
        else:   # the moving part of an articulated object
            tau_of = (W[ts.grasp].mean(0), 0.05)
    pl = plan(sc, W, Wn, ts, seed=seed, tau_of=tau_of)
    if pl is None:
        res.update(success=False, stage="no feasible plan")
        return res
    before = dict(pose=sc.obj_pose())
    if task in TOOL_TASK:
        before["zmin"] = min(sc._geom_zmin(g) for g in range(sc.model.ngeom) if sc.model.geom_bodyid[g] == sc.obj)
    if task in ART_TASK and sc.art.joint:
        before["joint"] = sc.joint()
        before["direction"] = direction
    renderer = cam = None
    if video:
        import mujoco
        renderer = mujoco.Renderer(sc.model, 720, 960)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = W.mean(0) + np.array([0, 0, 0.05])
        cam.distance, cam.elevation, cam.azimuth = 1.0, -25, 135
        sc.frames = []
        from fitpart.tasks.viz import screw_overlay, task_inset
        sc.overlay = screw_overlay(ts, pl["start"])
        inset = task_inset(W, ts.grasp, pl["gs"], pl["taus"] or [], pl["grasp"], ts,
                           f"BITSI: {task} part (purple)", size=(800, 360))
    execute(sc, pl, renderer, cam)
    ok, meas = success(task, sc, ts, extra, before, grasp=pl.get("grasp"))
    res.update(success=ok, stage="executed", **meas)
    if video and sc.frames:
        import imageio
        from PIL import Image, ImageDraw
        cap = f"Kinova + BITSI: {task} ({'SUCCESS' if ok else 'FAILED'})  " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in meas.items())
        frames = []
        ins = Image.fromarray(inset)
        pw = 560                                    # side panel: inset, not covering the scene
        ins = ins.resize((pw, int(ins.height * pw / ins.width)))
        for fr in sc.frames:
            sc_im = Image.fromarray(fr)
            im = Image.new("RGB", (sc_im.width + pw, sc_im.height), "white")
            im.paste(sc_im, (0, 0))
            im.paste(ins, (sc_im.width, 60))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, im.width, 34], fill=(27, 150, 90) if ok else (196, 0, 58))
            d.text((10, 10), cap, fill=(255, 255, 255))
            y = 60 + ins.height + 20
            for line in (f"task: {task}", f"screw: axis {np.round(ts.axis, 2).tolist()}, through q {np.round(ts.point, 3).tolist()}",
                         f"pitch {'inf (translation)' if not np.isfinite(ts.pitch) else ts.pitch}, "
                         f"magnitude {ts.theta if not np.isfinite(ts.pitch) else np.degrees(ts.theta):.2f}"
                         f"{' m' if not np.isfinite(ts.pitch) else ' deg'}, lift {ts.lift} m",
                         "red in the scene: the commanded screw (where it acts)"):
                d.text((sc_im.width + 16, y), line, fill=(31, 31, 30))
                y += 22
            frames.append(np.asarray(im))
        imageio.mimsave(video, frames + [frames[-1]] * 30, fps=30, macro_block_size=1)
        res["video"] = video
    return res


if __name__ == "__main__":
    task, key, out = sys.argv[1], sys.argv[2], sys.argv[3]
    video = sys.argv[sys.argv.index("--video") + 1] if "--video" in sys.argv else None
    spec = json.loads(sys.argv[sys.argv.index("--spec") + 1]) if "--spec" in sys.argv else None
    if spec and spec.get("task") and spec["task"] != task:
        task = spec["task"]
    r = trial(task, key, video, spec=spec)
    with open(out, "a") as f:
        f.write(json.dumps(r, default=float) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "screw"}, default=float), flush=True)
    os._exit(0)   # the OMPL 1.x bindings double-free at interpreter teardown; results are written
