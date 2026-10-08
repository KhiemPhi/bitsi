"""Franka Panda on a table in MuJoCo: IK, collision-checked motion planning, grasp-and-pour execution.

Robot: MuJoCo Menagerie `franka_emika_panda/panda.xml` (7 position-servo joints + tendon gripper,
ctrl 255 = open, 0 = closed), mounted on a table. The object is a free body made of CoACD convex
pieces (sim.mujoco_grasp.convex_parts); it is held by friction only.

Planning
  ik()            damped least squares on the tool point between the fingertips (site "tcp"),
                  position + orientation, joint limits clamped, several restarts
  plan_rrt()      RRT-Connect in joint space with collision checking (robot vs table, robot vs
                  object, robot vs itself), then random shortcutting
  cartesian()     a straight-line / screw end-effector path solved as a chain of IK steps
Execution
  execute()       time-parameterises a joint path (max joint speed) and tracks it with the arm's
                  position servos while stepping physics; every few steps a frame is recorded
"""
import os

import numpy as np

MENAGERIE = os.path.expanduser("~/.cache/fitpart_assets/menagerie/franka_emika_panda/panda.xml")
TABLE_TOP = 0.0          # robot base and table top at z = 0
TCP_OFFSET = 0.1034      # hand origin -> point between the fingertip pads, along hand z
WRIST_BEHIND_TCP = 0.10  # wrist camera this far behind the fingertip point (eyeinhand.CAM_BEHIND_TCP)
ARM_Q = slice(0, 7)
HOME = np.array([0, -0.3, 0, -2.2, 0, 1.9, 0.785])


def add_environment(spec, parts, obj_pos=(0.55, 0.0, 0.10), obj_quat=(1, 0, 0, 0), mass=0.35, friction=1.0):
    """Floor, table, lights, fixed cameras and the object (CoACD convex pieces) - shared by every robot."""
    import mujoco
    import trimesh
    wb = spec.worldbody
    # floor, table, lights, camera
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75],
                rgba=[0.75, 0.75, 0.73, 1])
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.55, 0.6, 0.375],
                pos=[0.45, 0, TABLE_TOP - 0.375], rgba=[0.62, 0.48, 0.34, 1], friction=[1.0, 0.02, 0.002])
    wb.add_light(pos=[0.5, -0.5, 2.0], dir=[0, 0.3, -1], diffuse=[0.8, 0.8, 0.8])
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    wb.add_light(pos=[1.5, 1.0, 1.5], dir=[-1, -0.6, -0.8], diffuse=[0.4, 0.4, 0.4])
    def look(pos, at):
        f = np.asarray(at, float) - pos
        f /= np.linalg.norm(f)
        x = np.cross(f, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(x, f)
        return list(x) + list(y)
    wb.add_camera(name="side", pos=[1.25, -0.95, 0.62], xyaxes=look(np.array([1.25, -0.95, 0.62]), [0.42, 0.0, 0.18]))
    wb.add_camera(name="front", pos=[1.45, 0.0, 0.45], xyaxes=look(np.array([1.45, 0.0, 0.45]), [0.45, 0.0, 0.18]))
    wb.add_camera(name="side_left", pos=[1.25, 0.95, 0.62], xyaxes=look(np.array([1.25, 0.95, 0.62]), [0.42, 0.0, 0.18]))
    wb.add_camera(name="back_left", pos=[0.10, 1.05, 0.62], xyaxes=look(np.array([0.10, 1.05, 0.62]), [0.42, 0.0, 0.18]))
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.93, 0.93, 0.95], rgb2=[0.55, 0.6, 0.68], width=256, height=256)
    # object
    vol = sum(trimesh.Trimesh(V, F).convex_hull.volume for V, F in parts)
    dens = mass / max(vol, 1e-9)
    body = wb.add_body(name="obj", pos=list(obj_pos), quat=list(obj_quat))
    body.add_freejoint(name="obj_free")
    for k, (V, F) in enumerate(parts):
        spec.add_mesh(name=f"p{k}", uservert=np.asarray(V, float).ravel().tolist())
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"p{k}", density=dens,
                      friction=[friction, 0.02, 0.002], condim=4, solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],
                      rgba=[0.80, 0.80, 0.85, 1])


def build_scene(parts, obj_pos=(0.55, 0.0, 0.10), obj_quat=(1, 0, 0, 0), mass=0.35, friction=1.0):
    import mujoco
    import trimesh
    spec = mujoco.MjSpec.from_file(MENAGERIE)
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    wb = spec.worldbody
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.004, 0, 0], rgba=[1, 0, 0, 0.6])
    # wrist camera (D405-like): on the hand, WRIST_BEHIND_TCP behind the fingertip point, offset
    # along hand x (perpendicular to the finger travel, so the fingers stay out of view), looking along
    # the approach axis (MuJoCo cameras look along their -z, hence the half turn about x)
    hand.add_camera(name="wrist", pos=[0.06, 0.0, TCP_OFFSET - WRIST_BEHIND_TCP], quat=[0, 1, 0, 0], fovy=58.0)
    add_environment(spec, parts, obj_pos, obj_quat, mass, friction)
    # Menagerie's gripper servo (force = 0.0157 ctrl - 100 L - 10 dL/dt on the finger tendon L in
    # [0, 0.08]) squeezes a 1 cm handle with ~1 N. Same open/closed positions, 100x stiffer, so a
    # closed grip reaches the actuator's 100 N force limit (real Panda hand: 70 N continuous).
    act = spec.actuator("actuator8")
    kp = 10000.0
    act.gainprm[0] = 0.08 * kp / 255.0
    act.biasprm[1] = -kp
    act.biasprm[2] = -200.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    return spec.compile()


class PandaScene:
    def __init__(self, parts, obj_xy=(0.55, 0.0), mass=0.35):
        import mujoco
        self.mj = mujoco
        self.model = build_scene(parts, obj_pos=(obj_xy[0], obj_xy[1], 0.15), mass=mass)
        self.data = mujoco.MjData(self.model)
        self.chk = mujoco.MjData(self.model)       # scratch data for collision checks / IK
        m = self.model
        self.tcp = m.site("tcp").id
        self.obj = m.body("obj").id
        self.obj_q = m.jnt_qposadr[m.joint("obj_free").id]
        self.obj_v = m.jnt_dofadr[m.joint("obj_free").id]
        self.lo = m.jnt_range[:7, 0].copy()
        self.hi = m.jnt_range[:7, 1].copy()
        self.table = m.geom("table").id
        root = m.body("link0").id
        self.robot_bodies = {b for b in range(m.nbody) if self._is_desc(b, root)}
        self.finger_bodies = {m.body("left_finger").id, m.body("right_finger").id}
        self.hand_bodies = self.finger_bodies | {m.body("hand").id}
        self.frames = []

    # gripper command in the Panda convention (255 = open, 0 = closed); robots override grip()
    grip_ctrl = 7

    def grip(self, value):
        return value

    def _is_desc(self, b, root):
        while b > 0:
            if b == root:
                return True
            b = self.model.body_parentid[b]
        return False

    # ------------------------------------------------------------------ state
    def reset(self, obj_R=np.eye(3), settle=1.5):
        mj, m, d = self.mj, self.model, self.data
        mj.mj_resetData(m, d)
        d.qpos[ARM_Q] = HOME
        d.qpos[7:9] = 0.04
        d.ctrl[:7] = HOME
        d.ctrl[7] = 255
        q = np.zeros(4)
        mj.mju_mat2Quat(q, np.asarray(obj_R, float).ravel())
        d.qpos[self.obj_q + 3:self.obj_q + 7] = q
        mj.mj_forward(m, d)
        # drop the object so its lowest point is 2 cm above the table
        zmin = min(self._geom_zmin(g) for g in range(m.ngeom) if m.geom_bodyid[g] == self.obj)
        d.qpos[self.obj_q + 2] += TABLE_TOP + 0.02 - zmin
        mj.mj_forward(m, d)
        for _ in range(int(settle / m.opt.timestep)):
            mj.mj_step(m, d)
        return self.obj_pose()

    def _geom_zmin(self, g):
        m, d = self.model, self.data
        V = m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]
        W = V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        return W[:, 2].min()

    def obj_pose(self):
        d = self.data
        return d.xpos[self.obj].copy(), d.xmat[self.obj].reshape(3, 3).copy()

    # ------------------------------------------------------------------ kinematics
    def fk(self, q):
        d = self.chk
        d.qpos[:] = self.data.qpos
        d.qpos[ARM_Q] = q
        self.mj.mj_kinematics(self.model, d)
        self.mj.mj_comPos(self.model, d)
        return d.site_xpos[self.tcp].copy(), d.site_xmat[self.tcp].reshape(3, 3).copy()

    def ik(self, pos, R, q0=None, iters=300, tol_p=2e-3, tol_r=0.03, restarts=8, seed=0, damping=1e-2):
        mj, m, d = self.mj, self.model, self.chk
        rng = np.random.default_rng(seed)
        starts = [self.data.qpos[ARM_Q].copy() if q0 is None else np.asarray(q0, float)]
        starts += [getattr(self, "home", HOME) + rng.normal(0, 0.4, 7) for _ in range(restarts)]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        best = None
        for q in starts:
            q = np.clip(q, self.lo, self.hi)
            for _ in range(iters):
                p, Rc = self.fk(q)
                ep = pos - p
                er = 0.5 * (np.cross(Rc[:, 0], R[:, 0]) + np.cross(Rc[:, 1], R[:, 1]) + np.cross(Rc[:, 2], R[:, 2]))
                if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                    break
                mj.mj_jacSite(m, d, jp, jr, self.tcp)
                J = np.vstack([jp[:, :7], jr[:, :7]])
                e = np.concatenate([ep, er])
                dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
                q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
            p, Rc = self.fk(q)
            err = np.linalg.norm(pos - p) + 0.1 * np.linalg.norm(Rc - R)
            ok = np.linalg.norm(pos - p) < tol_p * 2 and np.linalg.norm(Rc - R) < 0.1
            if ok and best is None or (ok and err < best[1]):
                best = (q.copy(), err)
                if q0 is not None:
                    break
        return None if best is None else best[0]

    # ------------------------------------------------------------------ collisions
    def in_collision(self, q, allow_obj_fingers=False, held=None, gripper=0.04, tol=0.002):
        """Robot vs table / object / itself at arm configuration q. If held = (T_tcp_obj), the
        object moves rigidly with the hand (transport); finger-object contacts are then allowed."""
        mj, m, d = self.mj, self.model, self.chk
        d.qpos[:] = self.data.qpos
        d.qpos[ARM_Q] = q
        d.qpos[7:9] = gripper
        if held is not None:
            p, R = self.fk(q)
            Rt, tt = held
            Ro = R @ Rt
            to = p + R @ tt
            qq = np.zeros(4)
            mj.mju_mat2Quat(qq, Ro.ravel())
            d.qpos[self.obj_q:self.obj_q + 3] = to
            d.qpos[self.obj_q + 3:self.obj_q + 7] = qq
        mj.mj_forward(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -tol:
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            o1, o2 = b1 == self.obj, b2 == self.obj
            if not (r1 or r2):
                if held is not None and (o1 or o2) and (c.geom1 == self.table or c.geom2 == self.table):
                    if c.dist < -0.004:
                        return True
                continue
            if (o1 or o2) and (allow_obj_fingers or held is not None):
                other = b2 if o1 else b1
                if other in self.hand_bodies:
                    continue
            if r1 and r2:
                # self-collision: ignored with the original planner; with OMPL (allowed_self set) any
                # robot-robot contact outside the allowed-collision matrix is a collision
                allowed = getattr(self, "allowed_self", None)
                if allowed is None or (min(b1, b2), max(b1, b2)) in allowed:
                    continue
                return True
            return True
        return False

    def contacts(self, q, gripper=0.04, tol=0.002):
        """Penetrating contacts of the robot at q (fingers open): list of (robot body name, other
        name, depth in m, world position). Same rule as in_collision without finger exemptions."""
        mj, m, d = self.mj, self.model, self.chk
        d.qpos[:] = self.data.qpos
        d.qpos[ARM_Q] = q
        d.qpos[7:9] = gripper
        mj.mj_forward(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -tol:
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if r1 == r2:
                continue
            rb, ob = (b1, b2) if r1 else (b2, b1)
            other = "mug" if ob == self.obj else ("table" if self.table in (c.geom1, c.geom2) else m.body(ob).name)
            out.append((m.body(rb).name, other, -float(c.dist), c.pos.copy()))
        return out

    def segment_free(self, qa, qb, step=0.05, **kw):
        n = int(np.ceil(np.abs(qb - qa).max() / step))
        return all(not self.in_collision(qa + (qb - qa) * t, **kw) for t in np.linspace(0, 1, n + 1)[1:])

    def path_free(self, qs, **kw):
        return all(self.segment_free(a, b, **kw) for a, b in zip(qs[:-1], qs[1:]))

    # ------------------------------------------------------------------ planning
    def plan_rrt(self, qa, qb, max_iter=3000, step=0.25, seed=0, shortcut=60, **kw):
        if getattr(self, "ompl", None) is not None:      # OMPL RRT-Connect (sim/ompl_planner.py)
            return self.ompl.plan(qa, qb, seed=seed, **kw)
        if self.segment_free(qa, qb, **kw):
            return [qa, qb]
        rng = np.random.default_rng(seed)
        A, B = [(qa, -1)], [(qb, -1)]

        def nearest(T, q):
            return int(np.argmin([np.abs(n[0] - q).sum() for n in T]))

        def extend(T, q):
            i = nearest(T, q)
            qn = T[i][0]
            d = q - qn
            dist = np.abs(d).max()
            qnew = q if dist <= step else qn + d * step / dist
            if self.segment_free(qn, qnew, **kw):
                T.append((qnew, i))
                return len(T) - 1, np.allclose(qnew, q)
            return None, False

        for it in range(max_iter):
            qr = rng.uniform(self.lo, self.hi) if rng.random() > 0.1 else B[rng.integers(len(B))][0]
            ia, _ = extend(A, qr)
            if ia is None:
                A, B = B, A
                continue
            # connect B towards A's new node
            while True:
                ib, reached = extend(B, A[ia][0])
                if ib is None or reached:
                    break
            if ib is not None and reached:
                pa, pb = [], []
                i = ia
                while i >= 0:
                    pa.append(A[i][0]); i = A[i][1]
                i = ib
                while i >= 0:
                    pb.append(B[i][0]); i = B[i][1]
                path = pa[::-1] + pb
                if not np.allclose(path[0], qa):
                    path = path[::-1]
                return self._shortcut(path, shortcut, rng, **kw)
            A, B = B, A
        return None

    def _shortcut(self, path, n, rng, **kw):
        path = list(path)
        for _ in range(n):
            if len(path) < 3:
                break
            i, j = sorted(rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self.segment_free(path[i], path[j], **kw):
                path = path[:i + 1] + path[j:]
        return path

    def cartesian(self, q0, poses, **kw):
        """Chain of IK solutions along a list of (pos, R) targets, each seeded by the previous one."""
        qs = [np.asarray(q0, float)]
        for p, R in poses:
            q = self.ik(p, R, q0=qs[-1], restarts=0, iters=200)
            if q is None or np.abs(q - qs[-1]).max() > 0.5:
                return None
            qs.append(q)
        return qs

    # ------------------------------------------------------------------ execution
    def execute(self, path, gripper=255, vmax=0.8, settle=0.3, renderer=None, cam="side", every=17):
        mj, m, d = self.mj, self.model, self.data
        dt = m.opt.timestep
        d.ctrl[self.grip_ctrl] = self.grip(gripper)
        for qa, qb in zip(path[:-1], path[1:]):
            T = max(np.abs(qb - qa).max() / vmax, 0.05)
            n = int(T / dt)
            for i in range(n):
                s = (i + 1) / n
                s = 3 * s ** 2 - 2 * s ** 3  # smooth start/stop per segment
                d.ctrl[:7] = qa + (qb - qa) * s
                mj.mj_step(m, d)
                self._record(renderer, cam, every)
        for _ in range(int(settle / dt)):
            mj.mj_step(m, d)
            self._record(renderer, cam, every)

    def set_gripper(self, value, seconds=0.8, renderer=None, cam="side", every=17):
        d = self.data
        d.ctrl[self.grip_ctrl] = self.grip(value)
        for _ in range(int(seconds / self.model.opt.timestep)):
            self.mj.mj_step(self.model, d)
            self._record(renderer, cam, every)

    def set_screw_overlay(self, axis, point, L=0.20, r_arc=0.04):
        """Draw the screw in the rendered scene: omega as a red arrow along the axis (through q),
        and a curled arrow around it giving the positive rotation (right-hand rule)."""
        w = np.asarray(axis, float) / np.linalg.norm(axis)
        q = np.asarray(point, float)
        segs = [("arrow", q - w * L, q + w * L, 0.009)]
        u = np.cross(w, [0, 0, 1.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(w, [1.0, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(w, u)
        c = q + w * L * 0.7
        th = np.linspace(0, 1.5 * np.pi, 14)
        pts = [c + r_arc * (np.cos(t) * u + np.sin(t) * v) for t in th]
        for k in range(len(pts) - 1):
            segs.append(("arrow" if k == len(pts) - 2 else "capsule", pts[k], pts[k + 1], 0.003))
        self.overlay = segs

    def camera_for(self, axis, lookat=None, distance=1.35, elevation=-24, turn=35.0):
        """A free camera that sees the screw axis side-on AND the handle: its viewing direction lies
        in the plane spanned by the handle side h = up x omega and the vertical (both perpendicular
        to omega), looking down at `elevation` deg, from the side of h away from the robot base."""
        import mujoco
        w = np.asarray(axis, float) / np.linalg.norm(axis)
        h = np.cross([0, 0, 1.0], w)
        h /= np.linalg.norm(h) + 1e-12
        tgt = self.obj_pose()[0] + np.array([0, 0, 0.12]) if lookat is None else np.asarray(lookat, float)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = tgt
        cam.distance = distance
        cam.elevation = elevation
        # camera sits at tgt - distance * forward; choose forward = -h or +h so that it sits farther from the base
        fwd = -h if (tgt + h)[0] ** 2 + (tgt + h)[1] ** 2 > (tgt - h)[0] ** 2 + (tgt - h)[1] ** 2 else h
        # turned `turn` deg off the handle direction: omega stays ~cos(turn) side-on and the handle
        # is seen from the side instead of end-on
        cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])) + turn)
        return cam

    def add_overlay(self, scn):
        mj = self.mj
        for kind, a, b, wdt in getattr(self, "overlay", []):
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            t = mj.mjtGeom.mjGEOM_ARROW if kind == "arrow" else mj.mjtGeom.mjGEOM_CAPSULE
            mj.mjv_initGeom(g, t, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([0.85, 0.05, 0.15, 1.0], np.float32))
            mj.mjv_connector(g, t, wdt * (2.5 if kind == "arrow" else 1.0), np.asarray(a, float), np.asarray(b, float))
            scn.ngeom += 1

    def _record(self, renderer, cam, every):
        if renderer is None:
            return
        self._k = getattr(self, "_k", 0) + 1
        if self._k % every == 0:
            renderer.update_scene(self.data, camera=cam)
            self.add_overlay(renderer.scene)
            self.frames.append(renderer.render().copy())

    def fingers_touch_object(self):
        m, d = self.model, self.data
        hit = set()
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if self.obj in (b1, b2):
                o = b2 if b1 == self.obj else b1
                if o in self.finger_bodies:
                    hit.add(o)
        return len(hit) == 2


def grasp_frame(c1, c2, approach):
    """Hand rotation for a grasp: hand y = closing direction, hand z = approach (towards object)."""
    y = (c2 - c1) / np.linalg.norm(c2 - c1)
    z = approach / np.linalg.norm(approach)
    z = z - (z @ y) * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)
