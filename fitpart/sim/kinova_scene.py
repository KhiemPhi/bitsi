"""Kinova Gen3 (7-DoF) with a Robotiq 2F-85, on the same table / object / cameras as PandaScene.

A drop-in for PandaScene (IK, collision checks, RRT, Cartesian paths, execution, video), so the
grasp-and-pour and eye-in-hand pipelines run unchanged:
  * gripper commands stay in the Panda convention (255 = open, 0 = closed); grip() maps them to the
    2F-85 actuator (0 = open, 255 = closed)
  * "tcp" = the 2F-85 pinch point (between the pads), "wrist" = a wrist camera WRIST_BEHIND_TCP behind
    it, offset across the finger travel (like the Gen3's built-in wrist camera module)
  * Menagerie's 2F-85 tendon actuator is limited to 5 N; it is made 4x stiffer and stronger (grip_force
    20): larger values (30, 100) fight the soft 4-bar linkage constraints and the fingers stop short
    (gap 1.75-4 cm instead of 1.4 cm in free space). Real 2F-85: 20-235 N, adjustable.
Joints 1, 3, 5, 7 are continuous on the real arm; IK limits them to +-2 pi.
"""
import os

import numpy as np

from .panda_scene import ARM_Q, TABLE_TOP, WRIST_BEHIND_TCP, PandaScene, add_environment

MENAGERIE = os.path.expanduser("~/.cache/fitpart_assets/menagerie")
GEN3_HOME = np.array([0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633])
PINCH = 0.145   # 2F-85 base -> pinch point


def build_kinova(parts=None, obj_pos=(0.55, 0.0, 0.10), obj_quat=(1, 0, 0, 0), mass=0.35, friction=1.0,
                 grip_force=20.0, extra=None):
    """extra: optional callable(spec) adding more bodies (e.g. tasks.articulated.build) before compile."""
    import mujoco
    spec = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, "kinova_gen3", "gen3.xml"))
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    grip = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, "robotiq_2f85", "2f85.xml"))
    spec.site("pinch_site").attach_body(grip.body("base_mount"), "g_", "")
    base = spec.body("g_base")
    base.add_site(name="tcp", pos=[0, 0, PINCH], size=[0.004, 0, 0], rgba=[1, 0, 0, 0.6])
    spec.delete(spec.camera("wrist"))  # Gen3's own wrist camera: replaced by one at the shared offset
    # fingers close along base y; the camera sits along base x, looking along +z (MuJoCo cameras look
    # along their -z: half turn about x)
    base.add_camera(name="wrist", pos=[0.06, 0.0, PINCH - WRIST_BEHIND_TCP], quat=[0, 1, 0, 0], fovy=58.0)
    if parts is None:  # reachability only: a tiny object far away
        parts = [(np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], float),
                  np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]
        obj_pos = (3.0, 3.0, 0.1)
    add_environment(spec, parts, obj_pos, obj_quat, mass, friction)
    # base_link has no joint, so MuJoCo treats it as part of the world and does not filter its contacts
    # with its child: base and shoulder meshes touch and friction locks joint 1 (0.44 rad short, motor
    # saturated). Exclude the pair explicitly.
    spec.add_exclude(bodyname1="base_link", bodyname2="shoulder_link")
    # the real Gen3 controller compensates gravity; without it the position servos sag by ~0.4 rad
    # with the 2F-85 on the wrist (tcp 20+ cm off target)
    for b in spec.bodies:
        if b.name and b.name not in ("world", "obj"):
            b.gravcomp = 1.0
    if extra is not None:
        extra(spec)
    act = spec.actuator("g_fingers_actuator")
    # position servo on the tendon, same ctrl range (0-255); stiffer and with a higher force limit
    scale = grip_force / 5.0
    act.gainprm[0] *= scale
    act.biasprm[1] *= scale
    act.biasprm[2] *= scale
    act.forcerange = [-5.0 * scale, 5.0 * scale]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    return spec.compile()


class KinovaScene(PandaScene):
    def __init__(self, parts=None, obj_xy=(0.55, 0.0), mass=0.35):
        import mujoco
        self.mj = mujoco
        self.model = build_kinova(parts, obj_pos=(obj_xy[0], obj_xy[1], 0.15), mass=mass)
        self._init_from_model()

    def _init_from_model(self):
        import mujoco
        self.data = mujoco.MjData(self.model)
        self.chk = mujoco.MjData(self.model)
        m = self.model
        self.home = GEN3_HOME
        self.tcp = m.site("tcp").id
        self.obj = m.body("obj").id
        self.obj_q = m.jnt_qposadr[m.joint("obj_free").id]
        self.obj_v = m.jnt_dofadr[m.joint("obj_free").id]
        rng = m.jnt_range[:7].copy()
        lim = m.jnt_limited[:7].astype(bool)
        rng[~lim] = [-2 * np.pi, 2 * np.pi]
        self.lo, self.hi = rng[:, 0], rng[:, 1]
        self.table = m.geom("table").id
        root = m.body("base_link").id
        self.robot_bodies = {b for b in range(m.nbody) if self._is_desc(b, root)}
        self.hand_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith("g_")}
        self.left = {b for b in self.hand_bodies if "left" in m.body(b).name}
        self.right = {b for b in self.hand_bodies if "right" in m.body(b).name}
        self.finger_bodies = self.left | self.right
        self.grip_ctrl = m.actuator("g_fingers_actuator").id
        self.frames = []

    def grip(self, value):
        return 255.0 - value

    def reset(self, obj_R=np.eye(3), settle=1.5):
        mj, m, d = self.mj, self.model, self.data
        mj.mj_resetData(m, d)
        d.qpos[ARM_Q] = self.home
        d.ctrl[:7] = self.home
        d.ctrl[self.grip_ctrl] = self.grip(255)
        q = np.zeros(4)
        mj.mju_mat2Quat(q, np.asarray(obj_R, float).ravel())
        d.qpos[self.obj_q + 3:self.obj_q + 7] = q
        mj.mj_forward(m, d)
        zmin = min(self._geom_zmin(g) for g in range(m.ngeom) if m.geom_bodyid[g] == self.obj)
        d.qpos[self.obj_q + 2] += TABLE_TOP + 0.02 - zmin
        mj.mj_forward(m, d)
        for _ in range(int(settle / m.opt.timestep)):
            mj.mj_step(m, d)
        return self.obj_pose()

    def in_collision(self, q, allow_obj_fingers=False, held=None, gripper=None, tol=0.002):
        """Robot vs table / object; the gripper keeps its current opening (the 2F-85 has no
        directly settable finger joints); held objects move with the hand as in PandaScene."""
        mj, m, d = self.mj, self.model, self.chk
        d.qpos[:] = self.data.qpos
        d.qpos[ARM_Q] = q
        if held is not None:
            p, R = self.fk(q)
            Rt, tt = held
            qq = np.zeros(4)
            mj.mju_mat2Quat(qq, (R @ Rt).ravel())
            d.qpos[self.obj_q:self.obj_q + 3] = p + R @ tt
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
                if held is not None and (o1 or o2) and (c.geom1 == self.table or c.geom2 == self.table) and c.dist < -0.004:
                    return True
                continue
            if r1 and r2:
                allowed = getattr(self, "allowed_self", None)
                if allowed is None or (min(b1, b2), max(b1, b2)) in allowed:
                    continue
                return True
            if (o1 or o2) and (allow_obj_fingers or held is not None):
                if (b2 if o1 else b1) in self.hand_bodies:
                    continue
            return True
        return False

    def fingers_touch_object(self):
        m, d = self.model, self.data
        hit_l = hit_r = False
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if self.obj in (b1, b2):
                o = b2 if b1 == self.obj else b1
                hit_l |= o in self.left
                hit_r |= o in self.right
        return hit_l and hit_r
