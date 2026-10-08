"""MuJoCo grasp-execution environment for evaluating part selections (env: graspsim).

Object: a free rigid body built from CoACD convex pieces of a (simplified) mesh, resting on a
floor plane after settling under gravity. Gripper: a floating parallel-jaw hand driven by a mocap
body (6-DoF), two finger pads on slide joints (Franka-like: 8 cm max opening, 2 cm wide pads,
force-limited). A trial: pre-grasp 10 cm back along the approach direction, move in, close,
lift 15 cm, hold. Success: the object rose >= 10 cm and is still between the fingers.

Grasps are given in the object's mesh frame as (c1, c2, approach) (metres), e.g. from
`fitpart.grasping.sample_grasps`; they are transformed with the settled object pose.
"""
import hashlib
import os
import pickle

import numpy as np

CACHE = os.path.expanduser("~/.cache/fitpart_coacd")
# Hand frame: x = closing axis, z = approach direction (towards the object). The fingertips are the
# forward-most point (z = 0); pads extend back to z = -0.045 and the palm to z = -0.065.
# Contacts are placed 8 mm behind the fingertips, the way a parallel gripper pinches a thin handle.
CONTACT_Z = -0.008


# ------------------------------------------------------------------ geometry
def convex_parts(mesh_path, scale=0.001, target_faces=3000, threshold=0.08, max_parts=16):
    """CoACD convex decomposition (cached). Returns list of (V, F) in metres, mesh frame."""
    import coacd
    import trimesh
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5(f"{mesh_path}|{scale}|{target_faces}|{threshold}|{max_parts}".encode()).hexdigest()
    path = os.path.join(CACHE, key + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    m = trimesh.load(mesh_path, force="mesh")
    m.apply_scale(scale)
    if len(m.faces) > target_faces:
        try:
            m = m.simplify_quadric_decimation(face_count=target_faces)
        except Exception:
            import open3d as o3d
            om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
            om = om.simplify_quadric_decimation(target_faces)
            m = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles))
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=threshold, max_convex_hull=max_parts,
                            preprocess_mode="auto")
    parts = [(np.asarray(v, float), np.asarray(f, int)) for v, f in parts]
    pickle.dump(parts, open(path, "wb"))
    return parts


def _mesh_xml(name, V):
    return f'<mesh name="{name}" vertex="{" ".join(f"{x:.6f}" for x in V.ravel())}"/>'


# per-category object masses (kg) for HANDAL; documented priors, identical for every method
HANDAL_MASS = {"adjustable_wrench": 0.3, "combinational_wrench": 0.2, "fixed_joint_pliers": 0.25, "hammer": 0.6,
               "ladle": 0.15, "locking_pliers": 0.4, "measuring_cup": 0.15, "mug": 0.35, "pots_pans": 1.0,
               "power_drill": 1.3, "ratchet": 0.3, "screwdriver": 0.12, "slip_joint_pliers": 0.25,
               "spatula": 0.1, "strainer": 0.2, "utensil": 0.1, "whisk": 0.1}

# six drop orientations (each axis of the mesh frame pointing up), averaged over in evaluation
def drop_orientations():
    Rs = []
    for ax in range(3):
        for sgn in (1, -1):
            up = np.zeros(3)
            up[ax] = sgn
            # rotation taking `up` to +z
            z = np.array([0, 0, 1.0])
            v = np.cross(up, z)
            c = up @ z
            if np.linalg.norm(v) < 1e-9:
                Rs.append(np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0]))
                continue
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            Rs.append(np.eye(3) + vx + vx @ vx / (1 + c))
    return Rs


def build_model(parts, density=None, mass=0.3, friction=1.0, max_open=0.08, finger_force=70.0):
    """density None: spread `mass` (kg) uniformly over the convex pieces' volume."""
    import mujoco
    import trimesh
    if density is None:
        vol = sum(trimesh.Trimesh(V, F).convex_hull.volume for V, F in parts)
        density = mass / max(vol, 1e-9)
    assets, geoms = [], []
    for k, (V, F) in enumerate(parts):
        assets.append(_mesh_xml(f"p{k}", V))
        geoms.append(f'<geom type="mesh" mesh="p{k}" density="{density}" friction="{friction} 0.02 0.002" '
                     f'condim="4" solref="0.004 1" solimp="0.95 0.99 0.001" rgba="0.7 0.6 0.5 1"/>')
    half_open = max_open / 2
    xml = f"""
<mujoco model="grasp">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" impratio="10" noslip_iterations="3"/>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" friction="1 0.02 0.002" rgba="0.8 0.8 0.8 1"/>
    <body name="obj" pos="0 0 0.3">
      <freejoint name="obj_free"/>
      {''.join(geoms)}
    </body>
    <body name="mocap" mocap="true" pos="0 0 1"/>
    <body name="hand" pos="0 0 1">
      <freejoint name="hand_free"/>
      <inertial pos="0 0 0" mass="0.7" diaginertia="0.002 0.002 0.002"/>
      <geom name="palm" type="box" size="{half_open + 0.012} 0.012 0.010" pos="0 0 -0.055" rgba="0.2 0.2 0.2 1"
            friction="1 0.02 0.002" condim="4"/>
      <body name="fl" pos="0 0 0">
        <joint name="jl" type="slide" axis="-1 0 0" range="0 {half_open}" damping="20"/>
        <geom name="pad_l" type="box" size="0.005 0.010 0.0225" pos="-0.005 0 -0.0225" friction="1.5 0.02 0.002"
              condim="4" solref="0.004 1" solimp="0.95 0.99 0.001" rgba="0.1 0.4 0.9 1" mass="0.05"/>
      </body>
      <body name="fr" pos="0 0 0">
        <joint name="jr" type="slide" axis="1 0 0" range="0 {half_open}" damping="20"/>
        <geom name="pad_r" type="box" size="0.005 0.010 0.0225" pos="0.005 0 -0.0225" friction="1.5 0.02 0.002"
              condim="4" solref="0.004 1" solimp="0.95 0.99 0.001" rgba="0.1 0.4 0.9 1" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <weld body1="mocap" body2="hand" solref="0.01 1"/>
  </equality>
  <actuator>
    <position name="al" joint="jl" kp="800" forcerange="-{finger_force} {finger_force}" ctrlrange="0 {half_open}"/>
    <position name="ar" joint="jr" kp="800" forcerange="-{finger_force} {finger_force}" ctrlrange="0 {half_open}"/>
  </actuator>
  <contact><exclude body1="hand" body2="fl"/><exclude body1="hand" body2="fr"/><exclude body1="fl" body2="fr"/></contact>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


# ------------------------------------------------------------------ helpers
def _quat_from_R(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def _R_from_quat(q):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


class GraspSim:
    def __init__(self, parts, **kw):
        import mujoco
        self.mj = mujoco
        self.model = build_model(parts, **kw)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.obj_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")]
        self.hand_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")]
        self.jl = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jl")]
        self.jr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jr")]
        self.open = m.actuator_ctrlrange[0, 1]
        self.lowest = min(V[:, 2].min() for V, _ in parts)
        self.parts = parts

    def _steps(self, n):
        for _ in range(n):
            self.mj.mj_step(self.model, self.data)

    def reset_and_settle(self, R0=np.eye(3), seconds=1.0):
        d = self.data
        self.mj.mj_resetData(self.model, d)
        # drop the object in orientation R0 so its lowest vertex is 5 mm above the floor
        allV = np.vstack([V for V, _ in self.parts]) @ R0.T
        d.qpos[self.obj_q:self.obj_q + 3] = [0, 0, -allV[:, 2].min() + 0.005]
        d.qpos[self.obj_q + 3:self.obj_q + 7] = _quat_from_R(R0)
        self._park_hand()
        self.mj.mj_forward(self.model, d)
        self._steps(int(seconds / self.model.opt.timestep))
        return self.obj_pose()

    def _park_hand(self):
        d = self.data
        d.mocap_pos[0] = [0, 0, 2.0]
        d.mocap_quat[0] = [1, 0, 0, 0]
        d.qpos[self.hand_q:self.hand_q + 3] = [0, 0, 2.0]
        d.qpos[self.hand_q + 3:self.hand_q + 7] = [1, 0, 0, 0]
        d.ctrl[:] = self.open
        d.qpos[self.jl] = d.qpos[self.jr] = self.open

    def obj_pose(self):
        q = self.data.qpos
        return q[self.obj_q:self.obj_q + 3].copy(), _R_from_quat(q[self.obj_q + 3:self.obj_q + 7])

    def clearance_ok(self, c1w, c2w, aw, approach_dist=0.10):
        """Hand frame for a world-frame grasp, and whether pre-grasp and grasp poses clear the floor."""
        x = c2w - c1w
        width = np.linalg.norm(x)
        if width < 1e-6 or width > 2 * self.open:
            return False
        x /= width
        z = aw - (aw @ x) * x
        if np.linalg.norm(z) < 1e-6:
            return False
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        Rh = np.stack([x, y, z], 1)
        grasp_pos = 0.5 * (c1w + c2w) - Rh @ np.array([0, 0, CONTACT_Z])
        for p in (grasp_pos - z * approach_dist, grasp_pos):
            corners = p + (Rh @ np.array([[sx * 0.05, sy * 0.012, sz] for sx in (-1, 1) for sy in (-1, 1)
                                           for sz in (0, -0.045, -0.065)]).T).T
            if corners[:, 2].min() < 0.002:
                return False
        return True

    def try_grasp(self, c1, c2, approach, pose, lift=0.15, approach_dist=0.10):
        """Execute one grasp given in the object frame; the object is at `pose` (t, R) after settling.
        Returns (success, info)."""
        mj, m, d = self.mj, self.model, self.data
        t, R = pose
        c1w, c2w = R @ c1 + t, R @ c2 + t
        aw = R @ (approach / np.linalg.norm(approach))
        x = c2w - c1w
        width = np.linalg.norm(x)
        if width > 2 * self.open:
            return False, "too wide"
        x /= width
        z = aw - (aw @ x) * x
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        Rh = np.stack([x, y, z], 1)          # hand frame columns: closing, width, approach
        mid = 0.5 * (c1w + c2w)
        grasp_pos = mid - Rh @ np.array([0, 0, CONTACT_Z])
        pre = grasp_pos - aw * approach_dist
        # reject poses that start inside the floor
        for p in (pre, grasp_pos):
            corners = p + (Rh @ np.array([[sx * 0.05, sy * 0.012, sz] for sx in (-1, 1) for sy in (-1, 1)
                                           for sz in (0, -0.045, -0.065)]).T).T
            if corners[:, 2].min() < 0.002:
                return False, "floor"
        quat = _quat_from_R(Rh)
        # teleport the hand to the pre-grasp, fingers open
        d.mocap_pos[0], d.mocap_quat[0] = pre, quat
        d.qpos[self.hand_q:self.hand_q + 3] = pre
        d.qpos[self.hand_q + 3:self.hand_q + 7] = quat
        d.qvel[self.hand_q:self.hand_q + 6] = 0
        d.ctrl[:] = self.open
        d.qpos[self.jl] = d.qpos[self.jr] = self.open
        mj.mj_forward(m, d)
        if self._hand_touching_object():
            return False, "pre-grasp collision"
        # approach
        steps = int(0.6 / m.opt.timestep)
        for i in range(steps):
            d.mocap_pos[0] = pre + (grasp_pos - pre) * (i + 1) / steps
            mj.mj_step(m, d)
        # close
        d.ctrl[:] = 0.0
        self._steps(int(0.5 / m.opt.timestep))
        z0 = self.obj_pose()[0][2]
        # lift
        steps = int(1.0 / m.opt.timestep)
        for i in range(steps):
            d.mocap_pos[0] = grasp_pos + np.array([0, 0, lift]) * (i + 1) / steps
            mj.mj_step(m, d)
        self._steps(int(1.0 / m.opt.timestep))  # hold
        z1 = self.obj_pose()[0][2]
        held = self._pads_touching_object()
        ok = (z1 - z0) >= 0.10 and held
        return bool(ok), f"dz={z1 - z0:.3f} held={held}"

    def _contacts_between(self, body_names):
        mj, m, d = self.mj, self.model, self.data
        ob = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "obj")
        bids = {mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, b) for b in body_names}
        hits = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 == ob and b2 in bids) or (b2 == ob and b1 in bids):
                hits.add(b2 if b1 == ob else b1)
        return hits

    def _hand_touching_object(self):
        return len(self._contacts_between(["hand", "fl", "fr"])) > 0

    def _pads_touching_object(self):
        return len(self._contacts_between(["fl", "fr"])) == 2


def evaluate_part(sim, part_pts, part_nrm, obj_pts, n_grasps=10, seed=0, R0=np.eye(3), pool=80):
    """Shared protocol for any part selection (points/normals in the mesh frame, metres):
    settle the object, move everything to the world frame, sample a pool of collision-free
    antipodal grasps with `fitpart.grasping.sample_grasps`, keep those whose pre-grasp and grasp
    poses clear the floor, and execute the first n_grasps.
    Returns dict(executable=#executed, success=#lifted, sampled=#pool, reasons=[...])."""
    from fitpart.grasping import sample_grasps
    t, R = sim.reset_and_settle(R0)
    Pw, Nw, Ow = part_pts @ R.T + t, part_nrm @ R.T, obj_pts @ R.T + t
    grasps = sample_grasps(Pw, Nw, Ow, n_contacts=max(400, 8 * pool), max_grasps=pool, seed=seed)
    ok_g = [g for g in grasps if sim.clearance_ok(g[0], g[1], g[2] / np.linalg.norm(g[2]))][:n_grasps]
    succ, reasons = 0, []
    ident = (np.zeros(3), np.eye(3))
    for c1, c2, a in ok_g:
        sim.reset_and_settle(R0)  # deterministic: same settled pose every trial
        ok, why = sim.try_grasp(c1, c2, a, ident)
        succ += ok
        reasons.append(why)
    return dict(executable=len(ok_g), success=succ, sampled=len(grasps), reasons=reasons)


# ------------------------------------------------------------------ screw tasks
def axis_angle(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def screw_transform(axis, point, theta, pitch=0.0):
    """g = exp(xi theta) for a screw (axis through point, pitch); same as ITSI's
    get_transformation_for_screw: R = Rot(axis, theta), p = (I - R) q + pitch theta axis."""
    R = axis_angle(axis, theta)
    q = np.asarray(point, float)
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return R, (np.eye(3) - R) @ q + pitch * theta * a


def pivot_screw(sim, pose, side=1):
    """Pivoting task in the spirit of ITSI's hand-set screws: axis along the bottom edge of the
    settled object's horizontal bounding box (parallel to its long side), pitch 0. `side` picks
    which of the two long bottom edges. Returns (axis, point) in the world frame."""
    t, R = pose
    V = np.vstack([V for V, _ in sim.parts]) @ R.T + t
    xy = V[:, :2] - V[:, :2].mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    long_ax = np.r_[vt[0], 0.0]
    short_ax = np.r_[vt[1], 0.0]
    s = (V - V.mean(0)) @ short_ax
    edge = V.mean(0) + short_ax * (s.max() if side > 0 else s.min())
    point = np.r_[edge[:2], V[:, 2].min()]
    # rotating about +long_ax by +theta lifts the far side when the edge is on the -short side
    axis = long_ax if side < 0 else -long_ax
    return axis, point


def _try_screw(self, c1, c2, approach, axis, point, theta=np.pi / 2, pitch=0.0, approach_dist=0.10,
               tol_angle_deg=20.0, tol_pos=0.03, seconds=2.0):
    """Grasp (no lift), then drive the hand along the screw exp(xi theta) applied to the grasp pose
    (ITSI pivoting_motion) while the environment contact is handled by the physics.
    Success: final object pose within tol of the commanded screw displacement, still held."""
    mj, m, d = self.mj, self.model, self.data
    x = c2 - c1
    width = np.linalg.norm(x)
    if width > 2 * self.open:
        return False, "too wide"
    x /= width
    aw = approach / np.linalg.norm(approach)
    z = aw - (aw @ x) * x
    z /= np.linalg.norm(z)
    Rh = np.stack([x, np.cross(z, x), z], 1)
    grasp_pos = 0.5 * (c1 + c2) - Rh @ np.array([0, 0, CONTACT_Z])
    pre = grasp_pos - z * approach_dist
    quat = _quat_from_R(Rh)
    d.mocap_pos[0], d.mocap_quat[0] = pre, quat
    d.qpos[self.hand_q:self.hand_q + 3] = pre
    d.qpos[self.hand_q + 3:self.hand_q + 7] = quat
    d.qvel[self.hand_q:self.hand_q + 6] = 0
    d.ctrl[:] = self.open
    d.qpos[self.jl] = d.qpos[self.jr] = self.open
    mj.mj_forward(m, d)
    if self._hand_touching_object():
        return False, "pre-grasp collision"
    steps = int(0.6 / m.opt.timestep)
    for i in range(steps):
        d.mocap_pos[0] = pre + (grasp_pos - pre) * (i + 1) / steps
        mj.mj_step(m, d)
    d.ctrl[:] = 0.0
    self._steps(int(0.5 / m.opt.timestep))
    t0, R0 = self.obj_pose()
    steps = int(seconds / m.opt.timestep)
    for i in range(steps):
        Rs, ps = screw_transform(axis, point, theta * (i + 1) / steps, pitch)
        d.mocap_pos[0] = Rs @ grasp_pos + ps
        d.mocap_quat[0] = _quat_from_R(Rs @ Rh)
        mj.mj_step(m, d)
    self._steps(int(0.5 / m.opt.timestep))
    t1, R1 = self.obj_pose()
    Rs, ps = screw_transform(axis, point, theta, pitch)
    R_exp, t_exp = Rs @ R0, Rs @ t0 + ps
    ang = np.degrees(np.arccos(np.clip((np.trace(R_exp.T @ R1) - 1) / 2, -1, 1)))
    dpos = np.linalg.norm(t1 - t_exp)
    held = self._pads_touching_object()
    ok = ang <= tol_angle_deg and dpos <= tol_pos and held
    return bool(ok), f"ang={ang:.0f} dpos={dpos:.3f} held={held}"


GraspSim.try_screw = _try_screw


def evaluate_part_screw(sim, part_pts, part_nrm, obj_pts, n_grasps=10, seed=0, R0=np.eye(3), pool=80,
                        side=1, theta=np.pi / 2):
    """Screw-task version of evaluate_part: same sampling and floor filtering, then the pivoting
    task about the settled object's bottom edge."""
    from fitpart.grasping import sample_grasps
    t, R = sim.reset_and_settle(R0)
    axis, point = pivot_screw(sim, (t, R), side)
    Pw, Nw, Ow = part_pts @ R.T + t, part_nrm @ R.T, obj_pts @ R.T + t
    grasps = sample_grasps(Pw, Nw, Ow, n_contacts=max(400, 8 * pool), max_grasps=pool, seed=seed)
    ok_g = [g for g in grasps if sim.clearance_ok(g[0], g[1], g[2] / np.linalg.norm(g[2]))][:n_grasps]
    succ, reasons = 0, []
    for c1, c2, a in ok_g:
        sim.reset_and_settle(R0)
        ok, why = sim.try_screw(c1, c2, a, axis, point, theta)
        succ += ok
        reasons.append(why)
    return dict(executable=len(ok_g), success=succ, sampled=len(grasps), reasons=reasons, axis=axis, point=point)
