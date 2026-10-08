"""Procedural articulated objects for the task-screw trials (PartNet-Mobility meshes and joints need a
SAPIEN account, not available here). Each object stands on the table in front of the robot under a
random yaw and random dimensions, has ONE task joint with known ground truth, and real-world-like
geometric cues (a lid overhangs its box by a lip, a cap is wider than the bottle neck, handles, knobs
and buttons protrude). Surface samples carry part labels for diagnostics only; BITSI sees points and
normals.

kinds (task):
  bottle  (turn)    fixed bottle, neck, cap on a hinge about the bottle axis
  lidbox  (hinge)   fixed box, lid hinged along its back edge, knob near the front edge
  drawer  (slide)   fixed cabinet, drawer sliding out of the front face, handle bar on the front
  button  (press)   fixed panel, spring button on top
  tongs   (squeeze) lower arm fixed on the table, upper arm hinged at one end, spring-opened 25 deg
  pivotbox (pivot)  a free box (rigid): tilt it about its bottom edge
  knife   (cut)     a free knife lying on the table (rigid): handle + blade
"""
from dataclasses import dataclass, field

import numpy as np

TABLE_TOP = 0.0
CENTRE = np.array([0.55, 0.0])
KINDS = ("bottle", "lidbox", "drawer", "button", "tongs", "pivotbox", "knife")


@dataclass
class Art:
    kind: str
    yaw: float
    dims: dict
    joint: str = None                       # task joint name (None for rigid kinds)
    gt_axis: np.ndarray = None              # world, unit
    gt_point: np.ndarray = None
    gt_pitch: float = 0.0                   # inf for prismatic
    geoms: list = field(default_factory=list)   # (part label, geom name, type, size, local pose) for sampling


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _quat_z(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def build(spec, kind, seed=0):
    """Add the object to an MjSpec (before compile). Returns an Art with the ground-truth joint."""
    import mujoco
    rng = np.random.default_rng(seed)
    yaw = float(rng.uniform(-np.pi / 3, np.pi / 3))
    R = _rot(yaw)
    wb = spec.worldbody
    G = mujoco.mjtGeom
    root = wb.add_body(name="art", pos=[CENTRE[0], CENTRE[1], TABLE_TOP], quat=_quat_z(yaw))
    a = Art(kind, yaw, {})
    grey, blue, red = [0.8, 0.8, 0.82, 1], [0.35, 0.5, 0.8, 1], [0.8, 0.35, 0.3, 1]

    def geom(body, label, name, gtype, size, pos, rgba, **kw):
        body.add_geom(name=name, type=gtype, size=list(size) + [0] * (3 - len(size)), pos=list(pos), rgba=rgba,
                      friction=[1.2, 0.02, 0.002], condim=4, **kw)
        a.geoms.append((label, name))

    if kind == "bottle":
        r, h = rng.uniform(0.032, 0.042), rng.uniform(0.14, 0.20)
        rn, hn = rng.uniform(0.012, 0.016), 0.03
        rc, hc = rng.uniform(0.018, 0.024), rng.uniform(0.022, 0.03)
        a.dims = dict(r=r, h=h, rn=rn, rc=rc, hc=hc)
        geom(root, "body", "bottle_body", G.mjGEOM_CYLINDER, (r, h / 2), (0, 0, h / 2), grey)
        geom(root, "neck", "bottle_neck", G.mjGEOM_CYLINDER, (rn, hn / 2), (0, 0, h + hn / 2), grey)
        cap = root.add_body(name="cap", pos=[0, 0, h + hn])
        cap.add_joint(name="cap_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.02,
                      frictionloss=0.01)
        geom(cap, "cap", "cap_geom", G.mjGEOM_CYLINDER, (rc, hc / 2), (0, 0, hc / 2), blue, mass=0.02)
        a.joint, a.gt_axis, a.gt_point = "cap_joint", np.array([0, 0, 1.0]), np.r_[CENTRE, h + hn]
    elif kind == "lidbox":
        w, d, h = rng.uniform(0.14, 0.22), rng.uniform(0.11, 0.16), rng.uniform(0.07, 0.12)
        lip, t = 0.006, 0.012
        a.dims = dict(w=w, d=d, h=h)
        geom(root, "body", "box_body", G.mjGEOM_BOX, (d / 2, w / 2, h / 2), (0, 0, h / 2), grey)
        # hinge along the back edge (+x in the object frame, away from the robot at yaw 0)
        lid = root.add_body(name="lid", pos=[d / 2 + lip, 0, h])
        lid.add_joint(name="lid_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[0, 2.0],
                      damping=0.05)
        geom(lid, "lid", "lid_geom", G.mjGEOM_BOX, ((d + 2 * lip) / 2, (w + 2 * lip) / 2, t / 2),
             (-(d / 2 + lip), 0, t / 2), blue, mass=0.05)
        geom(lid, "knob", "knob_geom", G.mjGEOM_BOX, (0.008, 0.02, 0.012), (-(d + 2 * lip) + 0.018, 0, t + 0.012),
             red, mass=0.01)
        a.joint = "lid_joint"
        a.gt_axis = R @ np.array([0, 1, 0.0])   # +angle lifts the front (knob) edge
        a.gt_point = np.r_[CENTRE, 0] + R @ np.array([d / 2 + lip, 0, h])
    elif kind == "drawer":
        w, d, h = rng.uniform(0.22, 0.30), rng.uniform(0.20, 0.26), rng.uniform(0.10, 0.16)
        a.dims = dict(w=w, d=d, h=h)
        # front face = -x (towards the robot at yaw 0); drawer slides along -x
        geom(root, "body", "cab_body", G.mjGEOM_BOX, (d / 2, w / 2, h / 2), (0, 0, h / 2), grey)
        dr = root.add_body(name="drawer", pos=[-d / 2, 0, h / 2])
        dr.add_joint(name="drawer_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[0, 0.16],
                     damping=2.0)
        geom(dr, "front", "drawer_front", G.mjGEOM_BOX, (0.008, w / 2 - 0.01, h / 2 - 0.01), (-0.008, 0, 0), blue,
             mass=0.1)
        geom(dr, "handle", "drawer_handle", G.mjGEOM_BOX, (0.008, 0.04, 0.007), (-0.035, 0, 0), red, mass=0.02)
        geom(dr, "handle", "handle_post1", G.mjGEOM_BOX, (0.012, 0.005, 0.005), (-0.022, 0.032, 0), red, mass=0.005)
        geom(dr, "handle", "handle_post2", G.mjGEOM_BOX, (0.012, 0.005, 0.005), (-0.022, -0.032, 0), red, mass=0.005)
        a.joint, a.gt_axis, a.gt_pitch = "drawer_joint", R @ np.array([-1, 0, 0.0]), np.inf
        a.gt_point = np.r_[CENTRE, h / 2] + R @ np.array([-d / 2, 0, 0])
    elif kind == "button":
        w, d, h = rng.uniform(0.12, 0.18), rng.uniform(0.10, 0.14), rng.uniform(0.05, 0.08)
        rb = rng.uniform(0.012, 0.018)
        a.dims = dict(w=w, d=d, h=h, rb=rb)
        geom(root, "body", "panel_body", G.mjGEOM_BOX, (d / 2, w / 2, h / 2), (0, 0, h / 2), grey)
        bx = rng.uniform(-d / 4, d / 4)
        bt = root.add_body(name="button", pos=[bx, 0, h])
        bt.add_joint(name="button_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, -1], range=[0, 0.01],
                     stiffness=300.0, damping=1.0)
        geom(bt, "button", "button_geom", G.mjGEOM_CYLINDER, (rb, 0.008), (0, 0, 0.008), red, mass=0.01)
        a.joint, a.gt_axis, a.gt_pitch = "button_joint", np.array([0, 0, -1.0]), np.inf
        a.gt_point = np.r_[CENTRE, h] + R @ np.array([bx, 0, 0])
    elif kind == "tongs":
        L = rng.uniform(0.16, 0.22)
        a.dims = dict(L=L)
        geom(root, "lower_arm", "tong_lower", G.mjGEOM_BOX, (L / 2, 0.012, 0.006), (0, 0, 0.006), grey)
        geom(root, "hinge_block", "tong_block", G.mjGEOM_BOX, (0.012, 0.014, 0.02), (L / 2 - 0.012, 0, 0.02), grey)
        up = root.add_body(name="upper", pos=[L / 2 - 0.012, 0, 0.035])
        up.add_joint(name="tong_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-0.45, 0.05],
                     stiffness=0.4, springref=0.0, damping=0.02)
        geom(up, "upper_arm", "tong_upper", G.mjGEOM_BOX, (L / 2, 0.012, 0.006), (-L / 2 + 0.012, 0, 0), blue, mass=0.03)
        # opened 25 deg (rotation about +y lifts the free end, which lies towards -x)
        up.quat = [np.cos(0.22), 0, np.sin(0.22), 0]
        a.joint, a.gt_axis = "tong_joint", R @ np.array([0, 1, 0.0])
        a.gt_point = np.r_[CENTRE, 0.035] + R @ np.array([L / 2 - 0.012, 0, 0])
    elif kind == "pivotbox":
        w, d, h = rng.uniform(0.06, 0.08), rng.uniform(0.10, 0.16), rng.uniform(0.14, 0.20)
        a.dims = dict(w=w, d=d, h=h)
        root.add_freejoint(name="art_free")
        geom(root, "body", "pbox", G.mjGEOM_BOX, (d / 2, w / 2, h / 2), (0, 0, h / 2), grey, mass=0.25)
    elif kind == "knife":
        Lh, Lb = rng.uniform(0.09, 0.12), rng.uniform(0.10, 0.15)
        a.dims = dict(Lh=Lh, Lb=Lb)
        root.add_freejoint(name="art_free")
        geom(root, "handle", "knife_handle", G.mjGEOM_BOX, (Lh / 2, 0.011, 0.009), (-Lh / 2, 0, 0.009), red, mass=0.06)
        geom(root, "blade", "knife_blade", G.mjGEOM_BOX, (Lb / 2, 0.012, 0.0015), (Lb / 2, 0, 0.0015), grey, mass=0.04)
    else:
        raise ValueError(kind)
    # moving parts rest on / slide against their base: no friction from those contacts
    for child in ("cap", "lid", "drawer", "button", "upper"):
        if any(b.name == child for b in spec.bodies):
            spec.add_exclude(bodyname1="art", bodyname2=child)
    return a


def sample_surface(model, data, art, density=4.0e5, seed=0):
    """Points, outward normals and part labels on the object's geoms (world frame), ~density per m^2."""
    import mujoco
    rng = np.random.default_rng(seed)
    P, N, L = [], [], []
    for label, name in art.geoms:
        g = model.geom(name).id
        t = model.geom_type[g]
        s = model.geom_size[g]
        Rg = data.geom_xmat[g].reshape(3, 3)
        c = data.geom_xpos[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            faces = []
            for ax in range(3):
                for sg in (-1, 1):
                    o = [k for k in range(3) if k != ax]
                    area = 4 * s[o[0]] * s[o[1]]
                    faces.append((ax, sg, o, area))
            for ax, sg, o, area in faces:
                n = max(int(area * density), 4)
                u = rng.uniform(-1, 1, (n, 2))
                p = np.zeros((n, 3))
                p[:, ax] = sg * s[ax]
                p[:, o[0]] = u[:, 0] * s[o[0]]
                p[:, o[1]] = u[:, 1] * s[o[1]]
                nn = np.zeros((n, 3))
                nn[:, ax] = sg
                P.append(p @ Rg.T + c)
                N.append(nn @ Rg.T)
                L += [label] * n
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, hh = s[0], s[1]
            n_side = max(int(2 * np.pi * r * 2 * hh * density), 8)
            th = rng.uniform(0, 2 * np.pi, n_side)
            z = rng.uniform(-hh, hh, n_side)
            p = np.stack([r * np.cos(th), r * np.sin(th), z], 1)
            nn = np.stack([np.cos(th), np.sin(th), np.zeros(n_side)], 1)
            P.append(p @ Rg.T + c)
            N.append(nn @ Rg.T)
            L += [label] * n_side
            for sg in (-1, 1):
                n_cap = max(int(np.pi * r * r * density), 4)
                rr = r * np.sqrt(rng.uniform(0, 1, n_cap))
                th = rng.uniform(0, 2 * np.pi, n_cap)
                p = np.stack([rr * np.cos(th), rr * np.sin(th), np.full(n_cap, sg * hh)], 1)
                nn = np.tile([0, 0, sg], (n_cap, 1)).astype(float)
                P.append(p @ Rg.T + c)
                N.append(nn @ Rg.T)
                L += [label] * n_cap
    P, N, L = np.vstack(P), np.vstack(N), np.array(L)
    keep = P[:, 2] > TABLE_TOP + 0.001          # faces resting on the table are not observable
    return P[keep], N[keep], L[keep]
