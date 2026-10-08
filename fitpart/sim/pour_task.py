"""'Pour from the mug' as a task screw, executed in MuJoCo (env: graspsim).

Task (defined by the evaluator from GT geometry): stand the mug upright; lift 10 cm (translation,
pitch = inf); rotate 90 deg (pitch 0) about the horizontal axis through the rim point opposite the
handle, perpendicular to the handle direction, so it pours away from the handle.
Success: final object pose within 20 deg / 3 cm of the commanded motion and still held.
"""
import numpy as np

from .mujoco_grasp import CONTACT_Z, _quat_from_R, axis_angle, drop_orientations


def drop_orientations_26():
    """Rotations putting each of 26 directions (6 faces, 12 edges, 8 corners of a cube) up."""
    from ..axes import rotation_to_up
    dirs = [np.array(v, float) for v in
            [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]]
    return [rotation_to_up(d / np.linalg.norm(d)) for d in dirs]


def upright_pose_support(sim, P, contact_tol=0.003):
    """Physical, label-free 'upright': settle from 26 drop orientations and keep the pose with the
    most surface on the table (points within contact_tol of the floor). A mug's flat base beats
    lying on its side (a line) or standing on its rim (a thin ring). Returns (R0, pose, n_contact)."""
    best = None
    for R0 in drop_orientations_26():
        t, R = sim.reset_and_settle(R0)
        W = P @ R.T + t
        n_contact = int((W[:, 2] < W[:, 2].min() + contact_tol).sum())
        if best is None or n_contact > best[2]:
            best = (R0, (t, R), n_contact)
    return best


def upright_pose_revolution(sim, P, N):
    """Label-free upright: rotational-symmetry axis of the cloud (axes.revolution_axis), closed end
    down (axes.base_down_sign), dropped in that orientation and settled. Returns (R0, pose, score)."""
    from ..axes import closed_end_sign, revolution_axis, rotation_to_up
    a, c, score = revolution_axis(P, N)
    a = a * closed_end_sign(P, a, c)
    R0 = rotation_to_up(a)
    return R0, sim.reset_and_settle(R0), score


def upright_pose_axis(sim, P, N):
    """Stand the object on its normal-tensor axis, closed end (base) down. Label-free.
    Returns (R0, settled pose, axis confidence)."""
    from ..axes import base_down_sign, normal_tensor_axis, rotation_to_up
    axis, conf, _ = normal_tensor_axis(N)
    axis = axis * base_down_sign(P, N, axis)
    R0 = rotation_to_up(axis)
    return R0, sim.reset_and_settle(R0), conf


def upright_pose(sim, P, handle_mask):
    """Among the drop orientations, the settled pose with the handle at mid-height and the rim up
    (a cup's bottom slice holds more surface than its open top slice)."""
    best = None
    for R0 in drop_orientations():
        t, R = sim.reset_and_settle(R0)
        W = P @ R.T + t
        z = W[:, 2]
        lo, hi = z.min(), z.max()
        body = W[~handle_mask]
        hz = (W[handle_mask, 2].mean() - lo) / max(hi - lo, 1e-9)
        bot = (body[:, 2] < lo + 0.12 * (hi - lo)).sum()
        top = (body[:, 2] > hi - 0.12 * (hi - lo)).sum()
        score = abs(hz - 0.5) + (0.5 if top > bot else 0.0)
        if best is None or score < best[0]:
            best = (score, R0, t, R)
    return best[1], (best[2], best[3])


def pour_screw_from(W, side_point, body_mask=None, lift=0.10):
    """Pour screw without labels: rotate about the rim on the side opposite `side_point` (the
    method's own handle estimate, or the grasp itself when a method has no part). body_mask =
    the points treated as the vessel (default all)."""
    body = W if body_mask is None else W[body_mask]
    c = body.mean(0)
    h = np.asarray(side_point, float) - c
    h[2] = 0
    h /= np.linalg.norm(h) + 1e-12
    up = np.array([0, 0, 1.0])
    axis = np.cross(h, up)
    axis /= np.linalg.norm(axis)
    far = body[np.argmax((body - c) @ -h)]
    point = np.array([far[0], far[1], body[:, 2].max()]) + up * lift
    return axis, point, h


def pour_screw(W, handle_mask, lift=0.10):
    """(axis, point) of the pour rotation, after the lift, in the world frame."""
    body, handle = W[~handle_mask], W[handle_mask]
    c = body.mean(0)
    h = handle.mean(0) - c
    h[2] = 0
    h /= np.linalg.norm(h)
    up = np.array([0, 0, 1.0])
    axis = np.cross(h, up)
    axis /= np.linalg.norm(axis)
    far = body[np.argmax((body - c) @ -h)]
    point = np.array([far[0], far[1], body[:, 2].max()]) + up * lift
    return axis, point


def try_pour(sim, c1, c2, approach, axis, point, lift=0.10, theta=np.pi / 2, approach_dist=0.10,
             tol_angle_deg=20.0, tol_pos=0.03, recorder=None):
    mj, m, d = sim.mj, sim.model, sim.data
    x = c2 - c1
    width = np.linalg.norm(x)
    if width > 2 * sim.open:
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
    d.qpos[sim.hand_q:sim.hand_q + 3] = pre
    d.qpos[sim.hand_q + 3:sim.hand_q + 7] = quat
    d.qvel[sim.hand_q:sim.hand_q + 6] = 0
    d.ctrl[:] = sim.open
    d.qpos[sim.jl] = d.qpos[sim.jr] = sim.open
    mj.mj_forward(m, d)
    if sim._hand_touching_object():
        return False, "pre-grasp collision"
    dt = m.opt.timestep

    def run(n, update=None):
        for i in range(n):
            if update is not None:
                update(i, n)
            mj.mj_step(m, d)
            if recorder is not None:
                recorder(d)

    run(int(0.6 / dt), lambda i, n: d.mocap_pos.__setitem__(0, pre + (grasp_pos - pre) * (i + 1) / n))
    d.ctrl[:] = 0.0
    run(int(0.5 / dt))
    t0, R0 = sim.obj_pose()
    up = np.array([0, 0, lift])
    run(int(0.8 / dt), lambda i, n: d.mocap_pos.__setitem__(0, grasp_pos + up * (i + 1) / n))
    g1 = grasp_pos + up

    def rot(i, n):
        Rs = axis_angle(axis, theta * (i + 1) / n)
        d.mocap_pos[0] = Rs @ (g1 - point) + point
        d.mocap_quat[0] = _quat_from_R(Rs @ Rh)

    run(int(1.5 / dt), rot)
    run(int(0.7 / dt))
    t1, R1 = sim.obj_pose()
    Rs = axis_angle(axis, theta)
    R_exp = Rs @ R0
    t_exp = Rs @ (t0 + up - point) + point
    ang = np.degrees(np.arccos(np.clip((np.trace(R_exp.T @ R1) - 1) / 2, -1, 1)))
    dpos = np.linalg.norm(t1 - t_exp)
    held = sim._pads_touching_object()
    ok = ang <= tol_angle_deg and dpos <= tol_pos and held
    return bool(ok), f"ang={ang:.0f} dpos={dpos:.3f} held={held}"
