"""SALSA on one object cloud, no ROS dependency: BITSI v4 parts -> task screw label -> end-effector pose sequence.

salsa_plan(points, command) runs fitpart/tasks/screws.py: analyze() segments the cloud with BITSI v4 (gravity axis
offered, robot setting), the task rule (turn, hinge, slide, press, squeeze, pivot, lift, swing, scoop, stir, cut,
handover, fasten) picks the part to grasp and the screw (axis, point, pitch, magnitude) that part should follow
(Lynch & Park). Antipodal grasps are sampled on that part (fitpart/grasping.py) and each candidate becomes a pose
sequence: pre-grasp -> grasp (close) -> optional straight lift -> screw motion -> release, exactly as tasks/run.py
builds its Cartesian targets (screw_poses), but WITHOUT IK, collision checks or the SOCP grasp ranking (no robot model
or object mass here). Candidates are ordered by a geometric heuristic (top-down approach first, grasp near the part
centre); the downstream motion planner takes the first one it can reach.

Pose convention: robot frame of the cloud (base_link); position = the point midway between the two finger contacts
(pinch point); rotation columns = hand x, y = closing direction, z = approach (towards the object), as
fitpart/sim/panda_scene.grasp_frame. Add your gripper's pinch-point-to-TCP offset along z before planning.
"""
import re
import numpy as np

TASKS = ('turn', 'hinge', 'slide', 'press', 'squeeze', 'pivot', 'lift', 'swing', 'scoop', 'stir', 'cut', 'handover', 'fasten')
DEFAULT_MAG = dict(lift=15, pivot=50, turn=180, hinge=70, slide=12, press=0.7, swing=40, scoop=35, stir=120, cut=10, squeeze=20,
                   fasten=90, handover=None)
LINEAR = ('lift', 'slide', 'press', 'cut')   # magnitudes in cm; others in degrees
ALIASES = dict(open='hinge', close='hinge', rotate='turn', twist='turn', unscrew='turn', screw='fasten', pull='slide',
               push='press', pick='lift', raise_='lift', hammer='swing', pour='pivot', tilt='pivot', give='handover', hand='handover')


def parse_command(text):
    """'lift 10 cm' / 'turn 90 deg reverse' / 'hinge' -> dict(task, magnitude, unit, direction); None if no task word."""
    t = str(text or '').lower().strip()
    if not t: return None
    words = re.findall(r'[a-z]+', t); task = next((w for w in words if w in TASKS), None)
    if task is None: task = next((ALIASES[w] for w in words if w in ALIASES), None)
    if task is None: raise ValueError(f'no task in {text!r}; use one of {", ".join(TASKS)}')
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*(degrees|degree|deg|°|centimeters|centimetres|cm|mm|m\b)?', t)
    mag, unit = (None, None) if m is None else (float(m.group(1)), m.group(2) or ('cm' if task in LINEAR else 'deg'))
    if unit in ('degrees', 'degree', '°'): unit = 'deg'
    if unit in ('centimeters', 'centimetres'): unit = 'cm'
    if unit == 'mm': mag, unit = mag / 10, 'cm'
    if unit == 'm': mag, unit = mag * 100, 'cm'
    direction = -1 if any(w in words for w in ('reverse', 'back', 'close', 'tighten', 'down', 'counter')) else 1
    if mag is None: mag, unit = DEFAULT_MAG.get(task), ('cm' if task in LINEAR else 'deg')
    return dict(task=task, magnitude=mag, unit=unit, direction=direction)


def apply_spec(ts, spec):
    """As fitpart/tasks/run.apply_spec: BITSI fixes axis, point and positive sense; the command sets size and sense."""
    mag = spec.get('magnitude')
    if mag is not None and spec.get('unit') in ('deg', 'cm'):
        val = np.deg2rad(mag) if spec['unit'] == 'deg' else mag / 100.0
        if not np.isfinite(ts.pitch) or spec['unit'] == 'deg': ts.theta = np.sign(ts.theta or 1.0) * val
    ts.theta *= spec.get('direction', 1)
    return ts


def rot_about(axis, angle):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def grasp_frame(c1, c2, approach):
    y = (c2 - c1) / np.linalg.norm(c2 - c1); z = approach / np.linalg.norm(approach)
    z = z - (z @ y) * y; z /= np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z], 1)


def quat_xyzw(R):
    w = np.sqrt(max(0., 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0., 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0., 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0., 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2]); y = np.copysign(y, R[0, 2] - R[2, 0]); z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w]); return (q / np.linalg.norm(q)).tolist()


def screw_poses(ts, p0, R0, up, n=12):
    """tasks/run.screw_poses: rotation theta*f about axis through point (+ lift), plus pitch*theta*f along the axis."""
    out = []
    for k in range(1, n + 1):
        f = k / n
        if not np.isfinite(ts.pitch): out.append((p0 + ts.axis * ts.theta * f, R0))
        else:
            Rk = rot_about(ts.axis, ts.theta * f); q = ts.point + (up * ts.lift if ts.lift else 0)
            out.append((Rk @ (p0 - q) + q + ts.pitch * ts.theta * f * ts.axis, Rk @ R0))
    return out


def pose(name, p, R, gripper):
    return dict(name=name, position=[float(v) for v in p], quaternion_xyzw=quat_xyzw(R), gripper=gripper)


def salsa_plan(points, command, voxel=0.004, n_candidates=5, approach_dist=0.10, retreat_dist=0.08, n_screw=12, seed=0):
    import open3d as o3d
    from bitsi_core import prepare
    from fitpart.tasks import screws as S
    from fitpart.grasping import sample_grasps
    spec = parse_command(command) if isinstance(command, str) else dict(command)
    pcd = prepare(points, voxel=voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30)); W = np.asarray(pcd.points); N = np.asarray(pcd.normals).copy()
    N[((W - W.mean(0)) * N).sum(1) < 0] *= -1
    if spec['task'] not in S.RULES:
        raise ValueError(f"task '{spec['task']}' is not in this build's fitpart/tasks/screws.py (has: {', '.join(S.RULES)})")
    A = S.analyze(W, N)
    ts = S.RULES[spec['task']](W, A)
    parts = [dict(name=f'part_{i}', kind=p.kind, order=int(p.order), n_points=int(len(p.idx)), centroid=W[p.idx].mean(0).tolist(),
                  points=W[p.idx]) for i, p in enumerate(A['parts'])]
    out = dict(command=command, spec=spec, n_points=int(len(W)), parts=parts, axis_root=np.asarray(A['axis']).tolist())
    if ts is None:
        out['error'] = f"rule '{spec['task']}' found no suitable part (e.g. no handle/child part for slide or press)"; return out
    ts = apply_spec(ts, spec); up = S.UP
    g = np.asarray(ts.grasp, bool)
    out['screw'] = dict(task=ts.task, axis=np.asarray(ts.axis, float).tolist(), point=np.asarray(ts.point, float).tolist(),
                        pitch=None if not np.isfinite(ts.pitch) else float(ts.pitch), translation=not np.isfinite(ts.pitch),
                        theta=float(ts.theta), theta_unit='m' if not np.isfinite(ts.pitch) else 'rad', lift=float(ts.lift),
                        push=bool(ts.push), across=bool(ts.across), grasp_points=int(g.sum()),
                        info={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ts.info.items()})
    out['grasp_part_points'] = W[g]
    cands = []
    if ts.push:   # press: fingers closed, push along the axis onto the top point
        z = np.asarray(ts.axis, float); x = np.cross(z, [1, 0, 0] if abs(z[0]) < .9 else [0, 1, 0]); x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z], 1); top = np.asarray(ts.point, float)
        seq = [pose('pre_press', top - z * approach_dist, R, 'closed'), pose('touch', top, R, 'closed')]
        seq += [pose(f'press_{k}', top + z * ts.theta * s, R, 'closed') for k, s in enumerate(np.linspace(.25, 1, 4))]
        seq.append(pose('retreat', top - z * retreat_dist, R, 'closed'))
        cands.append(dict(rank=0, contacts=None, score=None, poses=seq))
    else:
        gs = []
        for across in ((True, False) if ts.across else (False, True)):
            gs += sample_grasps(W[g], N[g], W, n_contacts=2000 if across else 1000, max_grasps=120, seed=seed, across_part=across)
        cg = W[g].mean(0) if g.any() else W.mean(0); ext = np.ptp(W, 0).max() + 1e-6
        def score(c):
            c1, c2, app = c; a = np.asarray(app, float); a /= np.linalg.norm(a)
            return float(-a @ up) - float(np.linalg.norm(0.5 * (c1 + c2) - cg)) / ext   # top-down first, then central
        order = sorted(range(len(gs)), key=lambda i: -score(gs[i]))[:n_candidates]
        for r, i in enumerate(order):
            c1, c2, app = gs[i]; Rg = grasp_frame(c1, c2, app); mid = 0.5 * (c1 + c2); z = Rg[:, 2]
            seq = [pose('pre_grasp', mid - z * approach_dist, Rg, 'open'), pose('grasp', mid, Rg, 'close')]
            p_last = mid
            if ts.lift:
                seq += [pose(f'lift_{k}', mid + up * ts.lift * s, Rg, 'hold') for k, s in enumerate(np.linspace(.2, 1, 5))]
                p_last = mid + up * ts.lift
            screw = screw_poses(ts, p_last, Rg, up, n_screw)
            seq += [pose(f'screw_{k}', p, R, 'hold') for k, (p, R) in enumerate(screw)]
            pe, Re = screw[-1] if screw else (p_last, Rg)
            seq += [pose('release', pe, Re, 'open'), pose('retreat', pe - Re[:, 2] * retreat_dist, Re, 'open')]
            cands.append(dict(rank=r, contacts=[np.asarray(c1).tolist(), np.asarray(c2).tolist()], score=score(gs[i]),
                              width=float(np.linalg.norm(np.asarray(c2) - np.asarray(c1))), poses=seq))
        if not cands: out['error'] = 'no antipodal grasp found on the selected part'
    out['candidates'] = cands
    return out
