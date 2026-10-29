"""Run every BITSI part of one object through tograsp-socp (github.com/apat20/tograsp-socp), plot the result.

For each part saved by the node (BITSI_OUT/<request>_<id>_<label>/<part>.ply, robot frame) this runs the same steps as
tograsp-socp's main_pickup.py / main_gcsm.py: bounding box -> antipodal contacts on two box faces -> SOCP task-dependent
grasp metric for the task screw -> ideal grasping region -> flange end-effector poses (grasp at G_DELTA, pre-grasp at
G_DELTA_INTER, config.py). Poses are mapped from the part's box frame back to the robot frame.

The task screw comes from the SALSA plan (--plan, the object's screw) or defaults to tograsp's pickup: pure translation
along +z. Translation -> force metric (main_pickup), rotation / general screw -> moment metric (main_gcsm).

Outputs in the object folder:
    socp_parts.png    4 views: all parts' points, every part's bounding box, all grasp poses, the best one highlighted
    socp_parts.html   the same in an interactive 3D page (plotly, if installed): rotate / zoom / toggle parts
    socp_parts.json   per part: box, dims, metric range, poses (flange, robot frame); the best pose
    socp_best.csv     pre-grasp + grasp of the best pose as 4x4 matrices for the CSV pose sender
    socp_poses.csv    every candidate (best first): pre-grasp + grasp lines, each with a note (part, metric, approach, fingertips)

usage:
    python3 socp_parts.py ~/bitsi_outputs/<request>_<id>_<label> [--plan ~/bitsi_outputs/plan.txt] [--base-yaw 90]
    python3 socp_parts.py ~/bitsi_outputs --latest            # newest object folder
"""
import argparse
import glob
import io
import json
import os
import sys
from contextlib import redirect_stdout
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("folder", help="object folder with one .ply per part (or BITSI_OUT with --latest)")
ap.add_argument("--latest", action="store_true", help="use the newest object folder inside FOLDER")
ap.add_argument("--plan", default="", help="SALSA plan (/salsa/plan JSON or saved echo) for the task screw")
ap.add_argument("--tograsp-dir", default="~/tograsp-socp", help="clone of github.com/apat20/tograsp-socp")
ap.add_argument("--threshold", type=float, default=None,
                help="grasping-region threshold on the normalized metric (default: 0.7 as main_pickup for a translation; 0 for a"
                     " swing / turn / fasten, where every feasible contact is kept and the pose nearest the screw axis is chosen)")
ap.add_argument("--min-points", type=int, default=100, help="skip parts with fewer points")
ap.add_argument("--base-yaw", type=float, default=0.0, help="degrees about z, robot frame -> server base (CSV only)")
ap.add_argument("--table-z", type=float, default=None, help="table height in the robot frame (default: lowest point of the object)")
ap.add_argument("--clearance", type=float, default=0.005, help="poses whose gripper comes closer than this to the table are dropped")
ap.add_argument("--no-html", action="store_true")
args = ap.parse_args()

TG_DIR = os.path.expanduser(args.tograsp_dir)
sys.path.insert(0, TG_DIR)
import open3d as o3d
import config as TC
import func.quaternion_lib as ql
import socp_module.two_finger_gfop as TGF
from point_cloud_module.process_point_cloud import PointCloud
from point_cloud_module.tograsp_socp import TOGrasp
from socp_module.two_finger_gfop import TwoFingerGFOP
import main_pickup as MP
import main_gcsm as MG
F_EXT0 = np.array(TC.F_EXTERNAL, float).reshape(6, 1)   # config.py's weight; restored after every part
# same matrix as tograsp's get_skew_symmetric; the original float() on 1-element arrays fails under numpy >= 2
TGF.get_skew_symmetric = lambda p: np.array([[0, -p[2, 0], p[1, 0]], [p[2, 0], 0, -p[0, 0]], [-p[1, 0], p[0, 0], 0]], float)

folder = os.path.expanduser(args.folder)
if args.latest:
    subs = [d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d) and glob.glob(os.path.join(d, "*.ply"))]
    if not subs:
        sys.exit(f"no object folder with .ply parts in {folder}")
    folder = max(subs, key=os.path.getmtime)
plys = sorted(glob.glob(os.path.join(folder, "*.ply")))
if not plys:
    sys.exit(f"no .ply parts in {folder} (run the node with BITSI_OUT set)")
print(f"object folder {folder}: {len(plys)} parts")


# ------------------------------------------------------------------ task screw (robot frame)
SCREW_TASK = "lift"
HAND_PIVOT = ("swing", "turn", "fasten", "twist")   # tasks whose screw axis passes through the hand (swing: the centre-of-percussion grip)


def load_screw():
    if not args.plan:
        return np.array([0, 0, 1.0]), np.zeros(3), True, "pickup (tograsp default): translate along +z"
    text = open(os.path.expanduser(args.plan)).read().strip()
    if not text.startswith("{"):
        import yaml
        text = yaml.safe_load(text.split("\n---")[0])["data"]
    plan = json.loads(text)
    label = os.path.basename(folder).split("_", 2)[-1].replace("_", " ")
    objs = [o for o in plan.get("objects", []) if o.get("screw")]
    obj = next((o for o in objs if str(o.get("label")) == label), objs[0] if objs else None)
    if obj is None:
        sys.exit("the plan has no task screw")
    s = obj["screw"]
    trans = bool(s.get("translation")) or s.get("pitch") is None
    global SCREW_TASK
    SCREW_TASK = s.get("task", "")
    return np.asarray(s["axis"], float), np.asarray(s["point"], float), trans, f"SALSA '{plan.get('command')}'"


u_base, q_base, TRANS, screw_src = load_screw()
COM = np.vstack([np.asarray(o3d.io.read_point_cloud(p).points) for p in plys]).mean(0)   # object centroid ~ centre of mass
u_base /= np.linalg.norm(u_base)
ON_AXIS = SCREW_TASK in HAND_PIVOT and not TRANS
THRESHOLD = args.threshold if args.threshold is not None else (0.0 if ON_AXIS else 0.7)
POSE_METRIC = {}   # id(grasp matrix) -> SOCP metric of the sampled contact pair nearest its fingertips
print(f"task screw: {screw_src}; axis {np.round(u_base, 3).tolist()}"
      + ("" if TRANS else f" through {np.round(q_base, 3).tolist()}") + f"; metric = {'force' if TRANS else 'moment'}")


# ------------------------------------------------------------------ per part: tograsp-socp
# gripper volume in the flange frame (z = approach, y = closing), Kinova Gen3 + Robotiq 2F-85, open: two fingers from the
# palm to the pinch point, and the gripper body from the flange to the palm. Corners only (boxes, convex).
_F = [(sx * 0.011, sy * 0.0425, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.095, 0.1434)]
_B = [(sx * 0.04, sy * 0.06, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, 0.095)]
GRIPPER_CORNERS = np.array(_F + _B + [(0, 0, 0)], float)


def above_table(T, table_z):
    return (T[:3, :3] @ GRIPPER_CORNERS.T + T[:3, 3:4])[2].min() >= table_z + args.clearance


def run_part(path):
    pcd = o3d.io.read_point_cloud(path)
    name = os.path.splitext(os.path.basename(path))[0]
    r = dict(part=name, n_points=len(pcd.points))
    if len(pcd.points) < args.min_points:
        r["skip"] = f"only {len(pcd.points)} points"; return r, None
    log = io.StringIO()
    try:
        with redirect_stdout(log):
            cloud = MP.build_cloud_object(PointCloud(), pcd)
            cloud.compute_bounding_box()
            g = TOGrasp(cloud)
            g.GRIPPER_WIDTH_TOLERANCE, g.GRIPPER_HEIGHT_TOLERANCE = TC.GRIPPER_WIDTH_TOLERANCE, TC.GRIPPER_HEIGHT_TOLERANCE
            g.G_DELTA, g.G_DELTA_INTER = TC.G_DELTA, TC.G_DELTA_INTER
            g.grasp_metric_threshold = THRESHOLD
            g.unit_vector_base = np.r_[u_base, 0.0]
            g.point_base = np.r_[q_base, 1.0]
            Gbb = cloud.g_bounding_box
            r.update(box_vertices=np.asarray(cloud.aligned_bounding_box_vertices).tolist(),
                     dims=[float(cloud.x_dim), float(cloud.y_dim), float(cloud.z_dim)])
            if not (cloud.y_dim < g.GRIPPER_WIDTH_TOLERANCE or cloud.x_dim < g.GRIPPER_WIDTH_TOLERANCE):
                r["skip"] = (f"box {cloud.x_dim:.2f} x {cloud.y_dim:.2f} m: no side narrower than the gripper "
                             f"({g.GRIPPER_WIDTH_TOLERANCE} m)")
                return r, None
            g.generate_contacts()
            # the WHOLE object's weight (tograsp: 10 N along -z of the base) acting at the object's centroid, in the part's
            # box frame: tograsp alone puts it at the box centre, which makes every part look ideal when fed one by one
            W = Gbb[:3, :3].T @ F_EXT0[:3, 0]
            r_com = Gbb[:3, :3].T @ (COM - Gbb[:3, 3])
            TC.F_EXTERNAL = np.r_[W, np.cross(r_com, W)].reshape(6, 1)
            gf = TwoFingerGFOP()
            gf.unit_vector = np.reshape(g.unit_vector, [3, 1])     # screw axis in the box frame (as main_pickup)
            try:
                vals = (MP if TRANS else MG).build_and_solve_gfop(gf, g)
            finally:
                TC.F_EXTERNAL = F_EXT0.copy()
            raw = np.asarray(vals, float).reshape(-1, 1)
            ok = ~np.isnan(raw)
            if not ok.any():
                r["skip"] = "no SOCP instance solved"; return r, None
            lo, hi = raw[ok].min(), raw[ok].max()
            g.test_datapoints = g.x_data
            g.computed = np.zeros_like(raw)
            g.computed[ok] = (raw[ok] - lo) / (hi - lo) if hi > lo else 1.0
            g.metric_values = g.computed
            g.get_ideal_grasping_region()
            g.get_end_effector_poses()
        poses, n_table = [], 0
        for T, Ti in zip(g.computed_end_effector_poses, g.computed_end_effector_poses_inter):
            T, Ti = np.array(T, float), np.array(Ti, float)
            T[3] = Ti[3] = [0, 0, 0, 1]    # tograsp sets [3,3]=1 on the grasp pose twice, never on the pre-grasp
            T, Ti = Gbb @ T, Gbb @ Ti
            if above_table(T, TABLE_Z) and above_table(Ti, TABLE_Z):
                poses.append((T, Ti))
            else:
                n_table += 1
        r.update(metric_min=float(lo), metric_max=float(hi), n_contacts=int(raw.size), n_feasible=int(ok.sum()),
                 n_poses=len(poses), n_dropped_table=n_table, metric_unit="N" if TRANS else "N m")
        if hi <= 0:
            r["skip"] = f"metric {hi:.2f} <= 0: the grasp cannot hold the object's weight"; return r, None
        if not poses:
            r["skip"] = ("all {} poses would hit the table".format(n_table) if n_table else
                         "SOCP solved but no end-effector pose passed the approach checks")
        if poses:   # each pose's own metric: the solved contact pair whose midpoint is nearest the pose's fingertips
            mid = 0.5 * (np.asarray(g.sampled_c1, float) + np.asarray(g.sampled_c2, float))
            mid = mid @ Gbb[:3, :3].T + Gbb[:3, 3]
            for T, _ in poses:
                d = np.linalg.norm(mid - (T[:3, 3] + TC.G_DELTA * T[:3, 2]), axis=1)
                d[~ok[:, 0]] = np.inf
                POSE_METRIC[id(T)] = float(raw[np.argmin(d), 0])
        return r, poses
    except Exception as e:
        r["skip"] = f"{type(e).__name__}: {e}"
        return r, None


TABLE_Z = args.table_z if args.table_z is not None else min(np.asarray(o3d.io.read_point_cloud(p).points)[:, 2].min() for p in plys)
results, all_poses = [], []
for p in plys:
    r, poses = run_part(p)
    results.append(r)
    if poses:
        r["poses"] = [dict(grasp=T.tolist(), pre_grasp=Ti.tolist(), metric=POSE_METRIC.get(id(T))) for T, Ti in poses]
        all_poses += [(r["part"], r["metric_max"], T, Ti) for T, Ti in poses]
    print(f"  {r['part']:<12s} {r['n_points']:6d} pts  " + (f"skipped: {r['skip']}" if "skip" in r and not poses else
          f"box {r['dims'][0]:.2f}x{r['dims'][1]:.2f}x{r['dims'][2]:.2f} m, metric {r['metric_min']:.2f}..{r['metric_max']:.2f} "
          f"{r['metric_unit']}, {r['n_feasible']}/{r['n_contacts']} SOCPs feasible, {r['n_poses']} poses"))

best = None
if all_poses:
    # best part = highest SOCP metric (ties within 0.1 broken by the part's worst metric, then its size). Within a part:
    # for a swing / turn / fasten (the hand is the pivot), the pose whose fingertips are nearest the screw axis (tograsp's moment metric is a pure couple, so it
    # does not see where the axis is; the hand should be the pivot, e.g. a swing pivot at the centre-of-percussion grip);
    # for a translation, tograsp's first pose.
    npts = {r["part"]: r["n_points"] for r in results}
    lo = {r["part"]: r.get("metric_min", 0) for r in results}

    def axis_dist(Tg):
        v = Tg[:3, 3] + TC.G_DELTA * Tg[:3, 2] - q_base
        return float(np.linalg.norm(v - (v @ u_base) * u_base)) if ON_AXIS else 0.0
    part, m, T, Ti = max(all_poses, key=lambda a: (round(a[1], 1), round(lo[a[0]], 1), npts[a[0]], -axis_dist(a[2])))
    best = dict(part=part, metric=m, pose_metric=POSE_METRIC.get(id(T)), grasp=T.tolist(), pre_grasp=Ti.tolist())
    print(f"best: part {part} (metric {m:.2f}); grasp flange at {np.round(T[:3, 3], 3).tolist()}")
    c, s = np.cos(np.radians(args.base_yaw)), np.sin(np.radians(args.base_yaw))
    Rb = np.eye(4); Rb[:2, :2] = [[c, -s], [s, c]]
    with open(os.path.join(folder, "socp_best.csv"), "w") as f:
        f.write(f"# tograsp-socp best pose: part {part}, metric {m:.3f}, flange poses, base yaw {args.base_yaw} deg\n")
        for name, M in (("pre_grasp", Ti), ("grasp", T)):
            f.write(",".join(f"{v:.6f}" for v in (Rb @ M).ravel()) + f"  # {name}\n")
    print(f"wrote {os.path.join(folder, 'socp_best.csv')}  (send with --gripper \"0:open,1:close\")")

    def approach_name(z):
        return ("top-down" if z[2] < -0.87 else "from below" if z[2] > 0.87 else
                "side, " + ["+x", "-x", "+y", "-y"][int(np.argmax([z[0], -z[0], z[1], -z[1]]))] + " direction" if abs(z[2]) < 0.5
                else f"tilted {np.degrees(np.arccos(-z[2])):.0f} deg from vertical")
    order = sorted(range(len(all_poses)), key=lambda i: (all_poses[i][2] is not T, -round(all_poses[i][1], 1), axis_dist(all_poses[i][2])))   # best first, then part metric, then axis distance
    unit = "N" if TRANS else "N m"
    with open(os.path.join(folder, "socp_poses.csv"), "w") as f:
        f.write(f"# tograsp-socp grasp poses for {os.path.basename(folder)}\n"
                f"# task screw ({screw_src}): {'translation' if TRANS else 'rotation'} axis {np.round(u_base, 3).tolist()}"
                + ("" if TRANS else f" through {np.round(q_base, 3).tolist()}") + f"; metric = {'force along' if TRANS else 'moment about'} the screw ({unit})\n"
                f"# parts from BITSI, screw from SALSA, grasps from tograsp-socp. Each line = one 4x4 pose, row-major, robot base frame"
                f" rotated by base yaw {args.base_yaw} deg. Pose = Kinova tool FLANGE (fingertips are {TC.G_DELTA} m further along z);"
                f" columns: x, y = finger closing direction, z = approach. Pre-grasp = {TC.G_DELTA_INTER} m flange stand-off, same orientation.\n"
                f"# Send in pairs: pre_grasp (gripper open), then grasp (close). Rows are ordered best first, then by part metric.\n")
        for n, i in enumerate(order):
            part, m, Tg, Tp = all_poses[i]
            pin = Tg[:3, 3] + TC.G_DELTA * Tg[:3, 2]
            note = (f"{'BEST ' if Tg is T else ''}{part} candidate {n}: pose metric {POSE_METRIC.get(id(Tg), float('nan')):.2f} {unit} (part max {m:.2f}); approach {approach_name(Tg[:3, 2])};"
                    f" fingertips at ({pin[0]:.3f}, {pin[1]:.3f}, {pin[2]:.3f}) m" + (f", {axis_dist(Tg) * 100:.1f} cm from the screw axis" if ON_AXIS else "") + f", closing along ({Tg[0, 1]:.2f}, {Tg[1, 1]:.2f}, {Tg[2, 1]:.2f})")
            f.write(",".join(f"{v:.6f}" for v in (Rb @ Tp).ravel()) + f"  # pre_grasp | {note}\n")
            f.write(",".join(f"{v:.6f}" for v in (Rb @ Tg).ravel()) + f"  # grasp | {note}\n")
    print(f"wrote {os.path.join(folder, 'socp_poses.csv')}  ({len(order)} grasp candidates, pre-grasp + grasp each, with notes)")
else:
    print("no part produced a grasp pose")
json.dump(dict(folder=folder, screw=dict(axis=u_base.tolist(), point=q_base.tolist(), translation=TRANS, source=screw_src),
               parts=results, best=best), open(os.path.join(folder, "socp_parts.json"), "w"), indent=1)


# ------------------------------------------------------------------ plots
EDGES = [(0, 1), (0, 2), (0, 3), (1, 6), (1, 7), (2, 5), (2, 7), (3, 5), (3, 6), (4, 5), (4, 6), (4, 7)]  # o3d box points
COLS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
clouds = [np.asarray(o3d.io.read_point_cloud(p).points) for p in plys]
pts_all = np.vstack(clouds)
ctr, rad = pts_all.mean(0), max(np.ptp(pts_all, 0).max() * 0.75, 0.08) + 0.12

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16, 13))
views = [(25, -60, "3D"), (90, -90, "top (x-y)"), (0, -90, "front (x-z)"), (0, 0, "side (y-z)")]
for k, (el, az, title) in enumerate(views):
    ax = fig.add_subplot(2, 2, k + 1, projection="3d")
    for i, (P, r) in enumerate(zip(clouds, results)):
        col = COLS[i % len(COLS)]
        ax.scatter(*P[::max(1, len(P) // 1500)].T, s=1.5, color=col, alpha=0.6,
                   label=f"{r['part']}" + ("" if r.get("poses") else " (no grasp)") if k == 0 else None)
        if r.get("box_vertices"):
            V = np.asarray(r["box_vertices"])
            for a, b in EDGES:
                ax.plot(*np.c_[V[a], V[b]], color=col, lw=1.2, alpha=0.9)
    L = 0.03
    for part, m, T, Ti in all_poses:
        for M, lw in ((T, 1.6), (Ti, 0.8)):
            for j, c in enumerate("rgb"):
                ax.quiver(*M[:3, 3], *(L * M[:3, j]), color=c, lw=lw, arrow_length_ratio=0.2)
        ax.plot(*np.c_[Ti[:3, 3], T[:3, 3]], color="k", lw=0.6, ls=":")
    if best:
        T = np.asarray(best["grasp"])
        ax.scatter(*T[:3, 3], s=120, marker="*", color="k", label="best grasp (flange)" if k == 0 else None)
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[1] - rad, ctr[1] + rad); ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(el, az); ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if k == 0:
        ax.legend(loc="upper left", fontsize=8, markerscale=4)
fig.suptitle(f"tograsp-socp per BITSI part - {os.path.basename(folder)}\n{screw_src}; boxes = each part's bounding box; "
             f"frames: x red, y green (closing), z blue (approach); bold = grasp, thin = pre-grasp", fontsize=11)
fig.tight_layout()
png = os.path.join(folder, "socp_parts.png")
fig.savefig(png, dpi=110)
print(f"wrote {png}")

if not args.no_html:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed (pip3 install --user plotly) - skipping socp_parts.html")
    else:
        tr = []
        for i, (P, r) in enumerate(zip(clouds, results)):
            col = COLS[i % len(COLS)]
            Q = P[::max(1, len(P) // 4000)]
            tr.append(go.Scatter3d(x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], mode="markers", marker=dict(size=2, color=col),
                                   name=f"{r['part']} ({r['n_points']} pts)", legendgroup=r["part"]))
            if r.get("box_vertices"):
                V = np.asarray(r["box_vertices"]); xs, ys, zs = [], [], []
                for a, b in EDGES:
                    xs += [V[a, 0], V[b, 0], None]; ys += [V[a, 1], V[b, 1], None]; zs += [V[a, 2], V[b, 2], None]
                tr.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=col, width=4),
                                       name=f"{r['part']} box", legendgroup=r["part"], showlegend=False))
        for part, m, T, Ti in all_poses:
            for M, w in ((T, 6), (Ti, 2)):
                for j, c in enumerate(("red", "green", "blue")):
                    e = M[:3, 3] + 0.03 * M[:3, j]
                    tr.append(go.Scatter3d(x=[M[0, 3], e[0]], y=[M[1, 3], e[1]], z=[M[2, 3], e[2]], mode="lines",
                                           line=dict(color=c, width=w), legendgroup=part, showlegend=False,
                                           hovertext=f"{part} metric {m:.2f}"))
        if best:
            T = np.asarray(best["grasp"])
            tr.append(go.Scatter3d(x=[T[0, 3]], y=[T[1, 3]], z=[T[2, 3]], mode="markers", name="best grasp",
                                   marker=dict(size=7, color="black", symbol="diamond")))
        f = go.Figure(tr)
        f.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                        title=f"tograsp-socp per BITSI part: {os.path.basename(folder)} ({screw_src})",
                        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"))
        html = os.path.join(folder, "socp_parts.html")
        f.write_html(html, include_plotlyjs=True)
        print(f"wrote {html}  (open in a browser; click a part in the legend to hide it)")
