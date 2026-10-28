"""Send a SALSA plan (/salsa/plan) to the motion-planning server over ZMQ (same protocol as the CSV pose sender).

The plan comes from a file (the raw JSON, or the output of `ros2 topic echo /salsa/plan --once --full-length`) or,
with --ros, straight from the topic. One candidate of one object is sent as a waypoint sequence; gripper actions come
from SALSA's per-pose `gripper` field:
    pre_grasp (open) -> grasp (CLOSE on arrival) -> lift_k / screw_k (hold) -> release (OPEN on arrival) -> retreat
    press tasks: fingers closed for the whole sequence (CLOSE at the first waypoint).

SALSA poses are at the pinch point (midway between the finger contacts), rotation columns x, y = closing direction,
z = approach. --tcp-offset moves each pose back along its own z to the TCP the server expects.

usage:
  python3 send_salsa.py plan.txt --dry-run                     # print the waypoints, send nothing
  python3 send_salsa.py plan.txt --tcp-offset 0.13             # send candidate 0 of the first planned object
  python3 send_salsa.py --ros --object drill --candidate 1     # read /salsa/plan live (inside a sourced ROS shell)
  python3 send_salsa.py plan.txt --only 0                      # just waypoint 0 (pre_grasp)
  python3 send_salsa.py plan.txt --step                        # one waypoint at a time, Enter before each
  python3 send_salsa.py plan.txt --tograsp --csv poses.csv     # flange poses as tograsp-socp computes them
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("plan", nargs="?", help="plan file: JSON, or saved `ros2 topic echo /salsa/plan` output")
ap.add_argument("--ros", action="store_true", help="read the latest message on /salsa/plan instead of a file")
ap.add_argument("--object", default="", help="object id or label substring (default: first object with candidates)")
ap.add_argument("--candidate", type=int, default=0, help="candidate rank to send (0 = best by SALSA's heuristic)")
ap.add_argument("--tcp-offset", type=float, default=0.0,
                help="metres from the pinch point back to the server's TCP, along the approach axis")
ap.add_argument("--pregrasp-offset", type=float, default=None,
                help="put pre_grasp this far (m) back from the grasp pinch point along the approach (the plan uses 0.10)")
ap.add_argument("--tograsp", action="store_true",
                help="tograsp-socp convention (config.py): flange poses, grasp at G_DELTA 0.1434 m and pre-grasp at "
                     "G_DELTA_INTER 0.18 m behind the grasp centre = --tcp-offset 0.1434 --pregrasp-offset 0.0366")
ap.add_argument("--base-yaw", type=float, default=0.0,
                help="degrees about the vertical (z) from the plan frame to the server's base frame; rotates positions "
                     "AND orientations (use when the robot base is turned relative to base_link)")
ap.add_argument("--tool-yaw", type=float, default=0.0,
                help="degrees about the approach axis applied to the gripper orientation only (use when the server's tool "
                     "frame is turned relative to SALSA's: x, y=closing, z=approach)")
ap.add_argument("--socp", action="store_true",
                help="re-rank the candidates with the tograsp-socp grasp metric for the plan's task screw; --candidate "
                     "then indexes the SOCP order (0 = best)")
ap.add_argument("--tograsp-dir", default="~/tograsp-socp", help="clone of github.com/apat20/tograsp-socp (for --socp)")
ap.add_argument("--weight", type=float, default=10.0, help="object weight in N for --socp (tograsp config: 10 N)")
ap.add_argument("--frame", default="base", help="frame name sent to the server (the plan is in its own frame_id)")
ap.add_argument("--addr", default="tcp://localhost:5555")
ap.add_argument("--seg-time", type=float, default=0.0,
                help="seconds per segment; 0 lets the server choose from its speed limits")
ap.add_argument("--dwell", type=float, default=0.5, help="seconds to hold after a gripper action")
ap.add_argument("--min-z", type=float, default=-0.05, help="refuse waypoints below this height (m, plan frame)")
ap.add_argument("--max-reach", type=float, default=0.90, help="refuse waypoints farther than this from the origin (m)")
ap.add_argument("--proto-dir", default="", help="directory holding common_pb2.py / commands_pb2.py")
ap.add_argument("--dry-run", action="store_true", help="print the waypoints, send nothing")
ap.add_argument("--only", type=int, default=None, help="send just this one waypoint index (see --dry-run for indices)")
ap.add_argument("--step", action="store_true",
                help="send the waypoints one at a time; press Enter before each (after the robot has stopped)")
ap.add_argument("--csv", default="", help="write the poses as 4x4 matrices for the CSV pose sender instead of sending")
ap.add_argument("--yes", action="store_true", help="do not ask for confirmation before sending")
args = ap.parse_args()
if not args.plan and not args.ros:
    ap.error("give a plan file or --ros")
if args.tograsp:     # tograsp-socp/config.py: G_DELTA, G_DELTA_INTER (grasp centre -> flange, along the approach z)
    args.tcp_offset, args.pregrasp_offset = 0.1434, 0.1800 - 0.1434


def socp_rank(obj, cands):
    """Re-rank SALSA's candidates with the tograsp-socp task-dependent grasp metric (socp_module.two_finger_gfop,
    parameters from its config.py): the largest force along the task screw (pure translation, e.g. lift) or moment
    about it (rotation / general screw) that the two soft-finger contacts can apply while holding the object's weight.
    Wrenches are taken at the object's centroid (translation) or at the screw axis point (rotation)."""
    sys.path.insert(0, os.path.expanduser(args.tograsp_dir))
    import config as TC
    import socp_module.two_finger_gfop as TG
    from socp_module.two_finger_gfop import TwoFingerGFOP
    p_hat = lambda p: np.array([[0, -p[2, 0], p[1, 0]], [p[2, 0], 0, -p[0, 0]], [-p[1, 0], p[0, 0], 0]], float)
    TG.get_skew_symmetric = p_hat     # same matrix; the original float() on 1-element arrays fails under numpy >= 2
    sc = obj.get("screw") or {}
    if not sc.get("axis"):
        sys.exit("--socp: the plan has no task screw for this object")
    u = np.asarray(sc["axis"], float); u /= np.linalg.norm(u)
    trans = bool(sc.get("translation")) or sc.get("pitch") is None
    parts = [p for p in obj.get("parts", []) if p.get("centroid") is not None]
    com = (np.average([p["centroid"] for p in parts], axis=0, weights=[p.get("n_points", 1) for p in parts])
           if parts else np.mean([np.mean(c["contacts"], 0) for c in cands if c.get("contacts")], 0))
    origin = com if trans else np.asarray(sc["point"], float)
    W = np.array([0, 0, -args.weight])                         # weight along -z of the plan frame (base_link z up)
    F_ext = np.r_[W, np.cross(com - origin, W)].reshape(6, 1)

    def frame(n):                                              # contact frame: z = inward normal
        x = np.cross(n, [1, 0, 0] if abs(n[0]) < .9 else [0, 1, 0]); x /= np.linalg.norm(x)
        return np.stack([x, np.cross(n, x), n], 1)

    scored = []
    for c in cands:
        if not c.get("contacts"):
            scored.append((np.nan, c)); continue
        c1, c2 = (np.asarray(p, float) for p in c["contacts"])
        n1 = (c2 - c1) / np.linalg.norm(c2 - c1)
        g = TwoFingerGFOP()
        g.R_OC_1, g.R_OC_2 = frame(n1), frame(-n1)
        g.p_OC_1, g.p_OC_2 = (c1 - origin).reshape(3, 1), (c2 - origin).reshape(3, 1)
        g.unit_vector = u.reshape(3, 1)
        g.F, g.contact_model, g.F_external = TC.F, TC.CONTACT_MODEL, F_ext
        g.e11, g.e12, g.e1r, g.mu1 = TC.E11, TC.E12, TC.E1R, TC.MU1
        g.e21, g.e22, g.e2r, g.mu2 = TC.E21, TC.E22, TC.E2R, TC.MU2
        g.sigma, g.mu = TC.SIGMA, TC.MU
        try:
            m = g.compute_metric_force() if trans else g.compute_metric_moment()
        except Exception as e:
            print(f"  SOCP error on candidate {c.get('rank')}: {e}"); m = np.nan
        scored.append((float(m) if m is not None else np.nan, c))
    scored.sort(key=lambda s: -np.inf if np.isnan(s[0]) else -s[0])
    unit = "N along the screw" if trans else "N m about the screw"
    print(f"SOCP ranking ({'force' if trans else 'moment'} metric, weight {args.weight} N, {len(cands)} candidates):")
    for k, (m, c) in enumerate(scored):
        print(f"  new {k}: SALSA candidate {c.get('rank')}  metric {'infeasible' if np.isnan(m) else f'{m:.2f} {unit}'}")
    if all(np.isnan(m) for m, _ in scored):
        sys.exit("--socp: no candidate has a feasible SOCP (check --weight, or the contacts)")
    return [c for _, c in scored]


def load_plan():
    if args.ros:
        import rclpy
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
        from std_msgs.msg import String
        rclpy.init()
        node = rclpy.create_node("salsa_sender")
        got = []
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, "/salsa/plan", lambda m: got.append(m.data), qos)
        t0 = time.time()
        while not got and time.time() - t0 < 10:
            rclpy.spin_once(node, timeout_sec=0.2)
        node.destroy_node(); rclpy.shutdown()
        if not got:
            sys.exit("no message on /salsa/plan within 10 s (is the node running with a SALSA command?)")
        return json.loads(got[-1])
    text = open(args.plan).read().strip()
    if text.startswith("{"):
        return json.loads(text)
    import yaml                                   # `ros2 topic echo` output: data: '{...}'  (then '---')
    doc = yaml.safe_load(text.split("\n---")[0])
    return json.loads(doc["data"])


plan = load_plan()
objs = [o for o in plan.get("objects", []) if o.get("candidates")]
if args.object:
    objs = [o for o in objs if args.object.lower() in str(o.get("label", "")).lower() or args.object == str(o.get("id"))]
if not objs:
    errs = "; ".join(f"{o.get('label')}: {o.get('error')}" for o in plan.get("objects", []) if o.get("error"))
    sys.exit("no object with SALSA candidates in the plan" + (f" ({errs})" if errs else ""))
obj = objs[0]
cands = obj["candidates"]
if args.socp:
    cands = socp_rank(obj, cands)
if not 0 <= args.candidate < len(cands):
    sys.exit(f"--candidate {args.candidate}: the plan has {len(cands)} candidate(s) for {obj['label']}")
seq = cands[args.candidate]["poses"]
print(f"plan request {plan.get('request_id')} frame {plan.get('frame_id')} command '{plan.get('command')}'")
print(f"object {obj.get('id')} ({obj.get('label')}), candidate {args.candidate} of {len(cands)}, {len(seq)} poses, "
      f"tcp offset {args.tcp_offset:.3f} m, base yaw {args.base_yaw:g} deg, tool yaw {args.tool_yaw:g} deg")

# gripper action on arrival: change of SALSA's gripper state between consecutive poses
GRIP = {"open": "open", "close": "close", "hold": "close", "closed": "close"}
state, acts = None, {}                            # None: the first waypoint always sets the gripper (open before a grasp)
for i, p in enumerate(seq):
    want = GRIP[p["gripper"]]
    if want != state:
        acts[i] = want
    state = want

grasp = next((p for p in seq if p["name"] == "grasp"), None)
poses = []
for i, p in enumerate(seq):
    R = Rotation.from_quat(p["quaternion_xyzw"]).as_matrix()      # scipy order is (x, y, z, w)
    t = np.asarray(p["position"], float)
    if p["name"] == "pre_grasp" and args.pregrasp_offset is not None and grasp is not None:
        t = np.asarray(grasp["position"], float) - R[:, 2] * args.pregrasp_offset
    t = t - R[:, 2] * args.tcp_offset
    R = R @ Rotation.from_euler("z", args.tool_yaw, degrees=True).as_matrix()     # tool frame: about the approach
    Rb = Rotation.from_euler("z", args.base_yaw, degrees=True).as_matrix()         # plan frame -> server base frame
    t, R = Rb @ t, Rb @ R
    if t[2] < args.min_z or np.linalg.norm(t) > args.max_reach:
        sys.exit(f"pose {i} ({p['name']}) at ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) is outside "
                 f"min_z {args.min_z} / reach {args.max_reach} m; not sending")
    poses.append((p["name"], t, R))
    print(f"  [{i:2d}] {p['name']:<10s} pos=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})  gripper={acts.get(i, '-')}")

if args.csv:   # for the CSV pose sender: one 4x4 matrix per line, 16 numbers row-major, '#' comments allowed
    with open(args.csv, "w") as f:
        f.write(f"# SALSA '{plan.get('command')}' object {obj.get('id')} ({obj.get('label')}) candidate {args.candidate}, "
                f"frame {plan.get('frame_id')}, tcp offset {args.tcp_offset} m, base yaw {args.base_yaw}, tool yaw {args.tool_yaw} deg\n")
        for name, t, R in poses:
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
            f.write(",".join(f"{v:.6f}" for v in T.ravel()) + f"  # {name}\n")
    g = ",".join(f"{i}:{a}" for i, a in sorted(acts.items()))
    print(f"wrote {args.csv}; send with:  python3 <csv sender> {args.csv} --gripper \"{g}\" --dry-run")
if args.dry_run or args.csv:
    print(f"dry run: {len(poses)} waypoints, nothing sent")
    sys.exit(0)
if plan.get("frame_id") and plan["frame_id"] not in (args.frame, f"{args.frame}_link"):
    print(f"WARNING: plan frame '{plan['frame_id']}' vs server frame '{args.frame}'")
if args.only is not None and not 0 <= args.only < len(poses):
    sys.exit(f"--only {args.only}: the plan has waypoints 0..{len(poses) - 1}")
# batches of waypoint indices, each sent as its own sequence
batches = [[args.only]] if args.only is not None else [[i] for i in range(len(poses))] if args.step else [list(range(len(poses)))]
if not args.step and not args.yes:
    what = f"waypoint {args.only} ({poses[args.only][0]})" if args.only is not None else f"{len(poses)} waypoints"
    if input(f"send {what} to {args.addr}? [y/N] ").strip().lower() != "y":
        sys.exit("not sent")

if args.proto_dir:
    sys.path.insert(0, args.proto_dir)
import zmq
import common_pb2 as common
import commands_pb2 as cmds

ACTIONS = {"open": common.GRIPPER_OPEN, "close": common.GRIPPER_CLOSE}
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 5000)
sock.setsockopt(zmq.LINGER, 0)
sock.connect(args.addr)


def rpc(command, reply_cls):
    sock.send(command.SerializeToString())
    reply = reply_cls()
    reply.ParseFromString(sock.recv())
    return reply


def send(idx):
    """Send waypoints idx (plan indices) as one sequence; wait for the planning result. False if it failed."""
    cmd = cmds.Command()
    cmd.goals.sequence_id = int(time.time() * 1000) % 2**31
    cmd.goals.frame = args.frame
    for i in idx:
        name, t, R = poses[i]
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        w = cmd.goals.waypoints.add()
        w.pose.x, w.pose.y, w.pose.z = t
        w.pose.qx, w.pose.qy, w.pose.qz, w.pose.qw = qx, qy, qz, qw
        w.segment_duration = args.seg_time
        if i in acts:
            w.gripper = ACTIONS[acts[i]]
            w.dwell_time = args.dwell

    def describe(f):
        k = f.waypoint_index
        where = f"waypoint {idx[k]} ({poses[idx[k]][0]})" if 0 <= k < len(idx) else "sequence"
        return f"{common.FailureCode.Name(f.code)} at {where}: {f.message} ({f.waypoints_planned} planned OK)"

    reply = rpc(cmd, cmds.CommandReply)
    if not reply.accepted:
        print("rejected: " + describe(reply.failure)); return False
    print("accepted sequence", reply.sequence_id)
    deadline = time.time() + 60
    while time.time() < deadline:
        q = cmds.Command()
        q.get_result.sequence_id = reply.sequence_id
        res = rpc(q, cmds.ResultReply)
        if res.status == cmds.PLAN_PENDING:
            time.sleep(0.3)
            continue
        print("plan status:", cmds.PlanStatus.Name(res.status))
        if res.status == cmds.PLAN_FAILED:
            print(f"planning failed: {describe(res.failure)}"); return False
        print("executing"); return True
    print("planning did not finish within 60 s"); return False


try:
    for b in batches:
        if args.step:
            i = b[0]; name, t, _ = poses[i]
            a = input(f"[{i}] {name} pos=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) gripper={acts.get(i, '-')}: "
                      f"Enter = send, s = skip, q = quit (wait until the robot has stopped) ").strip().lower()
            if a == "q": sys.exit("stopped")
            if a == "s": continue
        if not send(b):
            sys.exit(f"stopped (try --candidate {args.candidate + 1})" if not args.step else "stopped")
except zmq.Again:
    sys.exit("no reply from server (is it running, and is the address right?)")
