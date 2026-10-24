#!/usr/bin/env python3
"""ROS 2 Humble node: run BITSI Chebyshev segmentation on every object cloud from the perception pipeline.

Subscribes (reliable, transient-local, so a late start still receives the retained result):
  /perception/clouds  seg_interfaces/msg/ObjectClouds   one robot-frame PointCloud2 per object
  /perception/result  seg_interfaces/msg/SegObjects     labels (used to name objects; optional)
Publishes:
  /bitsi/result       std_msgs/String  JSON {request_id, frame_id, stamp, objects:[{id, label, segments:[...]}]}
  /bitsi/segments     sensor_msgs/PointCloud2  all segments of the latest object, xyz + float32 'segment' field
The pipeline repeats results for late subscribers with the same request_id; each (request_id, object id) is processed once.
Message fields are found by introspection (any PointCloud2 inside ObjectClouds, with an id/label beside it), so the node does
not hard-code the seg_interfaces field names; the first message's layout is logged.
"""
import json, os, threading, time, traceback
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField, RegionOfInterest
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import String, Header
from geometry_msgs.msg import Pose, PoseArray
from seg_interfaces.msg import ObjectClouds, SegObjects

from bitsi_core import run_bitsi
from bitsi_web import WebView
from salsa_core import parse_command, salsa_plan

ID_KEYS = ('id', 'object_id', 'obj_id', 'index'); LABEL_KEYS = ('label', 'name', 'class_name', 'category'); REQ_KEYS = ('request_id', 'req_id')


def first_attr(m, keys):
    for k in keys:
        if hasattr(m, k): return getattr(m, k)
    return None


def find_clouds(msg):
    """[(id, label, PointCloud2)] from any message holding PointCloud2s directly or in a sequence of sub-messages."""
    out = []
    for f in msg.get_fields_and_field_types():
        v = getattr(msg, f)
        if isinstance(v, PointCloud2): out.append((first_attr(msg, ID_KEYS), first_attr(msg, LABEL_KEYS), v))
        elif isinstance(v, (list, tuple)):
            for i, e in enumerate(v):
                if isinstance(e, PointCloud2): out.append((i, None, e))
                elif hasattr(e, 'get_fields_and_field_types'):
                    for g in e.get_fields_and_field_types():
                        w = getattr(e, g)
                        if isinstance(w, PointCloud2):
                            oid = first_attr(e, ID_KEYS); out.append((i if oid is None else oid, first_attr(e, LABEL_KEYS), w))
    return out


def image_to_np(msg):
    """sensor_msgs/Image -> numpy (H, W[, C]) without cv_bridge; colour returned as RGB."""
    enc = msg.encoding.lower(); ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(enc, 1)
    dt = np.uint16 if enc in ('mono16', '16uc1') else np.uint8; isz = np.dtype(dt).itemsize
    a = np.frombuffer(bytes(msg.data), dtype=dt).reshape(msg.height, msg.step // isz)[:, :msg.width * ch]
    a = a.reshape(msg.height, msg.width, ch) if ch > 1 else a.reshape(msg.height, msg.width)
    if enc.startswith('bgr'): a = a[..., [2, 1, 0]]
    return a[..., :3] if ch == 4 else a


def box_of(e):
    """2D box (x0, y0, x1, y1) and mask of one SegObjects element: from a RegionOfInterest, a 4-number bbox/box field, or the
    mask image's nonzero extent. Field names are found by introspection; returns (None, None) if nothing usable."""
    box = mask = None
    for f in e.get_fields_and_field_types():
        v = getattr(e, f)
        if isinstance(v, RegionOfInterest) and v.width > 0:
            box = (v.x_offset, v.y_offset, v.x_offset + v.width, v.y_offset + v.height)
        elif isinstance(v, Image) and v.width > 0 and mask is None:
            try:
                m = image_to_np(v); mask = (m if m.ndim == 2 else m.max(2)) > 0
            except Exception: mask = None
        elif 'box' in f.lower() and hasattr(v, '__len__') and len(v) == 4:
            x0, y0, a2, b2 = [float(t) for t in v]
            box = (x0, y0, a2, b2) if a2 > x0 and b2 > y0 else (x0, y0, x0 + a2, y0 + b2)
    if box is None and mask is not None and mask.any():
        ys, xs = np.nonzero(mask); box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return box, mask


def strip_arrays(d):
    """SALSA plan without point arrays (JSON-safe)."""
    out = {k: v for k, v in d.items() if k != 'grasp_part_points'}
    out['parts'] = [{k: v for k, v in p.items() if k != 'points'} for p in d.get('parts', [])]
    return out


def parse_target(text):
    """'cup' / 'cup, red bottle' / 'cup;phone' -> ['cup', 'red bottle', 'phone'] (lower case); empty or 'all' -> [] (every object)."""
    terms = [t.strip().lower() for t in str(text or '').replace(';', ',').split(',') if t.strip()]
    return [] if terms in ([], ['all'], ['*']) else terms


def layout(msg, depth=0):
    s = []
    for f, t in msg.get_fields_and_field_types().items():
        s.append('  ' * depth + f'{f}: {t}'); v = getattr(msg, f)
        if isinstance(v, (list, tuple)) and v and hasattr(v[0], 'get_fields_and_field_types') and depth < 2: s += layout(v[0], depth + 1)
    return s


class BitsiNode(Node):
    def __init__(self):
        super().__init__('bitsi_chebyshev')
        P = lambda n, d: self.declare_parameter(n, d).value
        self.p = dict(gripper_width=P('gripper_width', .13), gripper_height=P('gripper_height', .07), epsilon=P('epsilon', 5e-3),
                      strength_threshold=P('strength_threshold', .01), ibr_tolerance=P('ibr_tolerance', .07), voxel=P('voxel', .004),
                      version=P('bitsi_version', os.environ.get('BITSI_VERSION', 'v0')), up=list(P('up', [0., 0., 1.])))   # up: gravity in the cloud frame (v4-v6)
        self.save_dir = P('save_dir', ''); clouds_topic = P('clouds_topic', '/perception/clouds'); result_topic = P('result_topic', '/perception/result')
        web_port = int(P('web_port', 8090)); web_dir = P('web_dir', '/tmp/bitsi_web')
        self.target = parse_target(P('target', os.environ.get('BITSI_TARGET', '')))   # '' = every object; 'cup' or 'cup, red bottle' = only matching labels
        input_type = P('input_type', 'object_clouds')   # 'object_clouds' (seg_interfaces/ObjectClouds) or 'pointcloud2' (any sensor_msgs/PointCloud2 topic)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
        self.boxes = {}; self.overlays = []; self.labels = {}; self.done = set(); self.lock = threading.Lock(); self.busy = False; self.logged = set(); self.last_clouds = None
        if input_type == 'pointcloud2':   # one object per message; QoS best-effort-compatible for sensor-style publishers
            self.seq = 0; sq = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
            self.create_subscription(PointCloud2, clouds_topic, self.on_pointcloud2, sq)
        else:
            self.create_subscription(SegObjects, result_topic, self.on_result, qos)
            self.create_subscription(ObjectClouds, clouds_topic, self.on_clouds, qos)
        self.create_subscription(String, '/bitsi/target', lambda m: self.set_target(m.data), 10)
        self.create_subscription(String, '/bitsi/version', lambda m: self.set_version(m.data), 10)
        # SALSA: '' = segmentation only; e.g. 'lift 10 cm', 'turn 90 deg', 'hinge' = task screw + end-effector pose sequence
        self.salsa = P('salsa_command', os.environ.get('SALSA_COMMAND', '')); self.salsa_k = int(P('salsa_candidates', 5))
        if self.salsa: parse_command(self.salsa)   # fail fast on an unknown task word
        self.create_subscription(String, '/salsa/command', lambda m: self.set_salsa(m.data), 10)
        self.pub_salsa = self.create_publisher(String, '/salsa/plan', qos)
        self.pub_poses = self.create_publisher(PoseArray, '/salsa/poses', qos)
        overlay_topic = P('overlay_topic', '/perception/overlay')   # annotated camera image of the capture, shown on the web page
        if overlay_topic:
            self.create_subscription(Image, overlay_topic, self.on_overlay, QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.pub_json = self.create_publisher(String, '/bitsi/result', qos)
        self.pub_cloud = self.create_publisher(PointCloud2, '/bitsi/segments', qos)
        self.web = None
        if web_port:
            try: self.web = WebView(web_dir, web_port, on_target=self.set_target, target=', '.join(self.target), on_version=self.set_version, version=self.p['version'], on_salsa=self.set_salsa, salsa=self.salsa); self.get_logger().info(f'web view: http://<this-machine-ip>:{web_port}/  (serving {web_dir})')
            except OSError as e: self.get_logger().error(f'web view disabled, port {web_port}: {e}')
        self.get_logger().info(f'input_type={input_type}; listening on {clouds_topic}' + ('' if input_type == 'pointcloud2' else f' and {result_topic}') + f'; target={", ".join(self.target) or "(all objects)"}; params {self.p}; save_dir={self.save_dir or "(off)"}')

    def log_layout(self, tag, msg):
        if tag not in self.logged:
            self.logged.add(tag); self.get_logger().info(f'{tag} layout:\n' + '\n'.join(layout(msg)))

    def on_result(self, msg):
        self.log_layout('SegObjects', msg); req = first_attr(msg, REQ_KEYS)
        for f in msg.get_fields_and_field_types():
            v = getattr(msg, f)
            if isinstance(v, (list, tuple)):
                for i, e in enumerate(v):
                    lab = first_attr(e, LABEL_KEYS) if hasattr(e, 'get_fields_and_field_types') else None
                    if lab is not None:
                        oid = first_attr(e, ID_KEYS); key = (str(req), str(i if oid is None else oid)); self.labels[key] = str(lab)
                        if key not in self.boxes:
                            try: self.boxes[key] = box_of(e)
                            except Exception: self.boxes[key] = (None, None)
        if self.target and self.last_clouds is not None: self.on_clouds(self.last_clouds)   # labels may arrive after the clouds

    def on_overlay(self, msg):
        if self.web is None: return
        st = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if self.overlays and abs(self.overlays[-1][0] - st) < 1e-6: return   # the pipeline repeats the last capture
        try: self.overlays = (self.overlays + [(st, image_to_np(msg))])[-4:]
        except Exception as e: self.get_logger().warn(f'overlay not decodable ({msg.encoding}): {e}')

    def overlay_for(self, stamp):
        if not self.overlays: return None
        return min(self.overlays, key=lambda o: abs(o[0] - stamp))[1]

    def set_salsa(self, text):
        try: spec = parse_command(text)
        except ValueError as e: self.get_logger().error(str(e)); return
        self.salsa = str(text or '').strip(); self.get_logger().info(f'SALSA command: {self.salsa or "(off)"} -> {spec}')
        if self.web is not None: self.web.set_salsa(self.salsa)
        if self.last_clouds is not None: self.on_clouds(self.last_clouds)   # plan on the retained request

    def set_version(self, text):
        from bitsi_core import VERSIONS
        v = str(text).strip().lower(); v = v if v.startswith('v') else 'v' + v
        if v not in VERSIONS: self.get_logger().error(f'unknown BITSI version {text!r}; choose from {VERSIONS}'); return
        self.p['version'] = v; self.get_logger().info(f'BITSI version set to {v}')
        if self.web is not None: self.web.set_version(v)
        if self.last_clouds is not None: self.on_clouds(self.last_clouds)   # re-run the retained request with the new version

    def set_target(self, text):
        self.target = parse_target(text)
        self.get_logger().info(f'target set to: {", ".join(self.target) or "(all objects)"}')
        if self.web is not None: self.web.set_target(', '.join(self.target))
        if self.last_clouds is not None: self.on_clouds(self.last_clouds)   # re-check the retained request at once

    def wanted(self, req, oid, lab):
        if not self.target: return True
        label = (lab or self.labels.get((req, oid), '')).lower()
        return any(t == oid.lower() or (label and t in label) for t in self.target)

    def on_clouds(self, msg):
        self.log_layout('ObjectClouds', msg); req = str(first_attr(msg, REQ_KEYS)); self.last_clouds = msg
        items = [(str(oid), lab, c) for oid, lab, c in find_clouds(msg) if (req, str(oid), self.p['version'], self.salsa) not in self.done and self.wanted(req, str(oid), lab)]
        if not items: return
        with self.lock:
            if self.busy: self.get_logger().info(f'busy; request {req} will be picked up from the next repeat'); return
            self.busy = True
        threading.Thread(target=self.process, args=(req, items), daemon=True).start()

    def on_pointcloud2(self, msg):
        with self.lock:
            if self.busy: return   # drop frames while BITSI runs on the previous one
            self.busy = True
        self.seq += 1; req = f'{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
        threading.Thread(target=self.process, args=(req, [('0', msg.header.frame_id, msg)]), daemon=True).start()

    def process(self, req, items):
        try:
            out = dict(request_id=req, objects=[]); seg_pts = []; seg_ids = []; hdr = None; web_objs = []; salsa_out = []
            for oid, lab, cloud in items:
                hdr = cloud.header; label = lab or self.labels.get((req, oid), '')
                pts = pc2.read_points_numpy(cloud, field_names=('x', 'y', 'z'), skip_nans=True).astype(np.float64)
                t0 = time.time(); rec = dict(id=oid, label=str(label), n_points=int(len(pts)))
                try:
                    r = run_bitsi(pts, **self.p); rec.update({k: v for k, v in r.items() if k != 'segments'})
                    rec['segments'] = [{k: v for k, v in s.items() if k != 'points'} for s in r['segments']]
                    for j, s in enumerate(r['segments']): seg_pts.append(s['points']); seg_ids.append(np.full(len(s['points']), len(seg_ids), np.float32))
                    rec['seg_points'] = [s['points'] for s in r['segments']]
                    rec['raw_points'] = pts
                    st = hdr.stamp.sec + 1e-9 * hdr.stamp.nanosec; rec['image'] = self.overlay_for(st)
                    rec['bbox2d'], rec['mask2d'] = self.boxes.get((req, oid), (None, None))
                    if self.save_dir: self.save(req, oid, label, r)
                except Exception as e:
                    rec['error'] = f'{type(e).__name__}: {e}'
                if self.salsa:
                    try:
                        sp = salsa_plan(pts, self.salsa, voxel=self.p['voxel'] or 0.004, n_candidates=self.salsa_k)
                        rec['salsa'] = sp; salsa_out.append(dict(id=oid, label=str(label), **strip_arrays(sp)))
                    except Exception as e:
                        rec['salsa'] = dict(command=self.salsa, error=f'{type(e).__name__}: {e}', parts=[], candidates=[])
                        salsa_out.append(dict(id=oid, label=str(label), command=self.salsa, error=rec['salsa']['error']))
                rec['seconds'] = round(time.time() - t0, 3); web_objs.append(rec); out['objects'].append({k: v for k, v in rec.items() if k not in ('seg_points', 'raw_points', 'image', 'mask2d', 'salsa')}
                                       | ({'bbox2d': [float(t) for t in rec['bbox2d']]} if rec.get('bbox2d') is not None else {}))
                self.done.add((req, oid, self.p['version'], self.salsa))
                self.get_logger().info(f'request {req} object {oid} ({label}): {len(pts)} pts -> '
                                       f'{len(rec.get("segments", []))} segments in {rec["seconds"]}s' + (f' ERROR {rec["error"]}' if 'error' in rec else ''))
            if hdr is not None:
                out.update(frame_id=hdr.frame_id, stamp=hdr.stamp.sec + 1e-9 * hdr.stamp.nanosec)
                if seg_pts: self.pub_cloud.publish(self.to_cloud(hdr, np.concatenate(seg_pts), np.concatenate(seg_ids)))
            self.pub_json.publish(String(data=json.dumps(out)))
            if salsa_out and hdr is not None:
                plan = dict(request_id=req, frame_id=hdr.frame_id, stamp=out['stamp'], command=self.salsa,
                            pose_convention='pinch point between the finger contacts; rotation columns x, y=closing, z=approach',
                            objects=salsa_out)
                self.pub_salsa.publish(String(data=json.dumps(plan)))
                first = next((o for o in salsa_out if o.get('candidates')), None)
                if first is not None:   # PoseArray of the best candidate of the first planned object, for RViz / the planner
                    pa = PoseArray(header=Header(frame_id=hdr.frame_id, stamp=hdr.stamp))
                    for q in first['candidates'][0]['poses']:
                        ps = Pose(); ps.position.x, ps.position.y, ps.position.z = q['position']
                        ps.orientation.x, ps.orientation.y, ps.orientation.z, ps.orientation.w = q['quaternion_xyzw']; pa.poses.append(ps)
                    self.pub_poses.publish(pa)
                    self.get_logger().info(f"SALSA '{self.salsa}' object {first['id']} ({first['label']}): {len(first['candidates'])} candidate(s), "
                                           f"{len(pa.poses)} poses each -> /salsa/plan, /salsa/poses")
            if self.web is not None:
                try: self.web.add(req, out.get('frame_id', ''), web_objs)
                except Exception: self.get_logger().error('web render failed:\n' + traceback.format_exc())
        except Exception:
            self.get_logger().error(traceback.format_exc())
        finally:
            with self.lock: self.busy = False

    @staticmethod
    def to_cloud(hdr, xyz, seg):
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(('x', 'y', 'z', 'segment'))]
        data = np.concatenate([xyz.astype(np.float32), seg[:, None]], 1)
        return pc2.create_cloud(Header(frame_id=hdr.frame_id, stamp=hdr.stamp), fields, data.tolist())

    def save(self, req, oid, label, r):
        import open3d as o3d
        d = os.path.join(self.save_dir, f'{req}_{oid}_{str(label).replace(" ", "_")}'); os.makedirs(d, exist_ok=True)
        for s in r['segments']:
            o3d.io.write_point_cloud(os.path.join(d, f'{s["name"]}.ply'), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s['points'])))
        with open(os.path.join(d, 'bitsi.json'), 'w') as f:
            json.dump({**{k: v for k, v in r.items() if k != 'segments'}, 'segments': [{k: v for k, v in s.items() if k != 'points'} for s in r['segments']]}, f, indent=1)


def main():
    rclpy.init(); node = BitsiNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    main()
