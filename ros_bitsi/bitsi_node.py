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
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import String, Header
from seg_interfaces.msg import ObjectClouds, SegObjects

from bitsi_core import run_bitsi

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
                      strength_threshold=P('strength_threshold', .01), ibr_tolerance=P('ibr_tolerance', .07), voxel=P('voxel', .004))
        self.save_dir = P('save_dir', ''); clouds_topic = P('clouds_topic', '/perception/clouds'); result_topic = P('result_topic', '/perception/result')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
        self.labels = {}; self.done = set(); self.lock = threading.Lock(); self.busy = False; self.logged = set()
        self.create_subscription(SegObjects, result_topic, self.on_result, qos)
        self.create_subscription(ObjectClouds, clouds_topic, self.on_clouds, qos)
        self.pub_json = self.create_publisher(String, '/bitsi/result', qos)
        self.pub_cloud = self.create_publisher(PointCloud2, '/bitsi/segments', qos)
        self.get_logger().info(f'listening on {clouds_topic} and {result_topic}; params {self.p}; save_dir={self.save_dir or "(off)"}')

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
                        oid = first_attr(e, ID_KEYS); self.labels[(str(req), str(i if oid is None else oid))] = str(lab)

    def on_clouds(self, msg):
        self.log_layout('ObjectClouds', msg); req = str(first_attr(msg, REQ_KEYS))
        items = [(str(oid), lab, c) for oid, lab, c in find_clouds(msg) if (req, str(oid)) not in self.done]
        if not items: return
        with self.lock:
            if self.busy: self.get_logger().info(f'busy; request {req} will be picked up from the next repeat'); return
            self.busy = True
        threading.Thread(target=self.process, args=(req, items), daemon=True).start()

    def process(self, req, items):
        try:
            out = dict(request_id=req, objects=[]); seg_pts = []; seg_ids = []; hdr = None
            for oid, lab, cloud in items:
                hdr = cloud.header; label = lab or self.labels.get((req, oid), '')
                pts = pc2.read_points_numpy(cloud, field_names=('x', 'y', 'z'), skip_nans=True).astype(np.float64)
                t0 = time.time(); rec = dict(id=oid, label=str(label), n_points=int(len(pts)))
                try:
                    r = run_bitsi(pts, **self.p); rec.update({k: v for k, v in r.items() if k != 'segments'})
                    rec['segments'] = [{k: v for k, v in s.items() if k != 'points'} for s in r['segments']]
                    for j, s in enumerate(r['segments']): seg_pts.append(s['points']); seg_ids.append(np.full(len(s['points']), j, np.float32))
                    if self.save_dir: self.save(req, oid, label, r)
                except Exception as e:
                    rec['error'] = f'{type(e).__name__}: {e}'
                rec['seconds'] = round(time.time() - t0, 3); out['objects'].append(rec); self.done.add((req, oid))
                self.get_logger().info(f'request {req} object {oid} ({label}): {len(pts)} pts -> '
                                       f'{len(rec.get("segments", []))} segments in {rec["seconds"]}s' + (f' ERROR {rec["error"]}' if 'error' in rec else ''))
            if hdr is not None:
                out.update(frame_id=hdr.frame_id, stamp=hdr.stamp.sec + 1e-9 * hdr.stamp.nanosec)
                if seg_pts: self.pub_cloud.publish(self.to_cloud(hdr, np.concatenate(seg_pts), np.concatenate(seg_ids)))
            self.pub_json.publish(String(data=json.dumps(out)))
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
