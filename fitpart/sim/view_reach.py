"""Which eye-in-hand camera poses can the robot reach? (graspsim env: MuJoCo)

The wrist camera (RealSense D405-like, mounted on the hand looking along the gripper's approach
axis) sits CAM_BEHIND_TCP behind the point between the fingertips. A viewpoint = camera position
on a sphere around the object centre, optical axis through the centre. It is reachable when IK
finds an arm configuration putting the hand there (any of 4 rolls about the optical axis) that is
free of self- and table collisions. The object is not in the collision model: the view distance
keeps the fingertips >= 3 cm outside the object's bounding sphere (see eyeinhand.view_distance).

usage: python view_reach.py REQUESTS.json OUT.json [robot=panda]
  REQUESTS.json: list of {"key", "centre": [x,y,z], "dist": d}; OUT.json: key -> list of reachable
  (az_deg, el_deg) on the shared candidate grid (eyeinhand.CANDIDATE_GRID).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

CAM_BEHIND_TCP = 0.10
AZ = np.arange(0, 360, 15.0)
EL = np.array([15.0, 35.0, 55.0, 75.0])


def view_dir(az, el):
    a, e = np.deg2rad(az), np.deg2rad(el)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def hand_frames(forward):
    """4 hand orientations (columns x, y, z) with z = forward, rolled by 90 deg."""
    z = forward / np.linalg.norm(forward)
    x = np.cross([0, 0, 1.0], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    out = []
    for k in range(4):
        c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        xr, yr = c * x + s * y, -s * x + c * y
        out.append(np.stack([xr, yr, z], 1))
    return out


class Reach:
    def __init__(self, robot="panda"):
        self.robot = robot
        if robot == "panda":
            from fitpart.sim.panda_scene import PandaScene
            dummy = [(np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], float), np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]
            self.sim = PandaScene(dummy, obj_xy=(3.0, 3.0))
            self.sim.reset(settle=0.0)
        else:
            from fitpart.sim.kinova_scene import KinovaScene
            self.sim = KinovaScene()
            self.sim.reset()

    def reachable(self, centre, dist):
        out = []
        for az in AZ:
            for el in EL:
                d = view_dir(az, el)
                cam = centre + dist * d
                tcp = cam - CAM_BEHIND_TCP * d          # fingertips are ahead of the camera
                for R in hand_frames(-d):
                    q = self.sim.ik(tcp, R, restarts=4, iters=150)
                    if q is not None and not self.sim.in_collision(q):
                        out.append((float(az), float(el)))
                        break
        return out


if __name__ == "__main__":
    reqs = json.load(open(sys.argv[1]))
    out_path = sys.argv[2]
    robot = sys.argv[3] if len(sys.argv) > 3 else "panda"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    R = Reach(robot)
    cache = {}
    for i, rq in enumerate(reqs):
        if rq["key"] in out:
            continue
        ck = (round(rq["centre"][2], 2), round(rq["dist"], 2))   # object xy is fixed on the table
        if ck not in cache:
            cache[ck] = R.reachable(np.asarray(rq["centre"], float), rq["dist"])
        out[rq["key"]] = cache[ck]
        if i % 20 == 0:
            json.dump(out, open(out_path, "w"))
            print(i, "/", len(reqs), len(cache), "distinct", flush=True)
    json.dump(out, open(out_path, "w"))
    print("finished", flush=True)
