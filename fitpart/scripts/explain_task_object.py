"""BITSI-cuts explainer (scripts/video_bitsi3.render) for the exact cloud of a robot trial.
usage (sam3 env for skimage-free fit; any env with matplotlib/imageio):
  python explain_task_object.py pivot SEED OUT.mp4          (procedural box of tasks.run)
  python explain_task_object.py pour CAT/OID OUT.mp4        (HANDAL object, upright as in the pour scene)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.bitsi4 import BITSI4
from fitpart.scripts.video_bitsi3 import render

task, key, out = sys.argv[1], sys.argv[2], sys.argv[3]
if task == "pour":
    from fitpart.axes import closed_end_sign, revolution_axis, rotation_to_up
    cat, oid = key.split("/")
    z = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_pour", cat, f"{oid}.npz"), allow_pickle=True)
    P, N = z["P"].astype(float), z["N"].astype(float)
    a, c, _ = revolution_axis(P, N)
    R = rotation_to_up(a * closed_end_sign(P, a, c))
    W, Wn = P @ R.T, N @ R.T
    name = f"{cat} {oid}"
else:
    from fitpart.tasks.run import setup
    sc, W, Wn, extra = setup(task, key)
    name = f"{task} box {key}"
render(W, Wn, None, out, name, M=BITSI4(up=(0, 0, 1.0)), version="v4", n_sweep=90, n_end=50)
os._exit(0)
