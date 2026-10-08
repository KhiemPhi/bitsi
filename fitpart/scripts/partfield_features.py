"""Baseline: PartField (ICCV 2025) point features for every cached object.

Writes each cached object's points to <work>/ply/<cat>__<oid>.ply, runs PartField's
own inference in point-cloud mode (features keep the input point order), and copies
the features next to the cache entry as <cat>__<oid>.partfield.npy (float16, N x 448).

Run with the partfield env from the PartField repo root:
  cd PartField && ~/miniconda3/envs/partfield/bin/python /home/khiem/bitsi/fitpart/scripts/partfield_features.py CACHE_DIR WORK_DIR
"""
import glob, os, pickle, shutil, subprocess, sys
import numpy as np
from plyfile import PlyData, PlyElement

cache, work = sys.argv[1], sys.argv[2]
ply_dir = os.path.join(work, "ply")
os.makedirs(ply_dir, exist_ok=True)
todo = []
for f in sorted(glob.glob(os.path.join(cache, "*.pkl"))):
    if ".sam3" in os.path.basename(f):
        continue
    out = f[:-4] + ".partfield.npy"
    if os.path.exists(out):
        continue
    L = pickle.load(open(f, "rb"))
    uid = os.path.basename(f)[:-4]
    P = L["points"].astype(np.float32)
    v = np.array([tuple(p) for p in P], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(os.path.join(ply_dir, uid + ".ply"))
    todo.append((uid, out))
print(len(todo), "objects to run", flush=True)
if todo:
    subprocess.run([sys.executable, "partfield_inference.py", "-c", "configs/final/demo.yaml", "--opts",
                    "continue_ckpt", "model/model_objaverse.ckpt", "result_name", "partfield_features/fitpart",
                    "dataset.data_path", ply_dir, "is_pc", "True"], check=True)
    for uid, out in todo:
        # the dataloader's uid is the file name without extension
        src = f"exp_results/partfield_features/fitpart/part_feat_{uid}_0.npy"
        if os.path.exists(src):
            np.save(out, np.load(src).astype(np.float16))
        else:
            print("missing", src, flush=True)
print("done", flush=True)
