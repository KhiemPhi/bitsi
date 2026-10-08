"""BITSI (existing main.py pipeline) segmentation of each prepared mug -> <oid>.bitsi.npy (env: pointcloud)."""
import contextlib, glob, io, os, sys
sys.path.insert(0, '/home/khiem/bitsi')
import numpy as np
from fitpart.baselines.bitsi_v0 import segment
d = sys.argv[1]
for f in sorted(glob.glob(f'{d}/obj_*.npz')):
    z = np.load(f, allow_pickle=True)
    with contextlib.redirect_stdout(io.StringIO()):
        lab = segment(z['P'].astype(float), z['N'].astype(float), gripper_width=0.08)
    np.save(f[:-4] + '.bitsi.npy', lab)
    h = z['handle']
    best = max(np.unique(lab), key=lambda l: ((lab == l) & h).sum() / ((lab == l) | h).sum())
    iou = ((lab == best) & h).sum() / ((lab == best) | h).sum()
    print(os.path.basename(f), 'segments', lab.max() + 1, 'best handle IoU %.2f' % iou, flush=True)
