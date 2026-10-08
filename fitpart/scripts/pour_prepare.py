"""Sample a labelled 20k-point cloud per HANDAL mug from its real mesh (area-proportional samples of
the GT handle and rest meshes; labels are used for evaluation only) -> <out>/<oid>.npz.
env: graspsim.  usage: python pour_prepare.py OUT_DIR [n_points] [category=mug]"""
import glob, os, sys
import numpy as np, trimesh
out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
cat = sys.argv[3] if len(sys.argv) > 3 else 'mug'
H = f'/home/khiem/Robotics/obj-decomposition/HANDAL/{cat}'
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(f'{H}/models/*.ply')):
    oid = os.path.basename(f)[:-4]
    hm = trimesh.load(f'{H}/models_parts/{oid}_handle.ply', force='mesh'); hm.apply_scale(0.001)
    rm = trimesh.load(f'{H}/models_parts/{oid}_not.ply', force='mesh'); rm.apply_scale(0.001)
    nh = int(round(n * hm.area / (hm.area + rm.area)))
    ph, fh = trimesh.sample.sample_surface(hm, nh, seed=0)
    pr, fr = trimesh.sample.sample_surface(rm, n - nh, seed=0)
    P = np.vstack([ph, pr]); N = np.vstack([hm.face_normals[fh], rm.face_normals[fr]])
    gt = np.r_[np.ones(nh, int), np.zeros(n - nh, int)]
    perm = np.random.default_rng(0).permutation(n)
    np.savez_compressed(f'{out}/{oid}.npz', P=P[perm].astype(np.float32), N=N[perm].astype(np.float32),
                        handle=gt[perm].astype(bool), mesh=f, cat=cat)
    print(oid, nh, 'handle points', flush=True)
