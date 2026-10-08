"""FitPart + SAM 3 and SAM 3-lifted selections of "handle" on each prepared mug (env: sam3, GPU).
SAM 3 renders come from 200k points sampled on the mesh (full resolution), mapped to the 20k cloud
by nearest neighbour; no colours (HANDAL has none). Writes <oid>.sel.npz."""
import glob, os, sys, time
sys.path.insert(0, '/home/khiem/bitsi')
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
from fitpart import lifting
from fitpart.pipeline import Config, FitPart, _up_rotation
from fitpart.render import render_views
from fitpart.sam3_select import lift_query, query_masks, superpoint_scores, union_score
from fitpart.selection import node_membership, select
d = sys.argv[1]
fp = FitPart(Config(up=(0.0, 0.0, 1.0)))
for f in sorted(glob.glob(f'{d}/obj_*.npz')):
    out = f[:-4] + '.sel.npz'
    if os.path.exists(out):
        continue
    t = time.time()
    z = np.load(f, allow_pickle=True)
    P, N = z['P'].astype(float), z['N'].astype(float)
    L = fp.lift(P, N, None, geometry_variants=['divisive'])
    res = fp.assemble(L)
    sp, H = res.sp, res.hier
    mesh = o3d.io.read_triangle_mesh(str(z['mesh'])); mesh.scale(0.001, center=np.zeros(3)); mesh.compute_vertex_normals()
    full = mesh.sample_points_uniformly(200000)
    Pf, Nf = np.asarray(full.points), np.asarray(full.normals)
    R = _up_rotation((0.0, 0.0, 1.0))
    rend = render_views(Pf @ R.T, None, Nf @ R.T, k=10, size=512)
    nn = cKDTree(P).query(Pf)[1]
    sp_pix = np.stack([np.where(p2p >= 0, sp[nn[np.maximum(p2p, 0)]], -1) for p2p in rend['pix2pt']])
    lifted = lift_query(query_masks(rend['images'], ['handle'])[0], sp_pix, sp.max() + 1)
    _, sizes = node_membership(H, np.bincount(sp).astype(float))
    r = union_score(H, sp, lifted)
    h, _ = select(r, sizes, None, fp.cfg.eps_r, min_size=0.01 * len(sp))
    fit_mask = np.isin(sp, H.members[h])
    direct = (superpoint_scores(sp, lifted) >= 0.3)[sp]
    gt = z['handle']
    iou = lambda m: (m & gt).sum() / max((m | gt).sum(), 1)
    np.savez_compressed(out, fitpart=fit_mask, sam3_direct=direct, sam3_max=float(r.max()))
    print(os.path.basename(f), f'{time.time()-t:.0f}s  FitPart handle IoU {iou(fit_mask):.2f}  SAM3-direct IoU {iou(direct):.2f}  (SAM3 max node score {r.max():.2f})', flush=True)
