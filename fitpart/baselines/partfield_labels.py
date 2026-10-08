"""PartField (Liu et al., ICCV 2025) per-point part labels for every object, cached for eval_all.py.

The authors' point-cloud protocol: features from partfield_inference.py (Objaverse checkpoint,
is_pc True, input point order kept), L2-normalised, KMeans(random_state=0). KMeans needs a part
count; we give it the GT count of each object (oracle K, clipped to >= 2), which favours the
baseline: BITSI chooses its own part count.

usage (partfield env, from ~/bitsi/ext/PartField with the nvjitlink LD_LIBRARY_PATH fix):
  python /home/khiem/bitsi/fitpart/baselines/partfield_labels.py DATASET WORK_DIR
"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import KMeans

from fitpart.data import load_partnete, load_shapenetpart
from fitpart.scripts.eval_all import items

CACHE = os.path.expanduser("~/.cache/fitpart_baselines/partfield")


def gt_and_points(it):
    if "dir" in it:
        P, N, C, sem, ins = load_partnete(it["dir"])
        gt = (ins if ins is not None else sem).copy()
        gt[sem < 0] = gt.max() + 1
        return P, gt
    P, N, gt = load_shapenetpart(it["sid"], it["oid"])
    return P, gt


def main(dataset, work, chunk=100, k_from=None, tag="partfield", keep=None):
    """k_from: optional {key: K} (e.g. BITSI v4's own segment count per object) instead of the GT
    count; labels then go to the cache under `tag`. keep: optional set of keys to run."""
    """Chunks of `chunk` objects: write points, run PartField, cluster, delete the features
    (20k x 448 float32 per PartNet-E object would not fit on disk all at once)."""
    import shutil
    out_dir = os.path.join(os.path.dirname(CACHE), tag, dataset)
    os.makedirs(out_dir, exist_ok=True)
    todo = [it for it in items(dataset) if not os.path.exists(os.path.join(out_dir, it["key"].replace("/", "__") + ".npy"))
            and (keep is None or it["key"] in keep)]
    print(len(todo), "objects to run", flush=True)
    for c0 in range(0, len(todo), chunk):
        ply_dir = os.path.join(work, dataset, "ply")
        feat_dir = f"exp_results/partfield_features/fitpart_{dataset}"
        shutil.rmtree(ply_dir, ignore_errors=True)
        shutil.rmtree(feat_dir, ignore_errors=True)
        os.makedirs(ply_dir)
        ks = []
        for it in todo[c0:c0 + chunk]:
            uid = it["key"].replace("/", "__")
            P, gt = gt_and_points(it)
            v = np.zeros(len(P), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
            v["x"], v["y"], v["z"] = P.T
            PlyData([PlyElement.describe(v, "vertex")], text=False).write(os.path.join(ply_dir, uid + ".ply"))
            ks.append((uid, max(2, int(k_from[it["key"]]) if k_from is not None else len(np.unique(gt)))))
        subprocess.run([sys.executable, "partfield_inference.py", "-c", "configs/final/demo.yaml", "--opts",
                        "continue_ckpt", "model/model_objaverse.ckpt", "result_name", f"partfield_features/fitpart_{dataset}",
                        "dataset.data_path", ply_dir, "is_pc", "True"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for uid, k in ks:
            src = os.path.join(feat_dir, f"part_feat_{uid}_0.npy")
            if not os.path.exists(src):
                print("missing", src, flush=True)
                continue
            F = np.load(src).astype(np.float32)
            F /= np.linalg.norm(F, axis=-1, keepdims=True) + 1e-12
            np.save(os.path.join(out_dir, uid + ".npy"),
                    KMeans(n_clusters=k, random_state=0).fit(F).labels_.astype(np.int16))
        shutil.rmtree(feat_dir, ignore_errors=True)
        print(c0 + len(ks), "/", len(todo), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    # optional 3rd arg: a result .jsonl whose "v4" (or frozen v4 variant) n_pred sets K per object
    if len(sys.argv) > 3:
        import json
        kf = {}
        for l in open(sys.argv[3]):
            r = json.loads(l)
            v = r.get("v4") or r.get("v4 children=mdl cuts=no pen10")
            if v and "n_pred" in v:
                kf[r["key"]] = v["n_pred"]
        main(sys.argv[1], sys.argv[2], k_from=kf, tag="partfield_v4k", keep=set(kf))
    else:
        main(sys.argv[1], sys.argv[2])
