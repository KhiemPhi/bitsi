"""Stage 1b (GPU): SAM 3 text-prompted masks for every part name, lifted onto each
cached object's superpoints. Re-renders the views from the cached points (deterministic),
so it can run after lift_cache.py. Writes <obj>.sam3.pkl next to the cache entry.

usage: python sam3_cache.py CACHE_DIR META_JSON [n_views=10] [extra_queries_json]
env: FITPART_SAM3_COLOR=1 re-renders with the objects' real colors (reloaded from PartNet-E),
     FITPART_SAM3_THRESH (default 0.3), FITPART_SAM3_SUFFIX (default ".sam3.pkl").
extra_queries_json: optional {category: [phrases]} or the task file format
{category: [{"task": ..., "part": ...}]}; those phrases are queried in addition to the
part names. Existing sidecars are extended with any missing queries.
"""
import glob, json, os, pickle, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart import lifting
from fitpart.pipeline import _up_rotation
from fitpart.render import render_views
from fitpart.sam3_select import lift_query, query_masks

COLOR = os.environ.get("FITPART_SAM3_COLOR", "0") == "1"
FULLRES = os.environ.get("FITPART_SAM3_FULLRES", "0") == "1"  # render the full-resolution cloud
THRESH = float(os.environ.get("FITPART_SAM3_THRESH", "0.3"))
SUFFIX = os.environ.get("FITPART_SAM3_SUFFIX", ".sam3.pkl")


def main():
    cache, meta_json = sys.argv[1], sys.argv[2]
    n_views = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    extra = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {}
    extra = {c: [x["task"] if isinstance(x, dict) else x for x in v]
             for c, v in extra.items() if not c.startswith("_")}
    meta = json.load(open(meta_json))
    for f in sorted(glob.glob(os.path.join(cache, "*.pkl"))):
        if ".sam3" in os.path.basename(f):
            continue
        out = f[:-4] + SUFFIX
        have = pickle.load(open(out, "rb")) if os.path.exists(out) else {}
        try:
            t = time.time()
            L = pickle.load(open(f, "rb"))
            names = meta.get(L["cat"], [])
            queries = [q for q in names + extra.get(L["cat"], []) if q not in have]
            if not queries:
                continue
            P, N = L["points"].astype(float), L["normals"].astype(float)
            R = _up_rotation((0.0, 1.0, 0.0))
            C = None
            sp = L["sp"]
            M = sp.max() + 1
            from fitpart.data import PARTNETE_ROOT, load_partnete
            obj_dir = os.path.join(PARTNETE_ROOT, "data", "test", L["cat"], L["oid"])
            if FULLRES:
                # render all ~200k points (sharper images), then map every rendered point to its
                # nearest point of the cached subsample, which carries the superpoint labels
                from scipy.spatial import cKDTree
                Pf, Nf, Cf, _, _ = load_partnete(obj_dir, max_points=None)
                rend = render_views(Pf @ R.T, Cf if COLOR else None, Nf @ R.T, k=n_views, size=512)
                nn = cKDTree(P).query(Pf)[1]
                sp_pix = np.stack([np.where(p2p >= 0, sp[nn[np.maximum(p2p, 0)]], -1) for p2p in rend["pix2pt"]])
            else:
                if COLOR:
                    _, _, C, _, _ = load_partnete(obj_dir)
                rend = render_views(P @ R.T, C, N @ R.T, k=n_views, size=512)
                sp_pix = np.stack([lifting.sp_pixel_index(p2p, sp) for p2p in rend["pix2pt"]])
            qm = query_masks(rend["images"], queries, threshold=THRESH)
            have.update({q: lift_query(qm[i], sp_pix, M) for i, q in enumerate(queries)})
            pickle.dump(have, open(out + ".tmp", "wb"))
            os.replace(out + ".tmp", out)
            print(f"{L['cat']} {L['oid']} {len(queries)} queries {time.time() - t:.1f}s", flush=True)
        except Exception:
            print("FAILED", f, flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
