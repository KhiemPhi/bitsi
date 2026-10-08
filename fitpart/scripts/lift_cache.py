"""Stage 1 (GPU): lift objects once and cache everything the ablations need.

usage:
  python lift_cache.py partnete  CACHE_DIR [per_category] [vl_model]
  python lift_cache.py shapenetpart CACHE_DIR [per_category] [vl_model]
Skips objects already cached, so it can be resumed or run alongside a download.
"""
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.data import (load_partnete, load_shapenetpart, partnete_objects,
                          shapenetpart_categories, shapenetpart_split)
from fitpart.pipeline import Config, FitPart

GEOMS = ["none", "divisive", "divisive_jump", "divisive_single", "divisive_extrema", "divisive_axis", "global"]


def items(dataset, per_cat):
    taken = defaultdict(int)
    if dataset == "partnete":
        for cat, d in partnete_objects("test"):
            if taken[cat] < per_cat:
                taken[cat] += 1
                yield cat, os.path.basename(d), d
    else:
        cats = shapenetpart_categories()
        for sid, oid in shapenetpart_split("test"):
            if taken[sid] < per_cat:
                taken[sid] += 1
                yield cats[sid], oid, (sid, oid)


def main():
    dataset, cache = sys.argv[1], sys.argv[2]
    per_cat = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    vl = sys.argv[4] if len(sys.argv) > 4 else "siglip"
    os.makedirs(cache, exist_ok=True)
    up = (0.0, 1.0, 0.0)  # both ShapeNetPart and PartNet-E are y-up
    fp = FitPart(Config(vl_model=vl, up=up))
    for cat, oid, ref in items(dataset, per_cat):
        path = os.path.join(cache, f"{cat}__{oid}.pkl")
        if os.path.exists(path):
            continue
        try:
            t = time.time()
            if dataset == "partnete":
                P, N, C, sem, ins = load_partnete(ref)
            else:
                P, N, sem = load_shapenetpart(*ref)
                C, ins = None, None
            L = fp.lift(P, N, C, geometry_variants=GEOMS)
            L["thumb"] = L.pop("images")[::3, ::2, ::2]  # a few small views for figures
            L.update(cat=cat, oid=oid, sem=sem, ins=ins, n_points=len(P),
                     points=P.astype(np.float32), normals=N.astype(np.float32))
            with open(path, "wb") as f:
                pickle.dump(L, f)
            print(f"{cat} {oid} {time.time() - t:.1f}s "
                  + " ".join(f"{k}={v:.1f}" for k, v in L["timings"].items()), flush=True)
        except Exception:
            print(f"FAILED {cat} {oid}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
