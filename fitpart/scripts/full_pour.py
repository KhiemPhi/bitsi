"""Pour comparison on FULL object scans (the 20k mesh samples, no occlusion), as in section 18a, for
BITSI v4 (loop child), Find3D ("handle of a <category>") and the whole object.
usage (graspsim env): python full_pour.py DATA_DIR OID OUT.jsonl [--video VIDEO_DIR]
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.scripts.eih_pour import find3d_mask
from fitpart.scripts.panda_pour import part_mask, run as pour_run

ROBOT = os.environ.get("ROBOT", "panda")


def main():
    D, oid, out = sys.argv[1], sys.argv[2], sys.argv[3]
    vdir = sys.argv[sys.argv.index("--video") + 1] if "--video" in sys.argv else None
    if vdir:
        os.makedirs(vdir, exist_ok=True)
    z = np.load(f"{D}/{oid}.npz", allow_pickle=True)
    P, N, handle = z["P"].astype(float), z["N"].astype(float), z["handle"]
    cat = str(z["cat"]) if "cat" in z.files else "mug"
    sel = {"find3d": find3d_mask(P, N, cat)}       # same points, object frame (labels are per point)
    res = dict(oid=oid, cat=cat)
    for m in ("bitsi4", "find3d", "whole"):
        r = pour_run(D, oid, m, video=f"{vdir}/full_{cat}_{oid}_{m}.mp4" if vdir else None, sel=sel, robot=ROBOT)
        r.pop("video", None)
        if m != "whole":
            msk = part_mask(m, P, N, sel)[0] if m == "find3d" else None
            if msk is not None:
                r["part_iou_with_handle"] = float((msk & handle).sum() / max((msk | handle).sum(), 1))
        res[m] = r
    with open(out, "a") as f:
        f.write(json.dumps(res, default=float) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v.get(kk) for kk in ("success", "task_success", "planned")})
                      for k, v in res.items()}, default=float))


if __name__ == "__main__":
    main()
