"""Smoke test: BITC profiles + multi-scale Chebyshev cuts on a few ShapeNetPart objects."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fitpart.data import shapenetpart_split, load_shapenetpart, shapenetpart_categories
from fitpart.profiles import sweep_directions, bitc_profile
from fitpart.cheb import multiscale_cuts, jump_threshold_cuts
from numpy.polynomial import chebyshev as C

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
cats = shapenetpart_categories()
split = shapenetpart_split("test")
want = ["Mug", "Knife", "Airplane", "Chair", "Guitar", "Lamp"]
seen = {}
for sid, oid in split:
    n = cats[sid]
    if n in want and n not in seen:
        seen[n] = (sid, oid)
for name, (sid, oid) in seen.items():
    P, Nm, lab = load_shapenetpart(sid, oid)
    t0 = time.time()
    dirs = sweep_directions(P)
    profs = [bitc_profile(P, d) for d in dirs]
    res = [multiscale_cuts(p["t"], p["values"]) for p in profs]
    dt = time.time() - t0
    print(f"{name} {oid}: N={len(P)} dirs={len(dirs)} slabs={len(profs[0]['t'])} "
          f"deg={res[0][1]['degrees']} cuts/dir={[len(r[0]) for r in res]} time={dt:.2f}s")
    p, (cuts, info) = profs[0], res[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(p["t"], p["values"], ".", color="0.5", label="BITC (axis 1)")
    yr = np.ptp(p["values"]); y0 = p["values"].min()
    xs = np.linspace(-1, 1, 400)
    for deg, c in zip(info["degrees"], info["coeffs"]):
        tt = info["lo"] + (xs + 1) / 2 * (info["hi"] - info["lo"])
        ax.plot(tt, y0 + yr * C.chebval(xs, c), lw=1, label=f"deg {deg}")
    for cu in cuts:
        ax.axvline(cu.t, color="r", alpha=cu.strength, lw=2)
    for cu in jump_threshold_cuts(p["t"], p["values"], 0.2):
        ax.axvline(cu.t, color="k", ls=":", lw=1)
    ax.set_title(f"{name}: red = Chebyshev inflections (alpha = s), dotted = jumps >= 0.2")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(out, f"profile_{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
