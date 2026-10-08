"""Eye-in-hand partial views: segmentation quality vs number of views, per view strategy.

Objects: the held-out PartNet-E objects of tabletop categories (category size <= 0.6 m; doors,
fridges, furniture are not handled by a table-mounted arm): indomain (all 9 swept categories) +
outdomain (first 4 of each tabletop non-swept category).
Per object and strategy (eyeinhand.py: orbit, nbv_cov, nbv_bitsi; all start at HOME), views are
added up to K_MAX. At K in EVAL_K the fused cloud is segmented and scored
  observed  partition scores on the fused points (GT of the surface sample each came from)
  full      labels carried to every surface sample by nearest fused point, scored on all of them
            (parts never seen cost here)
plus coverage (fraction of surface samples within 2 spacings of a fused point) and camera travel.
"all" = every reachable view at once (upper bound).

usage:
  python eval_eyeinhand.py requests REQ.json                       (sam3 env)
  python view_reach.py REQ.json REACH.json panda                   (graspsim env)
  python eval_eyeinhand.py run REACH.json OUT.jsonl [robot]        (sam3 env)
"""
import json, os, sys, time, warnings, zlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy.spatial import cKDTree

from fitpart import eyeinhand as E
from fitpart.data import load_partnete
from fitpart.metrics import partition_scores
from fitpart.scales import CATEGORY_SIZE_M
from fitpart.scripts.eval_all import items

K_MAX = 8
STRATS = os.environ.get("STRATS", "orbit,nbv_cov,nbv_bitsi").split(",")
EVAL_K = (1, 2, 3, 4, 6, 8)
BASE_K = (1, 3, 8)              # v3 and sweep events are scored at these K only
SAVE_K = (1, 3, 8)              # fused clouds saved for the GPU baselines
CLOUDS = os.path.expanduser("~/.cache/fitpart_eih")


def objects():
    return [it for it in items("partnete")
            if it["split"] in ("indomain", "outdomain") and (CATEGORY_SIZE_M.get(it["cat"]) or 9) <= 0.6]


def seed_of(*a):
    return zlib.crc32("|".join(map(str, a)).encode())


def load(it):
    P, N, C, sem, ins = load_partnete(it["dir"])
    gt = (ins if ins is not None else sem).copy()
    gt[sem < 0] = gt.max() + 1
    W, Nw, _ = E.place(P, N, it["cat"], seed=seed_of(it["key"]))
    return W, Nw, C, gt


def segment(method, pts, n, col):
    if method == "v4":
        from fitpart.bitsi4 import BITSI4
        M = BITSI4()
        lab, root = M.fit(pts, n, col)
        return lab, root, M.h
    if method == "v3":
        from fitpart.bitsi3 import BITSI3
        return BITSI3().fit(pts, n, col)[0], None, None
    from fitpart.bitsi4 import hog_labels
    return hog_labels(pts, n, col), None, None


def score(lab, idx, pts, gt, W, tree_full_h):
    obs = partition_scores(gt[idx], lab)
    nn = cKDTree(pts).query(W)[1]
    full = partition_scores(gt, lab[nn])
    return obs, full


def run_object(args):
    it, reach, robot = args
    warnings.filterwarnings("ignore")
    t0 = time.time()
    W, Nw, C, gt = load(it)
    h = E.spacing(W)
    centre, diag = E.object_frame(W)
    dist = E.view_distance(diag)
    reach = [tuple(r) for r in reach]
    out = dict(key=it["key"], cat=it["cat"], split=it["split"], robot=robot, n_reach=len(reach), dist=dist)
    if not reach:
        out["error"] = "no reachable view"
        return out
    home = E.nearest_reachable(*E.HOME, reach)
    tree_W = cKDTree(W)
    views_cache = {}

    def view(v):
        if v not in views_cache:
            cam, Rc = E.camera_pose(centre, dist, *v)
            idx, meas = E.observe(W, Nw, cam, Rc, h, np.random.default_rng(seed_of(it["key"], *v)))
            views_cache[v] = (cam, Rc, idx, meas)
        return views_cache[v]

    def evaluate(F, k, strat, res, v4fit=None):
        idx, pts, n = F.cloud()
        col = C[idx] if C is not None else None
        cov = float(np.mean(cKDTree(pts).query(W)[0] < 2 * h))
        row = dict(k=k, n_points=len(idx), coverage=cov)
        for m in ("v4",) + (("v3", "hog") if k in BASE_K else ()):
            try:
                lab = v4fit[0] if (m == "v4" and v4fit is not None) else segment(m, pts, n, col)[0]
                row[m], row[m + "_full"] = score(lab, idx, pts, gt, W, None)
            except Exception as e:
                row[m] = dict(error=repr(e))
        if k in SAVE_K:
            d = os.path.join(CLOUDS, robot, strat)
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, f"{it['key'].replace('/', '__')}__K{k}.npz"),
                                idx=idx.astype(np.int32), pts=pts.astype(np.float32), n=n.astype(np.float16))
        res.append(row)

    def travel(seq):
        tot = 0.0
        for a, b in zip(seq[:-1], seq[1:]):
            ang = np.arccos(np.clip(E.view_dir(*a) @ E.view_dir(*b), -1, 1))
            tot += ang * dist
        return tot

    # -------- orbit
    if "orbit" not in STRATS and "nbv_cov" not in STRATS:
        return run_box_only(out, it, reach, robot, view, evaluate, travel, home, centre, dist, h, W, t0)
    seq = E.orbit_sequence(reach, K_MAX)
    F, res = E.Fused(h, seed=1), []
    for k, v in enumerate(seq, 1):
        cam, Rc, idx, meas = view(v)
        F.add(idx, meas, cam)
        if k in EVAL_K:
            evaluate(F, k, "orbit", res)
    out["orbit"] = dict(views=seq, travel=travel(seq), evals=res)

    # -------- volumetric NBV
    G = E.VoxelNBV(centre, diag)
    F, res, seq = E.Fused(h, seed=1), [], []
    v = home
    for k in range(1, K_MAX + 1):
        cam, Rc, idx, meas = view(v)
        F.add(idx, meas, cam)
        G.integrate(cam, Rc, meas)
        seq.append(v)
        if k in EVAL_K:
            evaluate(F, k, "nbv_cov", res)
        rest = [r for r in reach if r not in seq]
        if not rest:
            break
        gains = [G.gain(*E.camera_pose(centre, dist, *r)) for r in rest]
        v = rest[int(np.argmax(gains))]
    out["nbv_cov"] = dict(views=seq, travel=travel(seq), evals=res)

    # -------- bounding-box completion NBV (box refitted each view; stop when < 5% of its voxels unknown)
    if "nbv_box" in STRATS:
        G = E.BoxNBV(centre, diag)
        F, res, seq, stop_k = E.Fused(h, seed=1), [], [], None
        v = home
        for k in range(1, K_MAX + 1):
            cam, Rc, idx, meas = view(v)
            F.add(idx, meas, cam)
            G.integrate(cam, Rc, meas)
            seq.append(v)
            G.fit_box(F.cloud()[1], cams=[E.camera_pose(centre, dist, *u)[0] for u in seq])
            if stop_k is None and G.unknown_fraction() < G.stop:
                stop_k = k
            if k in EVAL_K:
                evaluate(F, k, "nbv_box", res)
            rest = [r for r in reach if r not in seq]
            if not rest:
                break
            gains = [G.gain(*E.camera_pose(centre, dist, *r)) for r in rest]
            v = rest[int(np.argmax(gains))]
        out["nbv_box"] = dict(views=seq, travel=travel(seq), evals=res, stop_k=stop_k)
    if "nbv_bitsi" not in STRATS:
        out["seconds"] = time.time() - t0
        return out

    # -------- BITSI NBV (ours)
    F, res, seq, stop_k = E.Fused(h, seed=1), [], [], None
    v = home
    for k in range(1, K_MAX + 1):
        cam, Rc, idx, meas = view(v)
        F.add(idx, meas, cam)
        seq.append(v)
        fidx, pts, n = F.cloud()
        col = C[fidx] if C is not None else None
        try:
            lab, root, hh = segment("v4", pts, n, col)
        except Exception:
            lab, root, hh = None, None, None
        if k in EVAL_K:
            evaluate(F, k, "nbv_bitsi", res, v4fit=(lab,) if lab is not None else None)
        rest = [r for r in reach if r not in seq]
        if root is None or not rest:
            v = rest[0] if rest else None
            if v is None:
                break
            continue
        pp, pn, obs = predicted_surface_safe(root, hh)
        unobs = obs == 0
        if stop_k is None and unobs.mean() < 0.05:
            stop_k = k
        gains = [E.bitsi_gain(pp, pn, unobs, *E.camera_pose(centre, dist, *r), pts, hh) for r in rest]
        v = rest[int(np.argmax(gains))]
    out["nbv_bitsi"] = dict(views=seq, travel=travel(seq), evals=res, stop_k=stop_k)

    # -------- every reachable view (upper bound)
    F, res = E.Fused(h, seed=1), []
    for v in reach:
        cam, Rc, idx, meas = view(v)
        F.add(idx, meas, cam)
    evaluate(F, len(reach), "all", res)
    out["all"] = dict(evals=res)
    out["seconds"] = time.time() - t0
    return out


def run_box_only(out, it, reach, robot, view, evaluate, travel, home, centre, dist, h, W, t0):
    G = E.BoxNBV(centre, (np.ptp(W, 0) ** 2).sum() ** 0.5)
    F, res, seq, stop_k = E.Fused(h, seed=1), [], [], None
    v = home
    for k in range(1, K_MAX + 1):
        cam, Rc, idx, meas = view(v)
        F.add(idx, meas, cam)
        G.integrate(cam, Rc, meas)
        seq.append(v)
        G.fit_box(F.cloud()[1], cams=[E.camera_pose(centre, dist, *u)[0] for u in seq])
        if stop_k is None and G.unknown_fraction() < G.stop:
            stop_k = k
        if k in EVAL_K:
            evaluate(F, k, "nbv_box", res)
        rest = [r for r in reach if r not in seq]
        if not rest:
            break
        gains = [G.gain(*E.camera_pose(centre, dist, *r)) for r in rest]
        v = rest[int(np.argmax(gains))]
    out["nbv_box"] = dict(views=seq, travel=travel(seq), evals=res, stop_k=stop_k)
    out["seconds"] = time.time() - t0
    return out


def predicted_surface_safe(root, h):
    try:
        return E.predicted_surface(root, h)
    except Exception:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "requests":
        reqs = []
        for it in objects():
            W, Nw, C, gt = load(it)
            centre, diag = E.object_frame(W)
            reqs.append(dict(key=it["key"], centre=centre.tolist(), dist=E.view_distance(diag)))
        json.dump(reqs, open(sys.argv[2], "w"))
        print(len(reqs), "objects")
    else:
        reach_all = json.load(open(sys.argv[2]))
        out_jsonl = sys.argv[3]
        robot = sys.argv[4] if len(sys.argv) > 4 else "panda"
        done = set()
        if os.path.exists(out_jsonl):
            done = {json.loads(l)["key"] for l in open(out_jsonl) if l.strip()}
        todo = [(it, reach_all[it["key"]], robot) for it in objects() if it["key"] not in done and it["key"] in reach_all]
        print(len(done), "done,", len(todo), "to run", flush=True)
        with Pool(int(os.environ.get("WORKERS", 14))) as p, open(out_jsonl, "a") as f:
            for i, r in enumerate(p.imap_unordered(run_object, todo, chunksize=1)):
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(i, "/", len(todo), r.get("seconds", r.get("error")), flush=True)
        print("finished", flush=True)
