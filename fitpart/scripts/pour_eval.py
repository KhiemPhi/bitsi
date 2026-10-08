"""'Pour from the mug' benchmark (env: graspsim): every method's region -> shared antipodal sampler
-> order (random, or by the SOCP task metric tau for the pour screw) -> execute up to 10 grasps.
Reports first-grasp success and successes out of 10 attempts (missing grasps count as failures).
Writes results JSON; with --video OID, records each method's first grasp on that mug."""
import glob, json, os, sys, time
sys.path.insert(0, '/home/khiem/bitsi')
import numpy as np
from fitpart.grasping import sample_grasps
from fitpart.sim.mujoco_grasp import GraspSim, convex_parts
from fitpart.sim.pour_task import pour_screw, try_pour, upright_pose
from fitpart.sim.screw_metric import screw_tau

D = sys.argv[1]
MASS = 0.35
N_TRY = 10


def candidates(sim, Pw, Nw, mask, pool=80, per_segment=None, seed=0):
    """Collision-free, floor-clear antipodal grasps on a region (or per segment, pooled)."""
    out = []
    regions = [mask] if per_segment is None else [per_segment == l for l in np.unique(per_segment)]
    budget = pool if per_segment is None else max(20, pool // len(regions))
    for m in regions:
        if m.sum() < 10:
            continue
        gs = sample_grasps(Pw[m], Nw[m], Pw, n_contacts=max(400, 8 * budget), max_grasps=budget, seed=seed)
        out += [g for g in gs if sim.clearance_ok(g[0], g[1], g[2] / np.linalg.norm(g[2]))]
    return out


def evaluate_object(f, record=None):
    z = np.load(f, allow_pickle=True)
    P, N, handle = z['P'].astype(float), z['N'].astype(float), z['handle']
    sel = np.load(f[:-4] + '.sel.npz')
    bitsi = np.load(f[:-4] + '.bitsi.npy')
    sim = GraspSim(convex_parts(str(z['mesh'])), mass=MASS)
    R0, (t, R) = upright_pose(sim, P, handle)
    Pw, Nw = P @ R.T + t, N @ R.T
    axis, point = pour_screw(Pw, handle)
    com = Pw.mean(0)
    whole = np.ones(len(P), bool)
    methods = {
        'Whole object, random': (whole, None, 'random'),
        'Whole object, ranked by tau': (whole, None, 'tau'),
        'BITSI segments + tau (ITSI-style)': (whole, bitsi, 'tau'),
        'SAM 3 lifted "handle", random': (sel['sam3_direct'], None, 'random'),
        'FitPart + SAM 3 "handle", random': (sel['fitpart'], None, 'random'),
        'FitPart + SAM 3 "handle", ranked by tau': (sel['fitpart'], None, 'tau'),
        'GT handle, random (oracle)': (handle, None, 'random'),
        'GT handle, ranked by tau (oracle)': (handle, None, 'tau'),
        'GT body, random': (~handle, None, 'random'),
    }
    res = {}
    for name, (mask, seg, order) in methods.items():
        if record is not None and name not in record['methods']:
            continue
        gs = candidates(sim, Pw, Nw, mask, per_segment=seg)
        if order == 'tau' and gs:
            taus = [screw_tau(g[0], g[1], com, MASS, axis, point) for g in gs]
            gs = [g for _, g in sorted(zip(np.nan_to_num(taus, nan=-1e9), gs), key=lambda x: -x[0])]
        gs = gs[:N_TRY]
        outcomes = []
        for k, (c1, c2, a) in enumerate(gs):
            sim.reset_and_settle(R0)
            rec = record['recorders'][name] if (record is not None and k == 0) else None
            ok, why = try_pour(sim, c1, c2, a, axis, point, recorder=rec)
            outcomes.append(ok)
            if record is not None:
                break
        res[name] = dict(first=bool(outcomes[0]) if outcomes else False, succ=int(sum(outcomes)), n=len(gs))
    return res


if __name__ == '__main__':
    files = sorted(glob.glob(f'{D}/obj_*.npz'))
    files = [f for f in files if not f.endswith('.sel.npz')]
    if '--video' in sys.argv:
        import imageio, mujoco
        from PIL import Image, ImageDraw, ImageFont
        try:
            FONT = ImageFont.load_default(size=20)
        except TypeError:
            FONT = ImageFont.load_default()
        oid = sys.argv[sys.argv.index('--video') + 1]
        names = sys.argv[sys.argv.index('--video') + 2].split('|')
        f = f'{D}/{oid}.npz'
        frames = {}
        recs = {}

        def make_rec(name):
            state = {'i': 0, 'r': None, 'cam': None}
            frames[name] = []

            def rec(d):
                state['i'] += 1
                if state['i'] % 17:
                    return
                if state['r'] is None:
                    state['r'] = mujoco.Renderer(d.model if hasattr(d, 'model') else MODEL[0], 360, 480)
                cam = mujoco.MjvCamera()
                cam.lookat[:] = [0, 0, 0.20]
                cam.distance, cam.azimuth, cam.elevation = 1.0, 135, -12
                state['r'].update_scene(d, cam)
                frames[name].append(state['r'].render().copy())
            return rec
        # mujoco.Renderer needs the model; patch via a global set when the sim is built
        MODEL = [None]
        import fitpart.sim.mujoco_grasp as mg
        orig_init = mg.GraspSim.__init__

        def init(self, *a, **k):
            orig_init(self, *a, **k)
            MODEL[0] = self.model
        mg.GraspSim.__init__ = init
        for n in names:
            recs[n] = make_rec(n)
        res = evaluate_object(f, record={'methods': names, 'recorders': recs})
        tiles = []
        T = max(len(v) for v in frames.values())
        for n in names:
            fr = frames[n] or [np.zeros((360, 480, 3), np.uint8)]
            fr = fr + [fr[-1]] * (T - len(fr))
            out = []
            for im in fr:
                img = Image.fromarray(im)
                dr = ImageDraw.Draw(img)
                ok = res.get(n, {}).get('first', False)
                dr.rectangle([0, 0, 480, 56], fill=(255, 255, 255))
                dr.text((8, 4), n, fill=(20, 20, 20), font=FONT)
                dr.text((8, 30), 'SUCCESS' if ok else 'FAIL', fill=(0, 140, 0) if ok else (200, 30, 30), font=FONT)
                out.append(np.asarray(img))
            tiles.append(out)
        cols = 2
        rows = (len(tiles) + 1) // 2
        blank = [np.full((360, 480, 3), 255, np.uint8)] * T
        tiles += [blank] * (rows * cols - len(tiles))
        video = [np.vstack([np.hstack([tiles[r * cols + c][t] for c in range(cols)]) for r in range(rows)]) for t in range(T)]
        path = sys.argv[sys.argv.index('--video') + 3]
        imageio.mimsave(path, video, fps=30, macro_block_size=8)
        print('video', path, len(video), 'frames', {k: v['first'] for k, v in res.items()})
        sys.exit(0)
    # spawn, not fork: CoACD / the SOCP solver start threads that deadlock in forked children
    from multiprocessing import get_context
    t = time.time()
    allres = []
    with get_context("spawn").Pool(12) as p:
        for f, r in zip(files, p.imap(evaluate_object, files)):
            allres.append(r)
            print(f"done {os.path.basename(f)} {time.time()-t:.0f}s", flush=True)
    json.dump({os.path.basename(f): r for f, r in zip(files, allres)}, open(f'{D}/pour_results.json', 'w'), indent=1)
    print(f"{len(files)} mugs, {time.time()-t:.0f}s\n| Method | first-grasp success | successes / 10 attempts |\n|---|---|---|")
    for name in allres[0]:
        first = np.mean([r[name]['first'] for r in allres]); s = np.mean([r[name]['succ'] / N_TRY for r in allres])
        print(f"| {name} | {first:.2f} ({sum(r[name]['first'] for r in allres)}/{len(allres)}) | {s:.2f} |")
