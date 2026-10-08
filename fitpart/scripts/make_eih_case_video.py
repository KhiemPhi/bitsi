"""One video per eye-in-hand pour case, and a compilation (eih_pour.py --video --dump outputs).

Per case: a title card, the look-around (arm driving through the BITSI next-best views; insets: the
fused cloud with predicted-but-unseen surface in red, and the live wrist depth image), then a 2 x 2
grid: BITSI v4 / Find3D / whole-object grasp-and-pour clips and a still of what the robot saw
(view map, all wrist depth images, fused cloud with the GT handle; handle surface never seen is faint).
usage: python make_eih_case_video.py DUMP_DIR VIDEO_DIR OUT_DIR cat/oid [cat/oid ...]
"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from fitpart.scripts.fig_eih_cases import cloud_ax, depth_mosaic, outcome, view_map

W_OUT, H_OUT = 1280, 1104
HERE = os.path.dirname(os.path.abspath(__file__))


def font(n):
    try:
        return ImageFont.load_default(size=n)
    except TypeError:
        return ImageFont.load_default()


def seen_still(d, path):
    W, gt = d["W"], d["gt_handle"]
    Wf, hf = d["W_full"], d["handle_full"]
    seen = cKDTree(W).query(Wf[hf])[0] < 0.004
    hdir = Wf[hf].mean(0) - Wf.mean(0)
    view = (22, np.degrees(np.arctan2(hdir[1], hdir[0])) - 55)
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    gs = fig.add_gridspec(2, 2, left=0.02, right=0.98, top=0.9, bottom=0.02, hspace=0.35, wspace=0.1)
    ax = fig.add_subplot(gs[0, 0], projection="polar")
    ax.set_theta_zero_location("E")
    view_map(ax, d)
    cloud_ax(fig.add_subplot(gs[0, 1], projection="3d"), W, np.where(gt, "#d62728", "#bdbcb6"), np.where(gt, 3.0, 1.2),
             f"fused cloud, handle (red) seen {seen.mean():.0%}", view, extra=Wf[hf][~seen])
    depth_mosaic(fig.add_subplot(gs[1, :]), d)
    fig.savefig(path)
    plt.close(fig)


def fit_frame(fr, header):
    im = Image.fromarray(fr)
    s = min(W_OUT / im.width, (H_OUT - 60) / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)))
    canvas = Image.new("RGB", (W_OUT, H_OUT), "white")
    canvas.paste(im, ((W_OUT - im.width) // 2, 60))
    ImageDraw.Draw(canvas).text((16, 16), header, fill=(31, 31, 30), font=font(26))
    return np.asarray(canvas)


def title_card(lines, n=60):
    canvas = Image.new("RGB", (W_OUT, H_OUT), "white")
    d = ImageDraw.Draw(canvas)
    y = H_OUT // 2 - 30 * len(lines)
    for i, t in enumerate(lines):
        d.text((80, y + 60 * i), t, fill=(31, 31, 30), font=font(34 if i == 0 else 26))
    fr = np.asarray(canvas)
    return [fr] * n


def case(dump, vdir, out_dir, cat, oid):
    base = f"{cat}_{oid}"
    d = dict(np.load(f"{dump}/{base}.npz"))
    res = json.load(open(f"{dump}/{base}.json"))
    still = f"{out_dir}/{base}_seen.png"
    seen_still(d, still)
    still_mp4 = f"{out_dir}/{base}_seen.mp4"
    imageio.mimsave(still_mp4, [np.asarray(Image.open(still).convert("RGB"))] * 2, fps=30, macro_block_size=1)
    grid = f"{out_dir}/{base}_grid.mp4"
    ours = "bitsi5g" if "bitsi5g" in res else "bitsi4"
    ours_lab = "BITSI v5g (ours)" if ours == "bitsi5g" else "BITSI v4 (ours)"
    specs = []
    for m, lab in ((ours, ours_lab), ("find3d", "Find3D"), ("whole", "Whole object")):
        txt, _ = outcome(res[m])
        specs.append(f"{vdir}/{base}_{m}.mp4:{lab}:{txt}")
    specs.append(f"{still_mp4}:What the robot saw:{len(d['views'])} views, handle seen {res['handle_seen_frac']:.0%}")
    subprocess.run([sys.executable, os.path.join(HERE, "compare_grid.py"), grid,
                    f"{cat.replace('_', ' ')} {oid}, robot's own partial views: grasp the handle and pour away from it"] + specs,
                   check=True, stdout=subprocess.DEVNULL)
    out = f"{out_dir}/{base}.mp4"
    w = imageio.get_writer(out, fps=30, macro_block_size=1)
    verdict = ", ".join(f"{n}: {outcome(res[m])[0].split(':')[0]}" for m, n in
                        ((ours, ours_lab.split(" (")[0]), ("find3d", "Find3D"), ("whole", "whole")))
    for fr in title_card([f"{cat.replace('_', ' ')} {oid}",
                          f"robot looks around with its wrist camera ({len(d['views'])} views), then each method grasps and pours",
                          f"handle surface seen: {res['handle_seen_frac']:.0%}",
                          verdict]):
        w.append_data(fr)
    for fr in imageio.get_reader(f"{vdir}/{base}_traverse.mp4"):
        w.append_data(fit_frame(fr, f"{cat.replace('_', ' ')} {oid}: looking around (BITSI next-best views)"))
    for fr in imageio.get_reader(grid):
        w.append_data(fit_frame(fr, ""))
    w.close()
    return out


def main():
    dump, vdir, out_dir, cases = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    comp_name = os.environ.get("COMP_NAME", "eih_cases_compilation.mp4")
    comp_note = os.environ.get("COMP_NOTE", "(not a random sample; overall rates: RESULTS.md section 22)")
    os.makedirs(out_dir, exist_ok=True)
    outs = []
    for co in cases:
        cat, oid = co.split("/")
        if not os.path.exists(f"{dump}/{cat}_{oid}.json"):
            print("missing", co)
            continue
        outs.append(case(dump, vdir, out_dir, cat, oid))
        print("saved", outs[-1], flush=True)
    if outs:
        comp = f"{out_dir}/{comp_name}"
        w = imageio.get_writer(comp, fps=30, macro_block_size=1)
        for fr in title_card(["Eye-in-hand grasp-and-pour: what the robot saw and what each method did",
                              f"{len(outs)} cases chosen to show successes and the different failure modes",
                              comp_note], n=90):
            w.append_data(fr)
        for o in outs:
            for fr in imageio.get_reader(o):
                w.append_data(fr)
        w.close()
        print("saved", comp)


if __name__ == "__main__":
    main()
