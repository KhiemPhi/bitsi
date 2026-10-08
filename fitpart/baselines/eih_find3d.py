"""Find3D labels for the fused eye-in-hand clouds saved by eval_eyeinhand.py (K in SAVE_K).

Same settings as find3d_labels.py (GT part names of the category as queries, colour of each fused
point's surface sample). Labels go next to each cloud as <cloud>.find3d.npy.
usage (find3d env): PYTHONPATH=~/bitsi/ext/Find3D:~/bitsi/ext/shims python eih_find3d.py ROBOT
"""
import functools, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.expanduser("~/bitsi/ext/Find3D"))
import numpy as np
import torch

from fitpart.data import PARTNETE_ROOT, load_partnete

CLOUDS = os.path.expanduser("~/.cache/fitpart_eih")


def main(robot):
    import model.backbone.pt3.model as pt3
    pt3.PointTransformerV3.__init__ = functools.partialmethod(pt3.PointTransformerV3.__init__, enable_flash=False)
    from model.evaluation.utils import load_model, preprocess_pcd
    from transformers import AutoModel, AutoTokenizer
    model = load_model()
    temp = float(np.exp(model.ln_logit_scale.item()))
    siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224").cuda().eval()
    tok = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
    meta = json.load(open(os.path.join(PARTNETE_ROOT, "PartNetE_meta.json")))
    files = sorted(glob.glob(os.path.join(CLOUDS, robot, "*", "*.npz")))
    colours, texts = {}, {}
    for i, f in enumerate(files):
        out = f[:-4] + ".find3d.npy"
        if os.path.exists(out):
            continue
        key = os.path.basename(f).split("__K")[0].replace("__", "/")
        cat = key.split("/")[0]
        if key not in colours:
            _, _, C, _, _ = load_partnete(os.path.join(PARTNETE_ROOT, "data", "test", key))
            colours = {key: C}
        z = np.load(f)
        C = colours[key][z["idx"]] if colours[key] is not None else np.full((len(z["idx"]), 3), 0.5)
        if cat not in texts:
            names = [f"{p} of a {cat}" for p in meta[cat]]
            inp = tok(names, padding="max_length", return_tensors="pt").to("cuda")
            with torch.no_grad():
                t = siglip.get_text_features(**inp)
            t = t if torch.is_tensor(t) else t.pooler_output
            texts[cat] = t / (t.norm(dim=-1, keepdim=True) + 1e-12)
        g = lambda a: torch.tensor(np.asarray(a, np.float32)).cuda()
        data = preprocess_pcd(g(z["pts"]), g(C).clamp(0, 1), g(z["n"]))
        with torch.no_grad():
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            feat = model(x=data)
            logits = torch.cat([torch.zeros(len(feat), 1, device="cuda"), feat @ texts[cat].T], 1)
            prob = torch.softmax(logits * temp, 1)
            nn = torch.cdist(data["xyz_full"], data["coord"]).argmin(1)
            np.save(out, prob[nn].argmax(1).cpu().numpy().astype(np.int16))
        if i % 200 == 0:
            print(i, "/", len(files), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
