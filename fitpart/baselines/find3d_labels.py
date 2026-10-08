"""Find3D (Ma et al., 2025) per-point part labels for every object, cached for eval_all.py.

Follows the authors' benchmark code (model/evaluation): their preprocessing (5000-point grid
sample), queries "<part> of a <category>" from the GT part vocabulary (PartNetE_meta.json, or
Find3D's own ShapeNetPart names), a leading "no label" logit of 0, softmax at the learned
temperature, and nearest-neighbour upsampling to every evaluated point. Canonical orientation
(no random rotation). ShapeNetPart has no colour; points get grey (feature 0), as an
uncoloured input to the model.

Find3D sees the GT part names of each category; BITSI sees no labels.
usage (find3d env):  PYTHONPATH=~/bitsi/ext/Find3D python find3d_labels.py DATASET
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.expanduser("~/bitsi/ext/Find3D"))
import numpy as np
import torch

from fitpart.data import PARTNETE_ROOT, load_partnete, load_shapenetpart
from fitpart.scripts.eval_all import items

CACHE = os.path.expanduser("~/.cache/fitpart_baselines/find3d")
SNP_PARTS = {'airplane': ['body', 'wing', 'tail', 'engine or frame'], 'bag': ['handle', 'body'],
             'cap': ['panels or crown', 'visor or peak'], 'car': ['roof', 'hood', 'wheel or tire', 'body'],
             'chair': ['back', 'seat pad', 'leg', 'armrest'], 'earphone': ['earcup', 'headband', 'data wire'],
             'guitar': ['head or tuners', 'neck', 'body'], 'knife': ['blade', 'handle'],
             'lamp': ['base', 'lampshade', 'fixing bracket', 'stem'], 'laptop': ['keyboard', 'screen or monitor'],
             'motorbike': ['gas tank', 'seat', 'wheel', 'handles or handlebars', 'light', 'engine or frame'],
             'mug': ['handle', 'cup'], 'pistol': ['barrel', 'handle', 'trigger and guard'],
             'rocket': ['body', 'fin', 'nose cone'], 'skateboard': ['wheel', 'deck', 'belt for foot'],
             'table': ['desktop', 'leg or support', 'drawer']}


def main(dataset):
    import functools
    import model.backbone.pt3.model as pt3
    # no flash-attn wheel for this machine: PTv3's own non-flash attention path, same weights
    pt3.PointTransformerV3.__init__ = functools.partialmethod(pt3.PointTransformerV3.__init__, enable_flash=False)
    from model.evaluation.utils import load_model, preprocess_pcd
    from transformers import AutoModel, AutoTokenizer
    os.makedirs(os.path.join(CACHE, dataset), exist_ok=True)
    model = load_model()
    temp = float(np.exp(model.ln_logit_scale.item()))
    siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224").cuda().eval()
    tok = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
    meta = json.load(open(os.path.join(PARTNETE_ROOT, "PartNetE_meta.json")))
    text_cache = {}
    its = items(dataset)
    for i, it in enumerate(its):
        out = os.path.join(CACHE, dataset, it["key"].replace("/", "__") + ".npy")
        if os.path.exists(out):
            continue
        if "dir" in it:
            P, N, C, _, _ = load_partnete(it["dir"])
            names = [f"{p} of a {it['cat']}" for p in meta[it["cat"]]]
        else:
            P, N, _ = load_shapenetpart(it["sid"], it["oid"])
            C = None
            cat = it["cat"].lower()
            names = [f"{p} of a {cat}" for p in SNP_PARTS[cat]]
        if C is None:
            C = np.full_like(P, 0.5)
        key = tuple(names)
        if key not in text_cache:
            inp = tok(names, padding="max_length", return_tensors="pt").to("cuda")
            with torch.no_grad():
                t = siglip.get_text_features(**inp)
            t = t if torch.is_tensor(t) else t.pooler_output  # newer transformers wrap it
            text_cache[key] = t / (t.norm(dim=-1, keepdim=True) + 1e-12)
        T = text_cache[key]
        f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32).cuda()
        data = preprocess_pcd(f(P), f(C).clamp(0, 1), f(N))
        with torch.no_grad():
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            feat = model(x=data)
            logits = torch.cat([torch.zeros(len(feat), 1, device="cuda"), feat @ T.T], 1)
            prob = torch.softmax(logits * temp, 1)
            nn = torch.cdist(data["xyz_full"], data["coord"]).argmin(1)
            lab = prob[nn].argmax(1).cpu().numpy().astype(np.int16)  # 0 = no label
        np.save(out, lab)
        if i % 100 == 0:
            print(i, "/", len(its), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
