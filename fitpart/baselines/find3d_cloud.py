"""Find3D on one point cloud with given part queries (same settings as find3d_labels.py).

usage (find3d env): PYTHONPATH=~/bitsi/ext/Find3D:~/bitsi/ext/shims python find3d_cloud.py IN.npz OUT.npy "q1;q2;..."
  IN.npz: P (N,3), N (N,3), optional C (N,3) in [0,1]. OUT.npy: per point 0 = no label, j = query j.
"""
import functools, os, sys
sys.path.insert(0, os.path.expanduser("~/bitsi/ext/Find3D"))
import numpy as np
import torch


def main(inp, out, queries):
    import model.backbone.pt3.model as pt3
    pt3.PointTransformerV3.__init__ = functools.partialmethod(pt3.PointTransformerV3.__init__, enable_flash=False)
    from model.evaluation.utils import load_model, preprocess_pcd
    from transformers import AutoModel, AutoTokenizer
    z = np.load(inp)
    P, N = z["P"], z["N"]
    C = z["C"] if "C" in z.files else np.full_like(P, 0.5)
    model = load_model()
    temp = float(np.exp(model.ln_logit_scale.item()))
    siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224").cuda().eval()
    tok = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
    inp_t = tok(queries, padding="max_length", return_tensors="pt").to("cuda")
    with torch.no_grad():
        t = siglip.get_text_features(**inp_t)
    t = t if torch.is_tensor(t) else t.pooler_output
    T = t / (t.norm(dim=-1, keepdim=True) + 1e-12)
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
        lab = prob[nn].argmax(1).cpu().numpy().astype(np.int16)
    np.save(out, lab)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(";"))
