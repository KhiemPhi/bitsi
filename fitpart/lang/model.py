"""Command -> task screw slots.

  * task and direction are learned jointly as one label "task|+1" / "task|-1" (19 classes):
      tfidf   word 1-2-grams + char 2-4-grams, logistic regression (baseline)
      siglip  frozen SigLIP-base text encoder (pre-trained, cached) + a 2-layer MLP head (trained here)
  * magnitude ("by 45 degrees", "10 cm") is parsed by a rule, not learned; the default per task is used
    when the command has none
  * the grasp part comes from the task (lang/data.py OBJECTS), unless the command names a part word
The screw itself (axis, point) comes from BITSI on the robot's point cloud (tasks/screws.py);
this model only says WHICH screw, in which direction, and how far.
usage: python -m fitpart.lang.model OUT_DIR
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.lang.data import HANDWRITTEN, OBJECTS, TASKS, make

DEFAULT_MAG = dict(pour=90, lift=15, pivot=50, turn=180, hinge=70, slide=12, press=0.7, swing=40, scoop=35,
                   stir=120, cut=10, squeeze=20, handover=None)
PART_WORDS = ["handle", "cap", "lid", "knob", "button", "arm", "blade", "body", "cover", "seat"]


def parse_magnitude(text):
    t = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(degrees|degree|deg|°)", t)
    if m:
        return float(m.group(1)), "deg"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(centimeters|centimetres|cm)", t)
    if m:
        return float(m.group(1)), "cm"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(millimeters|mm)", t)
    if m:
        return float(m.group(1)) / 10, "cm"
    if "halfway" in t or "half way" in t:
        return 0.5, "frac"
    return None, None


def label(r):
    return f"{r['task']}|{'+1' if r['direction'] > 0 else '-1'}"


class TfidfModel:
    def fit(self, texts, y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion
        self.vec = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                                 ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True))])
        X = self.vec.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=3000, C=10.0).fit(X, y)
        return self

    def predict(self, texts):
        return list(self.clf.predict(self.vec.transform(texts)))


class SiglipModel:
    """Frozen SigLIP text encoder + MLP head."""
    name = "google/siglip-base-patch16-224"

    def __init__(self, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.enc = AutoModel.from_pretrained(self.name).to(self.device).eval()

    def embed(self, texts, bs=256):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                inp = self.tok([t.lower() for t in texts[i:i + bs]], padding="max_length", truncation=True,
                               return_tensors="pt").to(self.device)
                f = self.enc.get_text_features(**inp)
                f = f if torch.is_tensor(f) else f.pooler_output
                out.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        return torch.cat(out)

    def fit(self, texts, y, epochs=300, seed=0):
        torch = self.torch
        torch.manual_seed(seed)
        self.classes = sorted(set(y))
        X = self.embed(texts)
        t = torch.tensor([self.classes.index(v) for v in y])
        self.head = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                        torch.nn.Linear(256, len(self.classes)))
        opt = torch.optim.AdamW(self.head.parameters(), lr=2e-3, weight_decay=1e-3)
        for _ in range(epochs):
            self.head.train()
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.head(X), t)
            loss.backward()
            opt.step()
        self.head.eval()
        return self

    def predict(self, texts):
        with self.torch.no_grad():
            return [self.classes[i] for i in self.head(self.embed(texts)).argmax(1).tolist()]


def to_screw_spec(text, lab):
    """Turn a predicted label + the text into the screw slots the robot needs."""
    task, d = lab.split("|")
    mag, unit = parse_magnitude(text)
    part = next((w for w in PART_WORDS if re.search(rf"\b{w}\b", text.lower())), OBJECTS[task][2])
    if mag is None:
        mag, unit = DEFAULT_MAG[task], ("cm" if task in ("lift", "slide", "press", "cut") else "deg")
    elif unit == "frac":
        mag, unit = DEFAULT_MAG[task] * mag, ("cm" if task in ("lift", "slide", "press", "cut") else "deg")
    return dict(task=task, direction=int(d), magnitude=mag, unit=unit, part=part)


def evaluate(model, rows):
    pred = model.predict([r["text"] for r in rows])
    task_acc = np.mean([p.split("|")[0] == r["task"] for p, r in zip(pred, rows)])
    full_acc = np.mean([p == label(r) for p, r in zip(pred, rows)])
    return float(task_acc), float(full_acc), pred


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train = make("train", 300, seed=0, objects="train")
    sets = {
        "same templates, new samples": make("train", 60, seed=1, objects="train"),
        "held-out templates, train objects": make("test", 60, seed=2, objects="train"),
        "held-out templates, held-out objects": make("test", 60, seed=3, objects="test"),
        "hand-written (40)": [dict(text=t, task=k, direction=d) for t, k, d in HANDWRITTEN],
    }
    mags = make("test", 80, seed=4, objects="test")
    mag_rows = [r for r in mags if r["magnitude"] is not None]
    mag_acc = np.mean([parse_magnitude(r["text"])[0] == r["magnitude"] for r in mag_rows])
    X, y = [r["text"] for r in train], [label(r) for r in train]
    report = dict(n_train=len(train), magnitude_parse_acc=float(mag_acc), n_magnitude=len(mag_rows), models={})
    models = {"tfidf + logistic regression": TfidfModel().fit(X, y), "siglip (frozen) + MLP": SiglipModel().fit(X, y)}
    for nm, mdl in models.items():
        report["models"][nm] = {}
        for sn, rows in sets.items():
            ta, fa, pred = evaluate(mdl, rows)
            report["models"][nm][sn] = dict(task_acc=ta, task_and_direction_acc=fa, n=len(rows))
            if sn.startswith("hand"):
                report["models"][nm]["handwritten_errors"] = [(r["text"], p, label(r)) for p, r in zip(pred, rows) if p != label(r)]
    json.dump(report, open(f"{out_dir}/lang_report.json", "w"), indent=1)
    import pickle
    sm = models["siglip (frozen) + MLP"]
    pickle.dump(dict(classes=sm.classes, head=sm.head.state_dict()), open(f"{out_dir}/siglip_head.pkl", "wb"))
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=1))
    for nm, r in report["models"].items():
        print("\n" + nm)
        for sn, v in r.items():
            if sn != "handwritten_errors":
                print(f"  {sn:42s} task {v['task_acc']:.3f}  task+direction {v['task_and_direction_acc']:.3f}  (n={v['n']})")
        print("  hand-written errors:", r["handwritten_errors"])


if __name__ == "__main__":
    main(sys.argv[1])
