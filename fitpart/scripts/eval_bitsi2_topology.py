"""Do BITSI v2 attachment counts identify handles? For every child primitive of the root, take
the GT semantic part covering most of its points (>= 50%, else 'body/unlabelled') and tabulate
against the attachment count (2+ = loop). Also: share of GT handles recovered as a child.

usage: python eval_bitsi2_topology.py Cat1,Cat2,... [per_cat=20] [start=0]
"""
import json, os, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from fitpart.bitsi2 import BITSI2
from fitpart.data import PARTNETE_ROOT, load_partnete

META = json.load(open(os.path.join(PARTNETE_ROOT, "PartNetE_meta.json")))


def run(d):
    cat = os.path.basename(os.path.dirname(d))
    P, N, C, sem, ins = load_partnete(d)
    names = META[cat]
    lab, root = BITSI2().fit(P, N, C)
    rows = []
    for ch in root.children:
        s = sem[ch.idx]
        v, cnt = np.unique(s, return_counts=True)
        top = v[np.argmax(cnt)]
        name = names[top] if top >= 0 and cnt.max() >= 0.5 * len(s) else "body/unlabelled"
        rows.append((cat, name, ch.n_attach))
    # GT handle instances recovered: some child with IoU >= 0.5
    rec = []
    if "handle" in names:
        hid = names.index("handle")
        inst = ins if ins is not None else sem
        for i in np.unique(inst[sem == hid]):
            g = (inst == i) & (sem == hid)
            best = max([(np.isin(np.arange(len(P)), ch.idx) & g).sum() / (np.isin(np.arange(len(P)), ch.idx) | g).sum()
                        for ch in root.children] + [0.0])
            rec.append(best)
    return rows, rec


if __name__ == "__main__":
    cats = sys.argv[1].split(",")
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    base = os.path.join(PARTNETE_ROOT, "data", "test")
    items = []
    for c in cats:
        ds = sorted((x for x in os.listdir(os.path.join(base, c)) if os.path.exists(os.path.join(base, c, x, "label.npy"))), key=int)[start:start + per]
        items += [os.path.join(base, c, x) for x in ds]
    with Pool(12) as p:
        out = p.map(run, items, chunksize=1)
    tab = defaultdict(Counter)
    for rows, _ in out:
        for cat, name, na in rows:
            tab[name]["loop (2+ patches)" if na >= 2 else ("1 patch" if na == 1 else "0 patches")] += 1
    print("| child's GT part | loop (2+ patches) | 1 patch | 0 patches |")
    for name, c in sorted(tab.items(), key=lambda x: -sum(x[1].values())):
        print(f"| {name} | {c['loop (2+ patches)']} | {c['1 patch']} | {c['0 patches']} |")
    rec = np.array([x for _, r in out for x in r])
    if len(rec):
        print(f"GT handles recovered as a child (IoU >= 0.5): {np.sum(rec >= .5)}/{len(rec)}, mean best IoU {rec.mean():.2f}")
