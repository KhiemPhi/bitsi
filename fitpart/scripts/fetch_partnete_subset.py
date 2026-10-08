"""Extract pc.ply + label.npy for the first K objects per category of PartSLIP's test.zip
by random-access reads over HTTP (no full 9.4 GB download).

usage: python fetch_partnete_subset.py K [dest=~/data/PartNetE/data] [Cat1,Cat2,...]
"""
import collections, os, sys, zipfile
from huggingface_hub import HfFileSystem

K = int(sys.argv[1])
dest = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/data/PartNetE/data")
CATS = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
fs = HfFileSystem()
z = zipfile.ZipFile(fs.open("datasets/minghua/PartSLIP/data/test.zip", "rb", block_size=2 ** 22))
objs = collections.defaultdict(list)
for n in z.namelist():
    parts = n.strip("/").split("/")
    if len(parts) == 3 and n.endswith("/"):
        objs[parts[1]].append(parts[2])
for cat in sorted(objs):
    if CATS is not None and cat not in CATS:
        continue
    print(cat, len(objs[cat]), "test objects", flush=True)
    for oid in sorted(objs[cat], key=int)[:K]:
        for fn in ("pc.ply", "label.npy"):
            member = f"test/{cat}/{oid}/{fn}"
            out = os.path.join(dest, member)
            if os.path.exists(out):
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(member) as src, open(out + ".part", "wb") as dst:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    dst.write(b)
            os.replace(out + ".part", out)
        print(cat, oid, flush=True)
