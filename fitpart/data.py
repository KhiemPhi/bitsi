"""Dataset loaders: ShapeNetPart (local) and PartNet-Ensembled (PartSLIP test split)."""
import json
import os

import numpy as np

SHAPENETPART_ROOT = os.path.expanduser("~/data/ShapeNetPart")
PARTNETE_ROOT = os.path.expanduser("~/data/PartNetE")


def shapenetpart_categories(root=SHAPENETPART_ROOT):
    cats = {}
    with open(os.path.join(root, "synsetoffset2category.txt")) as f:
        for line in f:
            if line.strip():
                name, sid = line.split()
                cats[sid] = name
    return cats


def shapenetpart_split(split="test", root=SHAPENETPART_ROOT):
    """List of (synset_id, object_id) in the official shuffled split."""
    with open(os.path.join(root, "train_test_split", f"shuffled_{split}_file_list.json")) as f:
        items = json.load(f)
    out = []
    for it in items:  # local lists are "sid/oid.txt"; the official ones "shape_data/sid/oid"
        sid, oid = it.split("/")[-2:]
        out.append((sid, oid.removesuffix(".txt")))
    return out


def load_shapenetpart(sid, oid, root=SHAPENETPART_ROOT):
    """Returns points (N,3), normals (N,3), labels (N,) global part ids."""
    data = np.loadtxt(os.path.join(root, sid, f"{oid}.txt"))
    return data[:, :3], data[:, 3:6], data[:, 6].astype(int)


def partnete_meta(root=PARTNETE_ROOT):
    with open(os.path.join(root, "PartNetE_meta.json")) as f:
        return json.load(f)


def partnete_objects(split="test", root=PARTNETE_ROOT):
    """List of (category, object_dir) under the unzipped PartSLIP data."""
    base = os.path.join(root, "data", split)
    if not os.path.isdir(base):  # zip unpacked one level up
        base = os.path.join(root, split)
    out = []
    for cat in sorted(os.listdir(base)):
        cdir = os.path.join(base, cat)
        if not os.path.isdir(cdir):
            continue
        for oid in sorted(os.listdir(cdir)):
            if os.path.isdir(os.path.join(cdir, oid)):
                out.append((cat, os.path.join(cdir, oid)))
    return out


def load_partnete(obj_dir, max_points=20000, seed=0):
    """points (N,3), normals (N,3), colors (N,3) in [0,1], semantic labels (N,) with
    -1 = unlabeled, instance labels (N,) or None.

    Clouds are randomly subsampled to max_points (labels stay exact; evaluation is on
    the subsample). Normals are estimated when the PLY has none.
    """
    import open3d as o3d
    pc = o3d.io.read_point_cloud(os.path.join(obj_dir, "pc.ply"))
    keep = None
    if max_points and len(pc.points) > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(len(pc.points), max_points, replace=False))
        pc = pc.select_by_index(keep)
    P = np.asarray(pc.points)
    C = np.asarray(pc.colors) if pc.has_colors() else None
    lab = np.load(os.path.join(obj_dir, "label.npy"), allow_pickle=True)
    if lab.dtype == object:
        lab = lab.item()
        sem = np.asarray(lab["semantic_seg"]).astype(int)
        ins = np.asarray(lab["instance_seg"]).astype(int) if "instance_seg" in lab else None
    else:
        sem, ins = lab.astype(int), None
    if keep is not None:
        sem = sem[keep]
        ins = ins[keep] if ins is not None else None
    if pc.has_normals():
        N = np.asarray(pc.normals)
    else:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        pc.orient_normals_towards_camera_location(P.mean(0) + np.array([0, 0, 10.0]))
        N = np.asarray(pc.normals)
    return P, N, C, sem, ins
