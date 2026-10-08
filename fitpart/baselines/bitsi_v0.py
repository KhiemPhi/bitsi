"""Adapter: the existing BITSI pipeline (main.py) as a baseline with per-point labels.

Same steps as batch_process_all_classes.process_single_object: dominant-axis BITSI
profile, BIC-chosen Chebyshev fit, inflections above strength_threshold, then IBR
fusion of consecutive segments. Leaves are mapped back to point indices exactly.
"""
import os
import sys

import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def segment(points, normals, gripper_width=0.13, gripper_height=0.07, epsilon=5e-3,
            strength_threshold=0.01, ibr_tolerance=0.07):
    import matplotlib
    matplotlib.use("Agg")
    from main import (auto_thickness, bitsi_metric, build_cloud_object, build_segmentation_tree,
                      fuse_consecutive_segments)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    cloud = build_cloud_object(pcd, gripper_width, gripper_height)
    thickness = auto_thickness(cloud, scale=0.03)
    bx, by, bz, slice_idx, px, py, pz = bitsi_metric(cloud, epsilon=epsilon, thickness=thickness)
    per_slice = [px, py, pz][slice_idx]
    root = build_segmentation_tree(per_slice, bx, by, bz, slice_idx, strength_threshold)
    root = fuse_consecutive_segments(root, ibr_tolerance=ibr_tolerance)
    leaves = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.children:
            stack.extend(n.children)
        else:
            leaves.append(n)
    # map leaf points back to indices (exact coordinates are preserved)
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    lab = np.full(len(points), -1)
    for k, leaf in enumerate(leaves):
        lp = np.asarray(leaf.points)
        if len(lp) == 0:
            continue
        _, idx = tree.query(lp)
        lab[idx] = k
    # points no slice picked up (outside slab bounds) join their nearest labeled point
    miss = lab < 0
    if miss.any() and (~miss).any():
        _, j = cKDTree(points[~miss]).query(points[miss])
        lab[miss] = lab[~miss][j]
    return lab
