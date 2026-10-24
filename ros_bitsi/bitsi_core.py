"""BITSI Chebyshev segmentation of one object cloud, no ROS dependency.

run_bitsi(points (N,3) in the robot frame) -> dict with one entry per BITSI segment (points stay in the input frame;
main.py maps slice points back to original_world_points). Mirrors main.py's pipeline: build_cloud_object -> auto_thickness ->
bitsi_metric -> build_segmentation_tree -> fuse_consecutive_segments -> rename_segments.
"""
import numpy as np
import open3d as o3d

import main as bitsi   # ~/bitsi/main.py (module-level imports only; its CLI runs under __main__)
from bitsi_slicer.bitsi_slicer import build_cloud_object, bitsi_metric


def prepare(points, voxel=0.004, nb_neighbors=20, std_ratio=2.0, min_points=50):
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(1)]
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
    if voxel > 0: pcd = pcd.voxel_down_sample(voxel)
    if len(pcd.points) > nb_neighbors: pcd, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    if len(pcd.points) < min_points: raise ValueError(f'only {len(pcd.points)} points after filtering (< {min_points})')
    return pcd


def leaves(node):
    return [node] if not node.children else [l for c in node.children for l in leaves(c)]


def run_bitsi(points, gripper_width=0.13, gripper_height=0.07, epsilon=5e-3, strength_threshold=0.01, ibr_tolerance=0.07,
              thickness_scale=0.03, voxel=0.004):
    pcd = prepare(points, voxel=voxel)
    cloud_object = build_cloud_object(pcd, gripper_width, gripper_height)
    thickness = bitsi.auto_thickness(cloud_object, scale=thickness_scale)
    bx, by, bz, slice_idx, px, py, pz = bitsi_metric(cloud_object, epsilon=epsilon, thickness=thickness)
    root = bitsi.build_segmentation_tree([px, py, pz][slice_idx], bx, by, bz, slice_idx, strength_threshold)
    root = bitsi.rename_segments(bitsi.fuse_consecutive_segments(root, ibr_tolerance=ibr_tolerance))
    segs = []
    for n in leaves(root):
        p = np.asarray(n.points, dtype=np.float64).reshape(-1, 3)
        if len(p) == 0: continue
        s = dict(name=n.name, n_points=int(len(p)), centroid=p.mean(0).tolist(), ibr_ratio=float(n.ibr_ratio), points=p)
        if len(p) >= 4:
            obb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)).get_oriented_bounding_box()
            s.update(obb_center=np.asarray(obb.center).tolist(), obb_extent=np.asarray(obb.extent).tolist(), obb_R=np.asarray(obb.R).tolist())
        segs.append(s)
    return dict(n_input=int(len(points)), n_filtered=int(len(pcd.points)), thickness=float(thickness), slice_axis=int(slice_idx), segments=segs)
