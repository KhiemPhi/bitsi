"""BITSI segmentation of one object cloud, no ROS dependency. version selects the algorithm:
  v0  main.py's BITSI (BITC sweep along one bounding-box axis, Chebyshev inflections, IBR fusion)       [default]
  v3  fitpart BITSI3: primitive-relative BITC channels, Chebyshev pieces, MDL cuts
  v4  fitpart BITSI4(up): v3 + children by code length, gravity-axis candidate (robot setting, RESULTS.md s32)
  v5  fitpart BITSI5(up, partial-cloud candidate: axis_rule=residual, spine_coding=free, partial_cuts=True)
  v6  fitpart BITSI6(up): gravity root axis, round-or-box sections (experimental; only if fitpart/bitsi6.py is present)
v3+ need normals: estimated here (k=30 neighbours) and oriented away from the cloud centroid.

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


VERSIONS = ('v0', 'v3', 'v4', 'v5', 'v6')


def make_model(version, up):
    up = tuple(float(u) for u in up)
    if version == 'v3':
        from fitpart.bitsi3 import BITSI3; return BITSI3()
    if version == 'v4':
        from fitpart.bitsi4 import BITSI4; return BITSI4(up=up)
    if version == 'v5':
        from fitpart.bitsi5 import BITSI5; return BITSI5(axis_rule='residual', spine_coding='free', up=up, partial_cuts=True)
    if version == 'v6':
        try: from fitpart.bitsi6 import BITSI6
        except ImportError as e: raise RuntimeError('BITSI v6 is not in this build (fitpart/bitsi6.py missing)') from e
        return BITSI6(up=up)
    raise ValueError(f'unknown BITSI version {version!r}; choose from {VERSIONS}')


def segment_dict(name, p, ibr=None):
    s = dict(name=name, n_points=int(len(p)), centroid=p.mean(0).tolist(), ibr_ratio=None if ibr is None else float(ibr), points=p)
    if len(p) >= 4:
        obb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)).get_oriented_bounding_box()
        s.update(obb_center=np.asarray(obb.center).tolist(), obb_extent=np.asarray(obb.extent).tolist(), obb_R=np.asarray(obb.R).tolist())
    return s


def leaves(node):
    return [node] if not node.children else [l for c in node.children for l in leaves(c)]


def run_bitsi(points, version='v0', up=(0., 0., 1.), gripper_width=0.13, gripper_height=0.07, epsilon=5e-3, strength_threshold=0.01,
              ibr_tolerance=0.07, thickness_scale=0.03, voxel=0.004):
    pcd = prepare(points, voxel=voxel)
    if version != 'v0':
        model = make_model(version, up)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30)); P = np.asarray(pcd.points); N = np.asarray(pcd.normals).copy()
        N[((P - P.mean(0)) * N).sum(1) < 0] *= -1
        lab, _ = model.fit(P, N); lab = np.asarray(lab)
        ids = [u for u in np.unique(lab)]; ids.sort(key=lambda u: -(lab == u).sum())
        segs = [segment_dict(f'part_{j}', P[lab == u]) for j, u in enumerate(ids)]
        return dict(version=version, n_input=int(len(points)), n_filtered=int(len(P)), thickness=None, slice_axis=None, segments=segs)
    cloud_object = build_cloud_object(pcd, gripper_width, gripper_height)
    thickness = bitsi.auto_thickness(cloud_object, scale=thickness_scale)
    bx, by, bz, slice_idx, px, py, pz = bitsi_metric(cloud_object, epsilon=epsilon, thickness=thickness)
    root = bitsi.build_segmentation_tree([px, py, pz][slice_idx], bx, by, bz, slice_idx, strength_threshold)
    root = bitsi.rename_segments(bitsi.fuse_consecutive_segments(root, ibr_tolerance=ibr_tolerance))
    segs = []
    for n in leaves(root):
        p = np.asarray(n.points, dtype=np.float64).reshape(-1, 3)
        if len(p): segs.append(segment_dict(n.name, p, n.ibr_ratio))
    return dict(version='v0', n_input=int(len(points)), n_filtered=int(len(pcd.points)), thickness=float(thickness), slice_axis=int(slice_idx), segments=segs)
