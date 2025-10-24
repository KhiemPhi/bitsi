import argparse
import open3d as o3d 
import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
from bitsi_slicer.bitsi_slicer import slice_object_bbox, bitsi_metric, build_cloud_object, get_overall_ibr_ratio
from numpy.polynomial import Chebyshev
from sklearn.cluster import KMeans
from numpy.polynomial import chebyshev as C
from sklearn.cluster import DBSCAN

import random



def fit_best_chebyshev(x, y, min_deg=2, max_deg=None, criterion="bic", strength_threshold=0.2):
    """
    Fit Chebyshev polynomials of increasing degree and pick the best one
    based on the chosen information criterion ('bic', 'aic', 'aicc', or 'adj_r2').

    Returns:
        best_coeffs, best_degree, best_yfit, best_metrics
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 points for fitting.")

    if max_deg is None:
        max_deg = min(8, n - 2)
    max_deg = max(max_deg, min_deg)

    # scale x → [-1, 1] for Chebyshev stability
    xmin, xmax = np.min(x), np.max(x)
    if xmax == xmin:
        x_scaled = np.zeros_like(x)
    else:
        x_scaled = 2 * (x - xmin) / (xmax - xmin) - 1.0

    def metrics(y_true, y_pred, k):
        rss = np.sum((y_true - y_pred) ** 2)
        tss = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
        r2 = 1 - rss / tss
        adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - k, 1)
        sigma2 = rss / max(n, 1)
        eps = 1e-18
        aic = n * np.log(sigma2 + eps) + 2 * k
        aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
        bic = n * np.log(sigma2 + eps) + k * np.log(max(n, 1))
        return dict(r2=r2, adj_r2=adj_r2, aic=aic, aicc=aicc, bic=bic, rss=rss)

    best = None
    all_models = []

    for deg in range(min_deg, max_deg + 1):
        try:
            ch = Chebyshev.fit(x_scaled, y, deg)
            y_fit = ch(x_scaled)
            m = metrics(y, y_fit, k=deg + 1)
            m["degree"] = deg
            m["model"] = ch
            m["y_fit"] = y_fit
            all_models.append(m)
        except np.linalg.LinAlgError:
            continue

    if not all_models:
        raise RuntimeError("Chebyshev fitting failed for all degrees.")

    if criterion.lower() in ("bic", "aic", "aicc"):
        key = criterion.lower()
        best = min(all_models, key=lambda m: m[key])
    elif criterion.lower() == "adj_r2":
        best = max(all_models, key=lambda m: m["adj_r2"])
    else:
        raise ValueError("criterion must be one of 'bic','aic','aicc','adj_r2'")

    best_ch = best["model"]
    coeffs = best_ch.coef
    degree = best["degree"]
    y_fit = best["y_fit"]
    
    
     # --- Compute inflection points directly on given x
    dcoeffs = C.chebder(coeffs)
    ddcoeffs = C.chebder(dcoeffs)
    ddy = C.chebval(x_scaled, ddcoeffs)

        # Sign change in second derivative → inflection point
    sign_change = np.diff(np.sign(ddy))
    inflection_idx = np.where(sign_change != 0)[0]
    inflections = np.int32(inflection_idx)

    # --- Compute strength of each inflection ---
    strengths = []
    for i in inflection_idx:
        # measure magnitude of curvature change around the inflection
        left = np.abs(ddy[i - 1]) if i > 0 else 0
        right = np.abs(ddy[i + 1]) if i < len(ddy) - 1 else 0
        strength = np.abs(right - left)
        strengths.append(strength)

    # normalize (optional, for easier comparison)
    strengths = np.array(strengths)
    if len(strengths) > 0:
        strengths = strengths / np.max(strengths)
    
    # Filter weak inflections
    strong_mask = strengths >= strength_threshold
    inflections = inflections[strong_mask]
    strengths = strengths[strong_mask]

    # Print results
    print(f"📍 Found {len(inflections)} inflection point(s) with strength threshold {strength_threshold}:")
    for idx, s in zip(inflections, strengths):
        print(f"   ↳ slice {idx:3d} | strength = {s:.3f}")
    
    return coeffs, degree, y_fit, best, inflections

def visualize_tree_segments(root_node):
    """
    Visualize all child segments of a tree (excluding the root node).

    Parameters
    ----------
    root_node : SegmentNode
        The root node of the segmentation tree.
    """
    # Gather all children recursively, excluding root
    def gather_children(node):
        nodes = []
        for c in node.children:
            nodes.append(c)
            nodes += gather_children(c)
        return nodes

    children = gather_children(root_node)
    if not children:
        print("⚠️ No child segments to visualize.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Segmented Parts (excluding root)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    cmap = plt.cm.get_cmap("tab20", len(children))

    for i, node in enumerate(children):
        if len(node.points) == 0:
            continue
        pts = np.asarray(node.points)
        color = cmap(i)[:3]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, color=color, alpha=0.8, label=node.name)

    # --- Fix scaling (make axes equal) ---
    all_pts = np.vstack([np.asarray(n.points) for n in children if len(n.points) > 0])
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2.0
    mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2.0
    mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()
    
def bitsi_with_extrema(bitsi_x, bitsi_y, bitsi_z, dominant_idx, visualize=True, strength_threshold=0.2, max_deg=10):
    # Axis conventions
    axis_names = ["X", "Y", "Z"]
    axis_colors = {"X": "r", "Y": "g", "Z": "b"}
    all_bitsi = [bitsi_x, bitsi_y, bitsi_z]


    dominant_axis = axis_names[dominant_idx]
    y = np.array(all_bitsi[dominant_idx], dtype=float)
    x = np.arange(len(y))
    # --- Fit best Chebyshev polynomial
    coeffs, degree, y_fit, metrics, inflections = fit_best_chebyshev(x, y, min_deg=2, max_deg=max_deg, criterion="bic", strength_threshold=strength_threshold)
    print(f"🎯 Chosen Chebyshev degree {degree}, adj R²={metrics['adj_r2']:.3f}, BIC={metrics['bic']:.2f}")


    # --- Plot
    if visualize:
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, "o", color=axis_colors[dominant_axis], label=f"BITSI ({dominant_axis}-axis)")
        plt.plot(x, y_fit, "--", color="k", alpha=0.7, label=f"Chebyshev deg {degree} fit")

        for i in inflections:
            plt.scatter(i, y_fit[i], color="orange", s=60, zorder=3)
            plt.text(i, y_fit[i] + 0.05, f"inflection (slice {i})", ha="center", fontsize=10, color="orange")

        plt.title(f"BITSI Curve Along Dominant Axis ({dominant_axis}-axis)")
        plt.xlabel("Slice Index")
        plt.ylabel("BITSI Value")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return x, y_fit, inflections


def pretty_print_bitsi(bitsi_x, bitsi_y, bitsi_z, slice_idx):
    axis_names = ["X", "Y", "Z"]
    axis_colors = {"X": "\033[91m", "Y": "\033[92m", "Z": "\033[94m"}  # red, green, blue
    reset = "\033[0m"

    # Compute mean and variance IBR values for each axis
    mean_vals = [np.mean(bitsi_x), np.mean(bitsi_y), np.mean(bitsi_z)]
    var_vals = [np.var(bitsi_x), np.var(bitsi_y), np.var(bitsi_z)]

    print("\n✅ BITSI Metrics Summary:")
    print("-" * 40)
    for i, (axis, mean_val, var_val) in enumerate(zip(axis_names, mean_vals, var_vals)):
        color = axis_colors[axis]
        print(f"{color}{axis}-axis:{reset} mean IBR = {mean_val:.3f}, variance = {var_val:.3f}")

    # Determine the dominant (smallest mean) axis
    dominant_axis_idx = int(np.argmin(mean_vals))
    dominant_axis = axis_names[dominant_axis_idx]

    print("-" * 40)
    print(f"🧭 Dominant slicing direction: {axis_colors[dominant_axis]}{dominant_axis}-axis{reset}")
    print(f"   (largest mean IBR = {mean_vals[dominant_axis_idx]:.3f})")

def auto_thickness(cloud_object, scale=0.01):
    bbox_min = np.min(cloud_object.points, axis=0)
    bbox_max = np.max(cloud_object.points, axis=0)
    graspable_dim = np.array([cloud_object.x_dim, cloud_object.y_dim, cloud_object.z_dim])
    search_idx = np.argmax(graspable_dim)   
    
    return scale * graspable_dim[search_idx]

def adjust_handal(pcd): 
    points_array = np.asarray(pcd.points)
    points_array /= 1000.0 
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.paint_uniform_color([0, 0, 1])
    return pcd

def adjust_kitti(pcd): 
    points_array = np.asarray(pcd.points)
    points_array /= 1000.0 
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.paint_uniform_color([0, 0, 1])
    return pcd

class SegmentNode:
    """Tree node representing a segmented region of the point cloud."""
    def __init__(self, name, points, slice_idx, parent=None, gripper_width=0.13, gripper_height=0.07, epsilon=5e-3):
        self.name = name
        self.points = points
        self.parent = parent
        self.children = []
        self.gripper_width = gripper_width
        self.gripper_height = gripper_height
        self.slice_idx = slice_idx
        # Create an Open3D point cloud for this segment
        self.pcd = o3d.geometry.PointCloud()
        if len(points) > 0:
            self.pcd.points = o3d.utility.Vector3dVector(points)
            # Optional: random color per node
            color = np.random.rand(3)
            self.pcd.paint_uniform_color(color)
        else:
            self.pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        
        self.cloud_object = build_cloud_object(self.pcd, gripper_width, gripper_height)
        thickness = auto_thickness(self.cloud_object, scale=0.03)         
        self.ibr_ratio = get_overall_ibr_ratio(self.cloud_object, epsilon=epsilon, thickness=thickness)
        
       
    
    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"SegmentNode(name={self.name}, points={len(self.points)}, children={len(self.children)})"

    def print_tree(self, level=0):
        indent = "  " * level
        parent_idx = self.parent.slice_idx if self.parent is not None else None
        print(f"{indent}📦 {self.name}: {len(self.points)} points (slice_idx={self.slice_idx}, parent={parent_idx}, ibr_ratio={self.ibr_ratio})")
        for child in self.children:
            child.print_tree(level + 1)
    
    

def fuse_consecutive_segments(node, ibr_tolerance=0.2):
    """
    Merge only consecutive child segments with similar IBR ratios.

    Parameters
    ----------
    node : SegmentNode
        Parent node whose consecutive children will be checked for fusion.
    ibr_tolerance : float
        Maximum allowed absolute difference in IBR ratio for merging.

    Returns
    -------
    node : SegmentNode
        The same node with updated children (merged if necessary).
    """
    if not node.children or len(node.children) < 2:
        return node

    fused_children = []
    i = 0
    while i < len(node.children):
        current = node.children[i]

        # Check next segment if exists
        if i + 1 < len(node.children):
            nxt = node.children[i + 1]
            diff = abs(current.ibr_ratio - nxt.ibr_ratio)

            if diff < ibr_tolerance:
                # Merge these two consecutive children
                merged_points = np.vstack([current.points, nxt.points])
                merged_name = f"{current.name}_{nxt.name}"
                merged_ibr = (current.ibr_ratio + nxt.ibr_ratio) / 2.0

                merged_child = SegmentNode(
                    name=merged_name,
                    points=merged_points,
                    slice_idx=current.slice_idx,
                    parent=node,
                    gripper_width=current.gripper_width,
                    gripper_height=current.gripper_height,
                )
                merged_child.ibr_ratio = merged_ibr
                fused_children.append(merged_child)

                # skip next since it’s merged
                i += 2
                continue

        # otherwise, keep current as-is
        fused_children.append(current)
        i += 1

    node.children = fused_children
    return node

def build_segmentation_tree(points_per_slice_to_use, bitsi_x, bitsi_y, bitsi_z, slice_idx, strength_threshold, max_deg=6, visualize=False):
    x, y_fit, inflections = bitsi_with_extrema(bitsi_x, bitsi_y, bitsi_z, slice_idx, visualize=visualize, strength_threshold=strength_threshold, max_deg=max_deg)
    x_min, x_max = np.min(x), np.max(x)
    boundaries = [x_min] + inflections.tolist() + [x_max]

    # Convert inflection ranges to slice objects
    segments = [
        slice(boundaries[i] + (1 if i > 0 else 0), boundaries[i + 1] + 1)
        for i in range(len(boundaries) - 1)
    ]

    # Collect segmented point groups
    point_segments = [points_per_slice_to_use[s] for s in segments]

    # --- Create tree
    root = SegmentNode("root_pcd", [p for sublist in points_per_slice_to_use for p in sublist], slice_idx)

    for i, pts in enumerate(point_segments):
        total_points = sum(len(point) for point in pts)
        if total_points == 0:
            continue
        child = SegmentNode(name=f"segment_{i+1}", points=[p for sublist in pts for p in sublist], parent=root, slice_idx=None)
        root.add_child(child)
        print(f"Segment {i+1}: {total_points} points")

    return root

def get_category_mapping():
    """Read the category mapping from synsetoffset2category.txt"""
    category_map = {}
    reverse_map = {}
    
    with open('/home/khiem/data/ShapeNetPart/synsetoffset2category.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    category_name, category_id = parts
                    category_map[category_id] = category_name
                    reverse_map[category_name.lower()] = category_id
    
    return category_map, reverse_map

def get_available_categories():
    """Get list of all available categories with their IDs"""
    category_map, _ = get_category_mapping()
    available = []
    
    for category_id, category_name in category_map.items():
        category_path = f'/home/khiem/data/ShapeNetPart/{category_id}'
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith('.txt')]
            if files:
                available.append((category_name, category_id, len(files)))
    
    return available

def load_point_cloud_shapenetpart(file_path):
    """
    Load point cloud data from a .txt file
    Expected format: x y z nx ny nz part_id (coordinates + normals + part annotations)
    """
    try:
        # Try to load as space-separated values
        data = np.loadtxt(file_path)
        
        if data.ndim == 1:
            # Single point, reshape to 2D
            data = data.reshape(1, -1)
        
        if data.shape[1] >= 3:
            # Extract coordinates (first 3 columns)
            points = data[:, :3]
            
            # Extract normals if available (columns 3-6)
            normals = data[:, 3:6] if data.shape[1] >= 6 else None
            
            # Extract part annotations if available (column 6)
            part_ids = data[:, 6].astype(int) if data.shape[1] >= 7 else None
            
            return points, normals, part_ids
        else:
            print(f"Warning: File {file_path} has unexpected format (only {data.shape[1]} columns)")
            return None, None, None
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None

def get_random_object(category_name, dataset_path="/home/khiem/data/ShapeNetPart"):
    """
    Get a random object from the specified category
    
    Args:
        category_name (str): Name of the category (e.g., "airplane", "chair", "table")
        dataset_path (str): Path to the ShapeNetPart dataset
    
    Returns:
        tuple: (points, normals, file_path, category_id) or (None, None, None, None) if not found
    """
    _, reverse_map = get_category_mapping()
    
    # Normalize category name
    category_name_lower = category_name.lower().strip()
    
    # Find category ID
    if category_name_lower in reverse_map:
        category_id = reverse_map[category_name_lower]
    else:
        print(f"Category '{category_name}' not found.")
        print("Available categories:")
        available = get_available_categories()
        for name, cat_id, count in available:
            print(f"  - {name} ({cat_id}): {count} files")
        return None, None, None, None
    
    # Get all files in the category
    category_path = os.path.join(dataset_path, category_id)
    if not os.path.exists(category_path):
        print(f"Category directory not found: {category_path}")
        return None, None, None, None
    
    files = [f for f in os.listdir(category_path) if f.endswith('.txt')]
    if not files:
        print(f"No .txt files found in {category_path}")
        return None, None, None, None
    
    # Select random file
    random_file = random.choice(files)
    file_path = os.path.join(category_path, random_file)
    
    # Load the point cloud
    points, normals, part_ids = load_point_cloud_shapenetpart(file_path)
    
    if points is not None:
        print(f"Loaded random {category_name} object:")
        print(f"  File: {category_id}/{random_file}")
        print(f"  Points: {len(points)}")
        print(f"  Has normals: {normals is not None}")
        print(f"  Has part annotations: {part_ids is not None}")
        if part_ids is not None:
            unique_parts = np.unique(part_ids)
            print(f"  Parts: {sorted(unique_parts)}")
        print(f"  Bounding box: [{points.min(axis=0)}] to [{points.max(axis=0)}]")
        
        return points, normals, part_ids, f"{category_id}/{random_file}", category_id
    else:
        return None, None, None, None, None



def apply_slicing_to_children(root, bitsi_x, bitsi_y, bitsi_z, epsilon=5e-3, thickness=None):
    """
    Apply slicing to all children using the 2nd slice index.
    
    Parameters
    ----------
    root : SegmentNode
        Root node containing children to be sliced
    bitsi_x, bitsi_y, bitsi_z : array-like
        BITSI metrics for each axis
    epsilon : float
        Epsilon parameter for slicing
    thickness : float, optional
        Thickness for slicing. If None, will be auto-calculated.
    
    Returns
    -------
    root : SegmentNode
        Root node with updated children (sliced if applicable)
    """
    # Determine the 2nd slice index (non-dominant axis with highest variance)  
    # Process each child
    new_children = []
    for child in root.children:
        if len(child.points) == 0:
            new_children.append(child)
            continue
            
        # Get BITSI metrics for this child
        child_bitsi_x, child_bitsi_y, child_bitsi_z, child_slice_idx, child_points_per_slice_x, child_points_per_slice_y, child_points_per_slice_z = bitsi_metric(
            child.cloud_object, epsilon=epsilon, thickness=thickness or auto_thickness(child.cloud_object, scale=0.03)
        )
        if root.slice_idx == 0:
            child_slice_idx = 1
        elif root.slice_idx == 1:
            child_slice_idx = 0
        print(child_slice_idx)
        
        # Use the 2nd slice index for this child
        child_points_per_slice = [child_points_per_slice_x, child_points_per_slice_y, child_points_per_slice_z]
        child_points_per_slice_to_use = child_points_per_slice[child_slice_idx]
        
        # Apply segmentation to this child
        child_segments = build_segmentation_tree(
            child_points_per_slice_to_use, 
            child_bitsi_x, child_bitsi_y, child_bitsi_z, 
            child_slice_idx, 
            strength_threshold=0.01
        )
        
        # Add the child's children as new children of root
        for grandchild in child_segments.children:
            grandchild.parent = root
            grandchild.slice_idx = child_slice_idx
            new_children.append(grandchild)
            print(f"  ↳ Created sub-segment: {grandchild.name} with {len(grandchild.points)} points")
    
    root.children = new_children
    return root

def visualize_shapepart_comparison(root_node, points, part_ids, parent_dir):
    """
    Visualize ShapeNetPart ground truth vs segmented parts side by side and calculate IoU.
    
    Parameters
    ----------
    root_node : SegmentNode
        Root node of the segmentation tree
    points : numpy array
        Original point cloud points
    part_ids : numpy array
        Ground truth part IDs for each point
    parent_dir : str
        Parent directory to determine if this is ShapeNetPart
    """
    if parent_dir != 'ShapeNetPart':
        print("⚠️ This comparison is only available for ShapeNetPart dataset")
        return
    
    # Gather all leaf segments
    def gather_leaves(node):
        leaves = []
        if not node.children:  # This is a leaf node
            leaves.append(node)
        else:
            for child in node.children:
                leaves += gather_leaves(child)
        return leaves

    segmented_leaves = gather_leaves(root_node)
    
    if not segmented_leaves:
        print("⚠️ No segmented parts to compare.")
        return
    
    # Create figure with side-by-side subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Left subplot: Ground truth parts
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Ground Truth Parts (ShapeNetPart)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y") 
    ax1.set_zlabel("Z")
    
    # Right subplot: Segmented parts
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("Segmented Parts (BITSI)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    # Plot ground truth parts
    unique_parts = np.unique(part_ids)
    gt_colors = plt.cm.get_cmap("tab20", len(unique_parts))
    
    for i, part_id in enumerate(unique_parts):
        mask = part_ids == part_id
        part_points = points[mask]
        if len(part_points) > 0:
            color = gt_colors(i)[:3]
            ax1.scatter(part_points[:, 0], part_points[:, 1], part_points[:, 2], 
                       s=5, color=color, alpha=0.8, label=f"Part {int(part_id)}")
    
    # Plot segmented parts
    seg_colors = plt.cm.get_cmap("tab20", len(segmented_leaves))
    
    for i, node in enumerate(segmented_leaves):
        if len(node.points) == 0:
            continue
        pts = np.asarray(node.points)
        color = seg_colors(i)[:3]
        ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                   s=5, color=color, alpha=0.8, label=node.name)
    
    # Fix scaling for both subplots
    all_pts = np.vstack([points] + [np.asarray(n.points) for n in segmented_leaves if len(n.points) > 0])
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2.0
    mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2.0
    mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2.0
    
    for ax in [ax1, ax2]:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate IoU between ground truth and segmented parts
    calculate_iou_metrics(points, part_ids, segmented_leaves)

def calculate_iou_metrics(points, part_ids, segmented_leaves):
    """
    Calculate IoU metrics between ground truth and segmented parts.
    Fuses adjacent slices to maximize IoU.
    
    Parameters
    ----------
    points : numpy array
        Original point cloud points
    part_ids : numpy array  
        Ground truth part IDs
    segmented_leaves : list
        List of segmented leaf nodes
    """
    print("\n📊 IoU Analysis (with slice fusion):")
    print("=" * 50)
    
    # Create point-to-segment mapping for segmented parts
    seg_point_to_part = {}
    for i, leaf in enumerate(segmented_leaves):
        if len(leaf.points) == 0:
            continue
        for point in leaf.points:
            # Find closest point in original point cloud
            distances = np.linalg.norm(points - point, axis=1)
            closest_idx = np.argmin(distances)
            seg_point_to_part[closest_idx] = i
    
    # Calculate IoU for each ground truth part with fusion
    unique_gt_parts = np.unique(part_ids)
    best_matches = {}
    
    for gt_part_id in unique_gt_parts:
        gt_mask = part_ids == gt_part_id
        gt_indices = np.where(gt_mask)[0]
        
        print(f"\n🔍 Analyzing GT Part {int(gt_part_id)} ({len(gt_indices)} points):")
        
        # Find which segmented parts overlap with this ground truth part
        seg_part_overlaps = {}
        for gt_idx in gt_indices:
            if gt_idx in seg_point_to_part:
                seg_part = seg_point_to_part[gt_idx]
                if seg_part not in seg_part_overlaps:
                    seg_part_overlaps[seg_part] = 0
                seg_part_overlaps[seg_part] += 1
        
        if not seg_part_overlaps:
            print(f"  ⚠️ No overlap with any segmented parts")
            best_matches[gt_part_id] = (None, 0.0)
            continue
        
        # Try different combinations of segmented parts (fusion)
        best_iou = 0
        best_combination = None
        best_combination_type = ""
        
        # 1. Try individual parts
        for seg_part, overlap_count in seg_part_overlaps.items():
            seg_mask = np.array([i in seg_point_to_part and seg_point_to_part[i] == seg_part 
                               for i in range(len(points))])
            seg_indices = np.where(seg_mask)[0]
            
            intersection = overlap_count
            union = len(gt_indices) + len(seg_indices) - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_combination = [seg_part]
                best_combination_type = "single"
        
        # 2. Try pairs of adjacent parts
        overlapping_parts = list(seg_part_overlaps.keys())
        for i in range(len(overlapping_parts)):
            for j in range(i + 1, len(overlapping_parts)):
                part1, part2 = overlapping_parts[i], overlapping_parts[j]
                
                # Check if parts are adjacent (have overlapping points)
                seg_mask1 = np.array([i in seg_point_to_part and seg_point_to_part[i] == part1 
                                    for i in range(len(points))])
                seg_mask2 = np.array([i in seg_point_to_part and seg_point_to_part[i] == part2 
                                    for i in range(len(points))])
                
                # Fuse the two parts
                fused_mask = seg_mask1 | seg_mask2
                fused_indices = np.where(fused_mask)[0]
                
                # Calculate overlap with fused part
                fused_overlap = 0
                for gt_idx in gt_indices:
                    if gt_idx in seg_point_to_part and seg_point_to_part[gt_idx] in [part1, part2]:
                        fused_overlap += 1
                
                intersection = fused_overlap
                union = len(gt_indices) + len(fused_indices) - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_combination = [part1, part2]
                    best_combination_type = "pair"
        
        # 3. Try triplets of parts
        for i in range(len(overlapping_parts)):
            for j in range(i + 1, len(overlapping_parts)):
                for k in range(j + 1, len(overlapping_parts)):
                    part1, part2, part3 = overlapping_parts[i], overlapping_parts[j], overlapping_parts[k]
                    
                    # Fuse the three parts
                    seg_mask1 = np.array([i in seg_point_to_part and seg_point_to_part[i] == part1 
                                        for i in range(len(points))])
                    seg_mask2 = np.array([i in seg_point_to_part and seg_point_to_part[i] == part2 
                                        for i in range(len(points))])
                    seg_mask3 = np.array([i in seg_point_to_part and seg_point_to_part[i] == part3 
                                        for i in range(len(points))])
                    
                    fused_mask = seg_mask1 | seg_mask2 | seg_mask3
                    fused_indices = np.where(fused_mask)[0]
                    
                    # Calculate overlap with fused part
                    fused_overlap = 0
                    for gt_idx in gt_indices:
                        if gt_idx in seg_point_to_part and seg_point_to_part[gt_idx] in [part1, part2, part3]:
                            fused_overlap += 1
                    
                    intersection = fused_overlap
                    union = len(gt_indices) + len(fused_indices) - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_combination = [part1, part2, part3]
                        best_combination_type = "triplet"
        
        # 4. Try all overlapping parts fused together
        if len(overlapping_parts) > 3:
            # Fuse all overlapping parts
            fused_mask = np.zeros(len(points), dtype=bool)
            for part in overlapping_parts:
                part_mask = np.array([i in seg_point_to_part and seg_point_to_part[i] == part 
                                    for i in range(len(points))])
                fused_mask |= part_mask
            
            fused_indices = np.where(fused_mask)[0]
            
            # Calculate overlap with all fused parts
            fused_overlap = 0
            for gt_idx in gt_indices:
                if gt_idx in seg_point_to_part and seg_point_to_part[gt_idx] in overlapping_parts:
                    fused_overlap += 1
            
            intersection = fused_overlap
            union = len(gt_indices) + len(fused_indices) - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_combination = overlapping_parts
                best_combination_type = "all_fused"
        
        best_matches[gt_part_id] = (best_combination, best_iou)
        
        if best_combination:
            print(f"  ✅ Best IoU = {best_iou:.3f} ({best_combination_type})")
            print(f"     Combination: {best_combination}")
        else:
            print(f"  ❌ No valid combination found")
    
    # Calculate overall metrics
    valid_matches = [iou for _, (_, iou) in best_matches.items() if iou > 0]
    mean_iou = np.mean(valid_matches) if valid_matches else 0
    
    print(f"\n📈 Overall Mean IoU: {mean_iou:.3f}")
    
    # Count how many GT parts have good matches (IoU > 0.5)
    good_matches = sum(1 for _, (_, iou) in best_matches.items() if iou > 0.5)
    print(f"🎯 Good matches (IoU > 0.5): {good_matches}/{len(unique_gt_parts)}")
    
    # Show fusion statistics
    fusion_types = {}
    for _, (combination, iou) in best_matches.items():
        if combination:
            if len(combination) == 1:
                fusion_type = "single"
            elif len(combination) == 2:
                fusion_type = "pair"
            elif len(combination) == 3:
                fusion_type = "triplet"
            else:
                fusion_type = "multi"
            
            if fusion_type not in fusion_types:
                fusion_types[fusion_type] = 0
            fusion_types[fusion_type] += 1
    
    print(f"\n🔧 Fusion Statistics:")
    for fusion_type, count in fusion_types.items():
        print(f"  {fusion_type.capitalize()}: {count} parts")
    
    return best_matches, mean_iou

def rename_segments(root, prefix="segment"):
    """
    Rename all segments in the tree with sequential numbers.
    
    Parameters
    ----------
    root : SegmentNode
        Root node of the segmentation tree
    prefix : str, optional
        Prefix to use for segment names, defaults to "segment"
        
    Returns
    -------
    root : SegmentNode
        Root node with renamed segments
    """
    def _rename_recursive(node, counter=[0]):
        # Skip renaming the root node
        if node.parent is not None:
            node.name = f"{prefix}_{counter[0]}"
            counter[0] += 1
            
        # Recursively rename children
        for child in node.children:
            _rename_recursive(child, counter)
            
    _rename_recursive(root)
    return root


def main():
    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    
    # Create an ArgumentParser Object
    parser = argparse.ArgumentParser(description='Task-Oriented Grasp Synthesis')

    # Add a command-line argument for the input filenamep
    parser.add_argument('--object', type=str, help='Name of the ycb object', default='toy')
    parser.add_argument('--width', type=float, help='gripper width', default=0.13)
    parser.add_argument('--height', type=float, help='gripper height', default=0.07)      
    parser.add_argument('--partial', action='store_true', help='use partial or not partial data')
    parser.add_argument('--epsilon', type=float, default=5e-3 )
    parser.add_argument('--parent_dir', type=str, default='YCBV')
    parser.add_argument('--obj_num', type=int, default=1)
    
    args = parser.parse_args()
    gripper_width = args.width 
    gripper_height = args.height  
    epsilon = args.epsilon 
    
    object_path = args.object
    parent_dir = args.parent_dir
    
    # Parameters for segmentation 
    ibr_tolerance = 0.07
    strength_threshold = 0.01
    epsilon = 5e-3 
    
    # -------------------------------------------------------------------------
    # Directory management
    # -------------------------------------------------------------------------
    if parent_dir == 'HANDAL':
        model_num = f"{args.obj_num:0{6}}"

        parent_dir = os.path.join("/home/khiem/Robotics/obj-decomposition/",args.parent_dir)
        pcd = o3d.io.read_point_cloud(f"{parent_dir}/{object_path}/models/obj_{model_num}.ply")
        pcd_handle =  o3d.io.read_point_cloud(f"{parent_dir}/{object_path}/models_parts/obj_{model_num}_handle.ply")
        pcd_body =  o3d.io.read_point_cloud(f"{parent_dir}/{object_path}/models_parts/obj_{model_num}_not.ply")
        
        
        pcd = adjust_handal(pcd)
        pcd_body = adjust_handal(pcd_body)
        pcd_handle = adjust_handal(pcd_handle)
        pcd_handle.paint_uniform_color([0, 1, 0])

    elif parent_dir == "YCBV" or parent_dir == 'YCBV-Partial':
        parent_dir = os.path.join("/home/khiem/Robotics/obj-decomposition",args.parent_dir)
      
        pcd = o3d.io.read_point_cloud(f"{parent_dir}/{object_path}/nontextured.ply")            
        pcd.paint_uniform_color([0, 0, 1])
        
    elif parent_dir == 'KITTI':
        parent_dir = os.path.join("/home/khiem/Robotics/obj-decomposition/",args.parent_dir)
        pcd = o3d.io.read_point_cloud(f"{parent_dir}/{object_path}.ply")   
        pcd = adjust_kitti(pcd)
    
    elif parent_dir == 'ShapeNetPart':
        points, normals, part_ids, file_path, category_id = get_random_object(object_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.paint_uniform_color([0, 0, 1])
        
    # -------------------------------------------------------------------------
    # Loading Point Cloud
    # -------------------------------------------------------------------------
    object_path = args.object
    print(f"📦 Loading object: {args.object}")
    pcd.paint_uniform_color([0, 0, 1]) # --> use only pc color 
    if pcd.is_empty():
        sys.exit(f"❌ Error: Loaded point cloud is empty at {object_path}")

    # -------------------------------------------------------------------------
    # Calculate BITSI Metric
    # -------------------------------------------------------------------------
    cloud_object = build_cloud_object(pcd, gripper_width, gripper_height)
    thickness = auto_thickness(cloud_object, scale=0.03)
    print(f"📏 Adaptive thickness selected: {thickness:.6f} (scale=5%)")
    print("⚙️  Computing BITSI metrics...")
    bitsi_x, bitsi_y, bitsi_z, slice_idx, points_per_slice_x, points_per_slice_y, points_per_slice_z = bitsi_metric(cloud_object, epsilon=epsilon, thickness=thickness)
    points_per_slice = [points_per_slice_x, points_per_slice_y, points_per_slice_z] 
    points_per_slice_to_use = points_per_slice[slice_idx]
    pretty_print_bitsi(bitsi_x, bitsi_y, bitsi_z, slice_idx)
   
    #visualize_bitsi(bitsi_x, bitsi_y, bitsi_z)

    # -------------------------------------------------------------------------
    # Segmentation With BITSI Metric
    # -------------------------------------------------------------------------
    root = build_segmentation_tree(points_per_slice_to_use, bitsi_x, bitsi_y, bitsi_z, slice_idx, strength_threshold)
    root = fuse_consecutive_segments(root, ibr_tolerance=ibr_tolerance)
    root = apply_slicing_to_children(root, bitsi_x, bitsi_y, bitsi_z, epsilon=epsilon, thickness=thickness)
    root = rename_segments(root)
    
    
    #root.print_tree()
    #fuse_oversegmented_children(root, ibr_tol=0.05, var_tol=0.01) 
    root.print_tree()
    if parent_dir == 'ShapeNetPart':
        # For ShapeNetPart, show comparison with ground truth
        visualize_shapepart_comparison(root, points, part_ids, parent_dir)
    else:
        # For other datasets, use regular visualization
        visualize_tree_segments(root)
    
if __name__ == "__main__":
    main()