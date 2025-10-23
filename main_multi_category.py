#!/usr/bin/env python3
"""
Multi-category scene generation script.
Loads objects from different categories and places them in a scene without segmentation.
Works with all datasets specified in the original main.py.
"""

import argparse
import open3d as o3d 
import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
import random
import math

# Import functions from the original main.py
from main import (
    adjust_handal, adjust_kitti, get_random_object, get_category_mapping, 
    get_available_categories, load_point_cloud_shapenetpart
)

class SceneObject:
    """Represents an object in the scene with its position and orientation."""
    
    def __init__(self, pcd, name, position=(0, 0, 0), rotation=0, scale=1.0):
        self.pcd = pcd
        self.name = name
        self.position = np.array(position)
        self.rotation = rotation
        self.scale = scale
        self.transformed_pcd = None
        
    def apply_transformation(self):
        """Apply position, rotation, and scale transformations to the point cloud."""
        # Create a copy of the point cloud
        self.transformed_pcd = o3d.geometry.PointCloud(self.pcd)
        
        # Apply scale
        if self.scale != 1.0:
            self.transformed_pcd.scale(self.scale, center=self.transformed_pcd.get_center())
        
        # Apply rotation around Z-axis
        if self.rotation != 0:
            rotation_matrix = np.array([
                [np.cos(self.rotation), -np.sin(self.rotation), 0],
                [np.sin(self.rotation), np.cos(self.rotation), 0],
                [0, 0, 1]
            ])
            self.transformed_pcd.rotate(rotation_matrix, center=self.transformed_pcd.get_center())
        
        # Apply translation
        self.transformed_pcd.translate(self.position)
        
        return self.transformed_pcd

def load_random_object_from_dataset(parent_dir, object_name, obj_num=None):
    """
    Load a random object from the specified dataset.
    
    Parameters
    ----------
    parent_dir : str
        Dataset directory (HANDAL, YCBV, KITTI, ShapeNetPart)
    object_name : str
        Object name/category
    obj_num : int, optional
        Object number for HANDAL dataset
        
    Returns
    -------
    tuple
        (pcd, object_info) where object_info contains metadata
    """
    if parent_dir == 'HANDAL':
        if obj_num is None:
            obj_num = random.randint(1, 100)  # Random object number
        
        model_num = f"{obj_num:0{6}}"
        parent_dir_path = os.path.join("/home/khiem/Robotics/obj-decomposition/", parent_dir)
        
        pcd = o3d.io.read_point_cloud(f"{parent_dir_path}/{object_name}/models/obj_{model_num}.ply")
        pcd = adjust_handal(pcd)
        pcd.paint_uniform_color([0, 0, 1])
        
        object_info = {
            'type': 'HANDAL',
            'object_name': object_name,
            'obj_num': obj_num,
            'model_num': model_num
        }
        
    elif parent_dir == "YCBV" or parent_dir == 'YCBV-Partial':
        parent_dir_path = os.path.join("/home/khiem/Robotics/obj-decomposition", parent_dir)
        pcd = o3d.io.read_point_cloud(f"{parent_dir_path}/{object_name}/nontextured.ply")
        pcd.paint_uniform_color([0, 0, 1])
        
        object_info = {
            'type': 'YCBV',
            'object_name': object_name
        }
        
    elif parent_dir == 'KITTI':
        parent_dir_path = os.path.join("/home/khiem/Robotics/obj-decomposition/", parent_dir)
        pcd = o3d.io.read_point_cloud(f"{parent_dir_path}/{object_name}.ply")
        pcd = adjust_kitti(pcd)
        
        object_info = {
            'type': 'KITTI',
            'object_name': object_name
        }
        
    elif parent_dir == 'ShapeNetPart':
        points, normals, part_ids, file_path, category_id = get_random_object(object_name)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.paint_uniform_color([0, 0, 1])
        
        object_info = {
            'type': 'ShapeNetPart',
            'object_name': object_name,
            'category_id': category_id,
            'file_path': file_path,
            'points': points,
            'normals': normals,
            'part_ids': part_ids
        }
    
    else:
        raise ValueError(f"Unknown dataset: {parent_dir}")
    
    if pcd.is_empty():
        return None, None
        
    return pcd, object_info

def get_available_objects_for_dataset(parent_dir):
    """
    Get available objects for a specific dataset.
    
    Parameters
    ----------
    parent_dir : str
        Dataset directory
        
    Returns
    -------
    list
        List of available object names
    """
    if parent_dir == 'HANDAL':
        # Get available objects from HANDAL dataset
        handal_path = "/home/khiem/Robotics/obj-decomposition/HANDAL"
        if os.path.exists(handal_path):
            return [d for d in os.listdir(handal_path) if os.path.isdir(os.path.join(handal_path, d))]
        return []
    
    elif parent_dir == "YCBV" or parent_dir == 'YCBV-Partial':
        # Get available objects from YCBV dataset
        ycbv_path = os.path.join("/home/khiem/Robotics/obj-decomposition", parent_dir)
        if os.path.exists(ycbv_path):
            return [d for d in os.listdir(ycbv_path) if os.path.isdir(os.path.join(ycbv_path, d))]
        return []
    
    elif parent_dir == 'KITTI':
        # Get available objects from KITTI dataset
        kitti_path = "/home/khiem/Robotics/obj-decomposition/KITTI"
        if os.path.exists(kitti_path):
            return [f.replace('.ply', '') for f in os.listdir(kitti_path) if f.endswith('.ply')]
        return []
    
    elif parent_dir == 'ShapeNetPart':
        # Get available categories from ShapeNetPart
        available_categories = get_available_categories()
        return [cat[0] for cat in available_categories]  # Return category names
    
    return []

def generate_scene_layout(num_objects, scene_size=2.0):
    """
    Generate random positions for objects in the scene.
    
    Parameters
    ----------
    num_objects : int
        Number of objects to place
    scene_size : float
        Size of the scene (radius)
        
    Returns
    -------
    list
        List of (position, rotation, scale) tuples
    """
    layouts = []
    
    for i in range(num_objects):
        # Random x, y position within scene bounds
        x = random.uniform(-scene_size, scene_size)
        y = random.uniform(-scene_size, scene_size)
        z = 0  # fixed height (no random z)

        rotation = 0.0  # fixed rotation
        scale = 1.0     # fixed scale

        layouts.append(((x, y, z), rotation, scale))
            
    return layouts

def create_multi_category_scene(parent_dir, object_categories, scene_size=2.0, 
                               gripper_width=0.13, gripper_height=0.07, epsilon=5e-3,
                               strength_threshold=0.01, ibr_tolerance=0.02):
    """
    Create a multi-category scene with objects from different categories.
    
    Parameters
    ----------
    parent_dir : str
        Dataset directory
    object_categories : list
        List of object categories to load
    scene_size : float
        Size of the scene
    gripper_width, gripper_height : float
        Gripper parameters
    epsilon : float
        BITSI epsilon parameter
    strength_threshold : float
        Inflection detection threshold
    ibr_tolerance : float
        Segment fusion tolerance
        
    Returns
    -------
    tuple
        (scene_pcd, scene_objects, segmentation_results)
    """
    print(f"🎬 Creating multi-category scene with {len(object_categories)} different objects from {parent_dir}")
    print("=" * 60)
    
    # Generate scene layout
    layouts = generate_scene_layout(len(object_categories), scene_size)
    
    # Load objects
    scene_objects = []
    all_point_clouds = []
    
    for i, (object_name, (position, rotation, scale)) in enumerate(zip(object_categories, layouts)):
        print(f"📦 Loading object {i+1}/{len(object_categories)}: {object_name}...")
        
        # Load object
        pcd, object_info = load_random_object_from_dataset(parent_dir, object_name)
        
        if pcd is None:
            print(f"⚠️ Failed to load {object_name}, skipping...")
            continue
        
        # Create scene object
        scene_obj = SceneObject(pcd, f"{object_name}_{i+1}", position, rotation, scale)
        scene_objects.append((scene_obj, object_info))
        
        # Apply transformations
        transformed_pcd = scene_obj.apply_transformation()
        
        # Assign unique color to each object
        color = plt.cm.get_cmap("tab20", len(object_categories))(i)[:3]
        transformed_pcd.paint_uniform_color(color)
        
        all_point_clouds.append(transformed_pcd)
        print(f"  ✅ {object_name}: {len(transformed_pcd.points)} points at {position}")
    
    if not all_point_clouds:
        print("❌ No objects loaded successfully!")
        return None, None, None
    
    # Combine all point clouds into scene
    scene_pcd = all_point_clouds[0]
    for pcd in all_point_clouds[1:]:
        scene_pcd += pcd
    
    print(f"\n🎯 Scene created: {len(scene_pcd.points)} total points")
    print(f"📏 Scene bounds: {scene_pcd.get_axis_aligned_bounding_box().get_extent()}")
    
    # Apply BITSI segmentation to the entire scene
    print("\n🔍 Applying BITSI segmentation to the entire scene...")
    
    try:
        # Import BITSI functions
        from main import (
            build_cloud_object, bitsi_metric, build_segmentation_tree, 
            fuse_consecutive_segments, auto_thickness, pretty_print_bitsi
        )
        
        # Build cloud object for the entire scene
        cloud_object = build_cloud_object(scene_pcd, gripper_width, gripper_height)
        thickness = auto_thickness(cloud_object, scale=0.03)
        print(f"📏 Adaptive thickness: {thickness:.6f}")
        
        # Compute BITSI metrics for the entire scene
        bitsi_x, bitsi_y, bitsi_z, slice_idx, points_per_slice_x, points_per_slice_y, points_per_slice_z = bitsi_metric(
            cloud_object, epsilon=epsilon, thickness=thickness
        )
        
        points_per_slice = [points_per_slice_x, points_per_slice_y, points_per_slice_z]
        points_per_slice_to_use = points_per_slice[slice_idx]
        
        # Print BITSI metrics
        pretty_print_bitsi(bitsi_x, bitsi_y, bitsi_z, slice_idx) 
        
        # Perform segmentation on the entire scene
        try:
            root = build_segmentation_tree(points_per_slice_to_use, bitsi_x, bitsi_y, bitsi_z, slice_idx, strength_threshold)
        except Exception as e:
            breakpoint()
            print(f"❌ Segmentation failed: {e}")
            return None, None, None
        #root = fuse_consecutive_segments(root, ibr_tolerance=ibr_tolerance)
        
        segmentation_results = {
            'root': root,
            'bitsi_x': bitsi_x,
            'bitsi_y': bitsi_y,
            'bitsi_z': bitsi_z,
            'slice_idx': slice_idx,
            'thickness': thickness,
            'cloud_object': cloud_object
        }
        
        print("✅ BITSI segmentation completed successfully!")
        
    except Exception as e:
        print(f"❌ BITSI segmentation failed: {e}")
        segmentation_results = None
    
    return scene_pcd, scene_objects, segmentation_results

def visualize_multi_category_scene(scene_pcd, scene_objects, segmentation_results=None, 
                                 show_individual_objects=True, show_segmentation=True):
    """
    Visualize the multi-category scene.
    
    Parameters
    ----------
    scene_pcd : o3d.geometry.PointCloud
        Combined scene point cloud
    scene_objects : list
        List of (SceneObject, object_info) tuples
    segmentation_results : dict, optional
        Segmentation results
    show_individual_objects : bool
        Whether to show individual objects
    show_segmentation : bool
        Whether to show segmentation results
    """
    if show_individual_objects:
        # Show individual objects
        print("\n🎨 Visualizing individual objects...")
        
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.set_title("Individual Objects in Scene")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        
        # Plot each object with different colors
        for i, (scene_obj, object_info) in enumerate(scene_objects):
            if scene_obj.transformed_pcd is not None:
                points = np.asarray(scene_obj.transformed_pcd.points)
                color = plt.cm.get_cmap("tab20", len(scene_objects))(i)[:3]
                ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           s=5, color=color, alpha=0.8, label=f"{object_info['object_name']}")
        
        ax1.legend()
        
        # Show combined scene
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_title("Combined Scene")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        
        points = np.asarray(scene_pcd.points)
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], s=3, alpha=0.6)
        
        # Fix scaling
        for ax in [ax1, ax2]:
            all_pts = points
            max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
            mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2.0
            mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2.0
            mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2.0
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()
    
    if show_segmentation and segmentation_results is not None:
        # Show segmentation results
        print("\n🔍 Visualizing BITSI segmentation results...")
        root = segmentation_results['root']
        root.print_tree()
        
        # Import visualization function
        from main import visualize_tree_segments
        visualize_tree_segments(root)

def main():
    """Main function for multi-category scene generation."""
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Multi-Category Scene Generation (No Segmentation)')
    
    # Dataset parameters
    parser.add_argument('--parent_dir', type=str, default='YCBV', 
                       choices=['HANDAL', 'YCBV', 'YCBV-Partial', 'KITTI', 'ShapeNetPart'],
                       help='Dataset directory')
    parser.add_argument('--objects', type=str, nargs='+', default=['mug', 'hammer', 'flat_screwdriver'],
                       help='List of object categories to load')
    parser.add_argument('--num_objects', type=int, default=None,
                       help='Number of random objects to load (if not specified, uses --objects list)')
    
    # Scene parameters
    parser.add_argument('--scene_size', type=float, default=0.5,
                       help='Size of the scene (radius)')
    
    # Visualization parameters
    parser.add_argument('--no_individual', action='store_true',
                       help='Skip individual object visualization')
    
    args = parser.parse_args()
    
    print("🎬 Multi-Category Scene Generation (No Segmentation)")
    print("=" * 60)
    print(f"📦 Dataset: {args.parent_dir}")
    
    # Determine object categories
    if args.num_objects is not None:
        # Load random objects
        available_objects = get_available_objects_for_dataset(args.parent_dir)
        if not available_objects:
            print(f"❌ No objects found for dataset {args.parent_dir}")
            return
        
        if args.num_objects > len(available_objects):
            print(f"⚠️ Requested {args.num_objects} objects, but only {len(available_objects)} available")
            args.num_objects = len(available_objects)
        
        object_categories = random.sample(available_objects, args.num_objects)
        print(f"🔢 Loading {args.num_objects} random objects: {object_categories}")
    else:
        # Use specified objects
        object_categories = args.objects
        print(f"🏷️  Loading specified objects: {object_categories}")
    
    print(f"📏 Scene size: {args.scene_size}")
    print("=" * 60)
    
    # Create multi-category scene
    scene_pcd, scene_objects, segmentation_results = create_multi_category_scene(
        parent_dir=args.parent_dir,
        object_categories=object_categories,
        scene_size=args.scene_size
    )
    
    if scene_pcd is None:
        print("❌ Failed to create scene!")
        return
    
    # Visualize results
    visualize_multi_category_scene(
        scene_pcd, scene_objects, segmentation_results,
        show_individual_objects=not args.no_individual
    )
    
    print("\n🎉 Multi-category scene generation completed!")
    print("📝 Note: No segmentation was performed - objects are loaded as-is")

if __name__ == "__main__":
    main()
