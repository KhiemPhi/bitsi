import numpy as np 
import matplotlib.pyplot as plt
from numpy import linalg as la
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d
import string
from .point_cloud import PointCloud
from .occupancy import OccupancyStruct
import os 



def build_cloud_object(pcd, gripper_width, gripper_height):
    cloud_object = PointCloud()
    cloud_object.processed_cloud = pcd
    cloud_object.points = np.asarray(cloud_object.processed_cloud.points)

    # Computing the normals for this point cloud:
    cloud_object.processed_cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    cloud_object.processed_cloud.estimate_normals()
    cloud_object.processed_cloud.orient_normals_consistent_tangent_plane(30)
    cloud_object.normals_base_frame = np.asarray(cloud_object.processed_cloud.normals)
    
    # Specifying gripper tolerances:
    cloud_object.gripper_width_tolerance = gripper_width
    cloud_object.gripper_height_tolerance = gripper_height

    # Computing the bounding boxes corresponding to the object point cloud: 
    cloud_object.compute_bounding_box()
    cloud_object.vertices = cloud_object.oriented_bounding_box_vertices
    cloud_object.plot_cube()

    return cloud_object



def get_face_distance(faces, object_frame_points, cloud_object=None):
    """
    Calculate the distance between selected bbox faces and points of an object in a cloud point.

    Args:
    - faces (list): A list of detected faces, each represented by a set [x1,y1,x2,y2]
    - object_frame_points (list): The coordinates of the object in the frame.
    - cloud_object (point_cloud): The point cloud representing the object.

    Returns:
    - distances (list): A list of distances between each selected bbox face and the specified object.
    """
    center_coordinates = np.array([np.mean(face, axis=0) for face in faces])
    threshold = 1e-10
    center_coordinates[np.abs(center_coordinates) < threshold] = 0
    
    face_distances = []
    unit_vecs = []
   
    for point in object_frame_points:
        distances = []
        unit_vec = []
        for face, center in zip(faces, center_coordinates):
            distance, unit_u = get_plane_distance(face, center, point)
            distances.append(distance)
            unit_vec.append(unit_u)
        face_distances.append(distances)
        unit_vecs.append(unit_vec)

    face_distances = np.asarray(face_distances)  
    unit_vecs = np.stack(unit_vecs)   
    return face_distances, unit_vecs

def get_faces(cloud_object): 
    left_face = [  cloud_object.transformed_vertices_object_frame[1], cloud_object.transformed_vertices_object_frame[0], cloud_object.transformed_vertices_object_frame[3], cloud_object.transformed_vertices_object_frame[6]    ]
    right_face = [ cloud_object.transformed_vertices_object_frame[2],  cloud_object.transformed_vertices_object_frame[7],  cloud_object.transformed_vertices_object_frame[4],  cloud_object.transformed_vertices_object_frame[5] ]
    
    front_face = [  cloud_object.transformed_vertices_object_frame[0], cloud_object.transformed_vertices_object_frame[2], cloud_object.transformed_vertices_object_frame[5], cloud_object.transformed_vertices_object_frame[3]    ]
    back_face = [ cloud_object.transformed_vertices_object_frame[7],  cloud_object.transformed_vertices_object_frame[1],  cloud_object.transformed_vertices_object_frame[6],  cloud_object.transformed_vertices_object_frame[4] ]

    top_face = [ cloud_object.transformed_vertices_object_frame[1],  cloud_object.transformed_vertices_object_frame[0],  cloud_object.transformed_vertices_object_frame[2],  cloud_object.transformed_vertices_object_frame[7] ]
    bottom_face = [ cloud_object.transformed_vertices_object_frame[6],  cloud_object.transformed_vertices_object_frame[3],  cloud_object.transformed_vertices_object_frame[5],  cloud_object.transformed_vertices_object_frame[4] ]
    
    front_face = np.asarray(front_face)
    back_face = np.asarray(back_face)

    left_face = np.asarray(left_face) # this is front face
    right_face = np.asarray(right_face) # this is front face

    top_face = np.asarray(top_face) # some bugs here
    bottom_face = np.asarray(bottom_face) # some bugs here
    
    return left_face, right_face, front_face, back_face, top_face, bottom_face


def slice_bbox(vertices, cloud_object, thickness, slice_idx):
    """
    Partition a set of points in a point cloud into bounding boxes.

    Parameters:
    - point_cloud: numpy array representing the point cloud (Nx3, N points in 3D space)
    - box_dimensions: tuple (lx, wx, h) representing the dimensions of the bounding box

    Returns:
    - list of numpy arrays, each array representing points within a bounding box
    """    
    #1. Pick Axis to Slice From
    start_value = np.min(vertices[:, slice_idx], axis=0)
    end_value = np.max(vertices[:, slice_idx], axis=0)
    # Determine the direction of the step size
    direction = 1 if thickness > 0 else -1
    # Calculate the adjusted stop value based on the direction
    stop_value = end_value + direction * np.finfo(float).eps  # Adding epsilon to avoid floating-point precision issues
    steps = np.arange(start_value, stop_value+thickness, thickness)
    new_vertices = []
    for i in range(len(steps) - 1):
        step_value_min = steps[i]
        step_value_max = steps[i+1]
        new_bbox = vertices.copy()
        
        for vertex in new_bbox: 
            if vertex[slice_idx] == start_value:
                vertex[slice_idx] = step_value_min
            else: 
                vertex[slice_idx] = step_value_max
        new_vertices.append(new_bbox)
        
    return new_vertices


def points_inside_bbox(points, bbox_min, bbox_max):
    """
    Check which points are inside a bounding box.

    Parameters:
    - points: List of 3D points where each point is represented as [x, y, z].
    - bbox_min: Minimum coordinates of the bounding box, e.g., [min_x, min_y, min_z].
    - bbox_max: Maximum coordinates of the bounding box, e.g., [max_x, max_y, max_z].

    Returns:
    - List of boolean values indicating whether each point is inside the bounding box.
    """
    return points[np.where(np.all((bbox_min <= points) & (points <= bbox_max), axis=1)==True)], np.where(np.all((bbox_min <= points) & (points <= bbox_max), axis=1)==True)

def organize_coordinates(coordinates):
    left_face = [  coordinates[1], coordinates[0], coordinates[3], coordinates[6]    ]
    right_face = [ coordinates[2],  coordinates[7],  coordinates[4],  coordinates[5] ]
    
    front_face = [  coordinates[0], coordinates[2], coordinates[5], coordinates[3]    ]
    back_face = [ coordinates[7],  coordinates[1],  coordinates[6],  coordinates[4] ]

    top_face = [ coordinates[1],  coordinates[0],  coordinates[2],  coordinates[7] ]
    bottom_face = [ coordinates[6],  coordinates[3],  coordinates[5],  coordinates[4] ]
    
    front_face = np.asarray(front_face)
    back_face = np.asarray(back_face)

    left_face = np.asarray(left_face) # this is front face
    right_face = np.asarray(right_face) # this is front face

    top_face = np.asarray(top_face) # some bugs here
    bottom_face = np.asarray(bottom_face) # some bugs here
    
    return [left_face, right_face, front_face, back_face, top_face, bottom_face]
    


def accumulate_slices(bbox_vertices, object_frame_points, face_distances, epsilon, ax2=None): 
     #3. Calculate Fill Ratio
    ibr_ratios = []
    points_per_slice = []
    empty_ratios = []
    for idx, vertices in enumerate(bbox_vertices):
        
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)        
        
        points_inside=[]
        points_idx = []
       
      
        
        points_inside, points_idx = points_inside_bbox(object_frame_points, bbox_min, bbox_max) 
        
               
        slice_dist = face_distances[points_idx]
        slice_dist = np.min(slice_dist, axis=1)
        
        
        boundary_idx = np.where(slice_dist < epsilon)[0]
        interior_idx = np.where(slice_dist >= epsilon)[0] 
        grid_vertices = np.hstack([  np.min(vertices, axis=0), np.max(vertices, axis=0) ])
        
        occupancy_struct = OccupancyStruct(grid_vertices, object_frame_points, 0.01)
        empty_ratio = occupancy_struct.occupied_comps
        empty_ratios.append(empty_ratio)
        
        total = len(boundary_idx) + len(interior_idx)
        ibr_ratio = len(boundary_idx) / total if total > 0 else 0

        if len(points_inside) > 0:
            points_per_slice.append(points_inside)
            if len(interior_idx) > 0 and len(boundary_idx) <= len(interior_idx):
                ibr_ratios.append(ibr_ratio)
            elif len(interior_idx) == 0:
                ibr_ratios.append(ibr_ratio) 
            else: 
                ibr_ratios.append(ibr_ratio) 
        else: 
            points_per_slice.append([])
            ibr_ratios.append(-1) 
        
    
    
    return ibr_ratios, empty_ratios, points_per_slice

def fuse_slices(points_per_slice, ibr_ratios, empty_ratios, theta, thickness, slice_idx, gripper_width): 
    point_groups = [ [points_per_slice[0]] ] # init to first location of non-empty points and get that idx
    group_idx = 0
    empty_hit = False
    
    #3. Group Fusion
    
    for ratio_idx in range(len(ibr_ratios) - 1):
        pair_diff = (ibr_ratios[ratio_idx]+empty_ratios[ratio_idx+1]) - (ibr_ratios[ratio_idx + 1]+empty_ratios[ratio_idx+1])
       
        if np.abs(pair_diff) >= theta or ( (len(point_groups[group_idx])+1)*thickness > gripper_width  ) or len(points_per_slice[ratio_idx]) == 0: # add to current group if no sharp-change and no exceeding grip and not too small 
            
            if len(points_per_slice[ratio_idx]) > 0:
                point_groups[group_idx].append(points_per_slice[ratio_idx])
                
                group_idx += 1 
                point_groups.append([])
                empty_hit = False
            elif empty_hit == False:                 
                empty_hit = True

            
            if ratio_idx < len(ibr_ratios) - 1 and len(points_per_slice[ratio_idx+1]) > 0:
                point_groups[group_idx].append(points_per_slice[ratio_idx+1])
            
        else: 
            point_groups[group_idx].append(points_per_slice[ratio_idx])
            if ratio_idx == len(ibr_ratios) - 1 and len(points_per_slice[ratio_idx+1]) > 0:
                point_groups[group_idx].append(points_per_slice[ratio_idx+1])
    
   
    return point_groups


def build_new_cloud_bbox(group, cloud_object):  
    group = list(filter(lambda x: len(x) > 0, group)) 
    points = np.vstack(group)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color([0, 0, 1])

    # Creating the cloud object and loading the necessary file:
    cloud_object_part = PointCloud()
    cloud_object_part.processed_cloud = pcd
    cloud_object_part.points = np.asarray(cloud_object_part.processed_cloud.points)
    cloud_object_part.compute_bounding_box()
    cloud_object_part.vertices = cloud_object_part.oriented_bounding_box_vertices
    cloud_object_part.gripper_width_tolerance = cloud_object.gripper_width_tolerance
    cloud_object_part.gripper_height_tolerance = cloud_object.gripper_height_tolerance
    cloud_object_part.plot_cube()
    return cloud_object_part

def refit_bbox(cloud_object, point_groups):
    parts = []
    for idx, group in enumerate(point_groups): 
        num_points = np.vstack(group).shape[0]
        if len(group) > 0 and num_points > 100:
            try:
                cloud_object_part = build_new_cloud_bbox(group, cloud_object)
            except: 
                breakpoint()
            parts.append(cloud_object_part)   
    return parts


def calculate_face_area(vertices):
    # Convert vertices to NumPy arrays
    v1 = np.array(vertices[1]) - np.array(vertices[0])
    v2 = np.array(vertices[2]) - np.array(vertices[0])
    v3 = np.array(vertices[3]) - np.array(vertices[0])

    # Calculate the cross products for the two triangles
    cross_product1 = np.cross(v1, v2)
    cross_product2 = np.cross(v2, v3)

    # Calculate the areas of the two triangles
    area = 0.5 * (np.linalg.norm(cross_product1) + np.linalg.norm(cross_product2))

    return area



def get_ibr_ratio(cloud_object, epsilon):
    left_face, right_face, front_face, back_face, top_face, bottom_face = get_faces(cloud_object)
    faces = [left_face, right_face, front_face, back_face, top_face, bottom_face]
    object_frame_points = np.asarray(cloud_object.cloud_object_frame.points)   
    face_distances, unit_vec = get_face_distance(faces, object_frame_points, cloud_object) 

    #1. Select Slicing Dimesions
    boundary_idx = np.where(np.any(face_distances <= epsilon, axis=1))[0]
    interior_idx =np.where(np.any(face_distances > epsilon, axis=1))[0]
    ibr_ratio = len(boundary_idx) / len(interior_idx)
    vertices = cloud_object.oriented_bounding_box_vertices
    grid_vertices = np.hstack([  np.min(vertices, axis=0), np.max(vertices, axis=0) ])
    occupancy_struct = OccupancyStruct(grid_vertices, cloud_object.points, 0.01)
    empty_ratio = occupancy_struct.weighted_empty_percentage  
    ibr_ratio = ibr_ratio * empty_ratio
    
    
    return ibr_ratio  



def fuse_consecutive_boxes(cloud_object_parts, cloud_object, slice_idx, area_threshold=0.8, gripper_length=0.11, resolution=0.01):
    fused_boxes = [cloud_object_parts[0]]
    merged_indices = []
    fused_pointer = 0

    for i in range(1, len(cloud_object_parts)):
        curr_left_face, curr_right_face, curr_front_face, curr_back_face, curr_top_face, curr_bottom_face = get_faces(cloud_object_parts[i])
        prev_left_face, prev_right_face, prev_front_face, prev_back_face, prev_top_face, prev_bottom_face = get_faces(fused_boxes[fused_pointer])  
        prev_points = np.asarray(fused_boxes[fused_pointer].processed_cloud.points)

        if slice_idx == 0:                
            area_diff =  calculate_face_area(curr_back_face) -  calculate_face_area(prev_front_face) 
        elif slice_idx == 1: 
            area_diff =  calculate_face_area(curr_left_face) - calculate_face_area(prev_right_face) 
        elif slice_idx == 2:   
            area_diff = calculate_face_area(curr_bottom_face) - calculate_face_area(prev_top_face)  
   
        if np.abs(area_diff) <= area_threshold: 
            new_points = [np.asarray(cloud_object_parts[i].processed_cloud.points), prev_points] 
            fused_object = build_new_cloud_bbox(new_points, cloud_object)
            # check empty
            empty_ratio, occupancy_struct = get_empty_ratio(fused_object, resolution)
            # if occupancy_struct.occupied_comps == 1 and (fused_object.x_dim < gripper_length or fused_object.y_dim < gripper_length or fused_object.z_dim < gripper_length):           
            #     # Either choose the fused object or just appent the parts
            #     fused_boxes[fused_pointer] = fused_object
            #     merged_indices.append(i-1)
            #     merged_indices.append(i)
            # else: 
            #     fused_boxes.append(cloud_object_parts[i])
            #     fused_pointer += 1
            fused_boxes[fused_pointer] = fused_object
            merged_indices.append(i-1)
            merged_indices.append(i)
        
           
        else: 
            fused_boxes.append(cloud_object_parts[i])
            fused_pointer += 1    
    
    if len(fused_boxes) > 0 and len(fused_boxes) < len(cloud_object_parts):
        new_points = [np.asarray(cloud_object_parts[-1].processed_cloud.points), prev_points] 
        fused_object = build_new_cloud_bbox(new_points, cloud_object)
        fused_boxes[fused_pointer] = fused_object
        merged_indices.append(i-1)
        merged_indices.append(i)

       
    
    return fused_boxes
       
def slice_subparts(part, cloud_object, face_distances, uncheck_idx, thickness, epsilon, theta,  area_threshold, gripper_length, resolution):
    vertices = part.oriented_bounding_box_vertices
    left_face, right_face, front_face, back_face, top_face, bottom_face = get_faces(part)
    faces = [left_face, right_face, front_face, back_face, top_face, bottom_face]
    object_frame_points = np.asarray(part.points)   
    face_distances, unit_vec = get_face_distance(faces, object_frame_points, cloud_object)
    sub_slice_bbox_vertices = slice_bbox(vertices, part, thickness=thickness, slice_idx=uncheck_idx) 
    object_frame_points = np.asarray(part.points)   
    sub_ibr_ratios, empty_ratios, sub_point_groups = accumulate_slices(sub_slice_bbox_vertices, object_frame_points, face_distances, epsilon)
    sub_point_groups                  = fuse_slices(sub_point_groups, sub_ibr_ratios, empty_ratios,  theta, thickness, slice_idx=uncheck_idx, gripper_width=gripper_length)
   
    sub_parts = refit_bbox(cloud_object, sub_point_groups)
    sub_parts = fuse_consecutive_boxes(sub_parts, cloud_object, slice_idx=uncheck_idx, area_threshold=area_threshold, resolution=resolution)
    return sub_parts, sub_point_groups

def get_empty_ratio(part, resolution=0.01):
    vertices = part.oriented_bounding_box_vertices
    grid_vertices = np.hstack([  np.min(vertices, axis=0), np.max(vertices, axis=0) ])
    occupancy_struct = OccupancyStruct(grid_vertices, part.points, resolution)
    empty_ratio = occupancy_struct.weighted_empty_percentage  
    return empty_ratio, occupancy_struct

def oversize_correct(parts, point_groups,ibr_ratios, cloud_object, face_distances, uncheck_idx, thickness, epsilon, theta, base_ibr, area_threshold, gripper_length, resolution=0.01):
   
    fixed_parts = []
    fixed_point_groups = []
    
    for idx, ratio in enumerate(ibr_ratios): 
        empty_ratio, occupancy_struct = get_empty_ratio(parts[idx], resolution)
        dimensions = np.array( [parts[idx].x_dim, parts[idx].y_dim, parts[idx].z_dim ])
        uncheck_dimensions = dimensions[uncheck_idx]
        new_slice_idx = uncheck_idx[np.argmax(uncheck_dimensions)]
      
        if ( (parts[idx].x_dim > gripper_length or parts[idx].y_dim > gripper_length) or occupancy_struct.occupied_comps > 1 or ratio < base_ibr  )    :
            
            sub_parts, sub_point_groups = slice_subparts(parts[idx], cloud_object, face_distances, new_slice_idx, thickness, epsilon, theta, area_threshold, gripper_length, resolution)
            for group in sub_point_groups: 
                fixed_point_groups.append(group)
            for part in sub_parts:
                fixed_parts.append(part)
        else: 
            fixed_parts.append(parts[idx])
            fixed_point_groups.append(point_groups[idx])
       
    
    return fixed_parts, fixed_point_groups


def search_dimesion_slice(cloud_object, gripper_length, ibr_ratio, base_ibr):
    graspable_dim = np.array([cloud_object.x_dim, cloud_object.y_dim, cloud_object.z_dim])
    depth_dim = cloud_object.z_dim
    
    if ibr_ratio > base_ibr: 
        print('Bounding Box is Well Approximated')
        return None, None
    elif np.any(graspable_dim > gripper_length): 
        slice_idx = np.argmax(graspable_dim)
        if slice_idx == 2: 
            uncheck_idx = [0,1]
        else:
            uncheck_idx = [np.int(np.logical_not(np.argmax(graspable_dim)))]
    
        
    else:
        print("Splitting not required")
        return None, None
    return slice_idx, uncheck_idx 

def bitsi_metric (cloud_object, epsilon=1e-3, thickness=0.001, gripper_length=0.13, base_ibr=0.7):
    
    left_face, right_face, front_face, back_face, top_face, bottom_face = get_faces(cloud_object)
    faces = [left_face, right_face, front_face, back_face, top_face, bottom_face]
    object_frame_points = np.asarray(cloud_object.cloud_object_frame.points)   
    face_distances, unit_vec = get_face_distance(faces, object_frame_points, cloud_object) 
    vertices = cloud_object.transformed_vertices_object_frame

    boundary_idx = np.where(np.any(face_distances <= epsilon, axis=1))[0]
    interior_idx =np.where(np.any(face_distances > epsilon, axis=1))[0]
    ibr_ratio = len(boundary_idx) / len(interior_idx)
    
    slice_idx, uncheck_idx  = search_dimesion_slice(cloud_object, gripper_length, ibr_ratio, base_ibr)
       
    slice_bbox_vertices_x = slice_bbox(vertices, cloud_object, thickness=thickness, slice_idx=0)
    slice_bbox_vertices_y = slice_bbox(vertices, cloud_object, thickness=thickness, slice_idx=1)
    slice_bbox_vertices_z = slice_bbox(vertices, cloud_object, thickness=thickness, slice_idx=2)

    ibr_ratios_x, _, _ = accumulate_slices(slice_bbox_vertices_x, object_frame_points, face_distances, epsilon)
    ibr_ratios_y, _, _ = accumulate_slices(slice_bbox_vertices_y, object_frame_points, face_distances, epsilon)
    ibr_ratios_z, _, _ = accumulate_slices(slice_bbox_vertices_z, object_frame_points, face_distances, epsilon)

    return ibr_ratios_x, ibr_ratios_y, ibr_ratios_z, slice_idx


def slice_object_bbox(cloud_object, gripper_length=0.11, epsilon=1e-3, thickness=0.001, theta=0.1, base_ibr=0.7, area_threshold=1e-4, resolution=0.01, check=False, obj_name="" ):
    
    left_face, right_face, front_face, back_face, top_face, bottom_face = get_faces(cloud_object)
    faces = [left_face, right_face, front_face, back_face, top_face, bottom_face]
    object_frame_points = np.asarray(cloud_object.cloud_object_frame.points)   
    face_distances, unit_vec = get_face_distance(faces, object_frame_points, cloud_object) 
    vertices = cloud_object.transformed_vertices_object_frame

    #1. Select Slicing Dimesions
    boundary_idx = np.where(np.any(face_distances <= epsilon, axis=1))[0]
    interior_idx =np.where(np.any(face_distances > epsilon, axis=1))[0]
    ibr_ratio = len(boundary_idx) / len(interior_idx)
    
   
    slice_idx, uncheck_idx  = search_dimesion_slice(cloud_object, gripper_length, ibr_ratio, base_ibr)
    if slice_idx == None: 
        return None, None  
    
    
    
    #1. Calculate Fill Ratio and Collect Slices
    slice_bbox_vertices = slice_bbox(vertices, cloud_object, thickness=thickness, slice_idx=slice_idx) 
    
    
    
    ibr_ratios, empty_ratios, points_per_slice = accumulate_slices(slice_bbox_vertices, object_frame_points, face_distances, epsilon)
    point_groups                  = fuse_slices(points_per_slice, ibr_ratios, empty_ratios, theta, thickness, slice_idx, gripper_length)
    #2. Fit New Bounding Box
    parts = refit_bbox(cloud_object, point_groups)
    #3. Fuse Parts
    parts = fuse_consecutive_boxes(parts, cloud_object, slice_idx, area_threshold=area_threshold)
    ibr_ratios = [ get_ibr_ratio(part, epsilon=epsilon) for part in parts ] 
    #4. Correction Refinment Step
    if slice_idx != 2:
       parts, point_groups = oversize_correct(parts, point_groups,ibr_ratios, cloud_object, face_distances, uncheck_idx, thickness, epsilon, theta, base_ibr, area_threshold, gripper_length, resolution)
    
    return parts, point_groups, slice_bbox_vertices
  



def get_plane_distance(plane_points, center_point, grasp_center):
    # We need to compute the distance of the grasp center to the first plane only:
    vect_1 = np.subtract(plane_points[3, :], plane_points[0, :])
    vect_2 = np.subtract(plane_points[1, :], plane_points[0, :])
    # vect_2 = np.subtract(plane_points[1, :], plane_points[2, :])

    # Normal vector corresponding to the plane:
    n = np.cross(vect_1, vect_2)
    unit_u = np.divide(n, la.norm(n))

    # Translating each of the points by a specified distance: 
    # Computing the D in the equation of the plane Ax + By + Cz + D = 0:
    D = np.dot(center_point, unit_u)

    num = la.norm(unit_u[0]*grasp_center[0] + unit_u[1]*grasp_center[1] + unit_u[2]*grasp_center[2] - D)
    den = np.sqrt(unit_u[0]**2 + unit_u[1]**2 + unit_u[2]**2)
    distance = np.divide(num, den)
    return distance, unit_u



