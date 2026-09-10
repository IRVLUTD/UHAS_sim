#!/usr/bin/env python
import numpy as np
import json
import os 
import tf_ros as tf
import time
from utils_urdf import *
import colorsys
import argparse
import numpy.ma as ma
import trimesh
from urdf_parser_py.urdf import URDF
import urdf_parser_py
import matplotlib.pyplot as plt
import collections
try:
    collections.Iterable
except AttributeError:
    import collections.abc
    collections.Iterable = collections.abc.Iterable

import pyvista as pv


def visualize_meshes_and_vertices(meshes_dict, joint_info, kinematic_chains, vertices_dict, q=None, title="Meshes and Color-coded Vertices", base_link="base_link"):
    # Build parent map for traversing kinematic chains
    parent_map = {}
    for chain in kinematic_chains:
        for i in range(1, len(chain)):
            parent_map[chain[i]] = chain[i - 1]
    
    # Function to compute transformation from base to a joint
    def compute_transform_to_base(joint_name):
        if joint_name == base_link:
            return np.eye(4)
        transforms = []
        current_joint = joint_name
        while current_joint != base_link:
            xyz = joint_info[current_joint][0]  # Position
            quat = joint_info[current_joint][1]  # Quaternion
            joint_type = joint_info[current_joint][2]  # Joint type
            T = tf.quaternion_matrix(quat)  # Convert quaternion to 4x4 matrix
            T[:3, 3] = xyz  # Set translation
            if q is not None and joint_type == "revolute":
                q_val = q.get(current_joint, 0.0)  # Joint angle or default to 0
                R_z = tf.rotation_matrix(q_val, [0, 0, 1])  # Rotation around z-axis
                T = T @ R_z  # Apply joint rotation
            transforms.append(T)
            current_joint = parent_map[current_joint]
        T_base_to_joint = np.eye(4)
        for T in reversed(transforms):
            T_base_to_joint = T_base_to_joint @ T  # Chain transformations
        return T_base_to_joint
    
    # Initialize PyVista plotter
    plotter = pv.Plotter()
    legend_data = []
    
    # Define default colors for joints
    default_colors = ['red', 'green', 'blue', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan',
                      'magenta', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'coral', 'gold', 'indigo']
    
    # Create color map for vertices based on parent frame
    color_map = {joint: default_colors[i % len(default_colors)] for i, joint in enumerate(vertices_dict.keys())}
    
    # Visualize meshes in gray with transparency
    for joint_name in meshes_dict.keys():
        if joint_name not in joint_info:
            continue
        T_base_to_joint = compute_transform_to_base(joint_name)
        for mesh in meshes_dict[joint_name]:
            # Convert trimesh to PyVista PolyData
            vertices = mesh.vertices
            faces = mesh.faces
            # PyVista requires faces to be prefixed with the number of vertices per face (3 for triangles)
            pv_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()
            pv_mesh = pv.PolyData(vertices, faces=pv_faces)
            # Apply transformation with explicit inplace=True
            pv_mesh.transform(T_base_to_joint, inplace=True)
            # Add to plotter in gray with transparency
            plotter.add_mesh(pv_mesh, color='lightgray', opacity=1.0, show_edges=True)
    
    # Visualize vertices with colors based on parent frame
    for joint_name, vertices in vertices_dict.items():
        if vertices.size == 0:
            print(f"Skipping {joint_name}: empty vertices array")
            continue
        T_base_to_joint = compute_transform_to_base(joint_name)
        homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        base_vertices = (T_base_to_joint @ homogeneous_vertices.T).T[:, :3]
        cloud = pv.PolyData(base_vertices)
        plotter.add_points(cloud, color=color_map[joint_name], point_size=5)
        legend_data.append((joint_name, color_map[joint_name]))
    
    # Add legend if there are entries
    # if legend_data:
    #     plotter.add_legend(legend_data, loc="lower right", size=[0.3, 0.3])
    
    # Add title and display
    plotter.add_title(title)
    plotter.show()

def visualize_meshes_and_kinematic_lines(meshes_dict, joint_info, kinematic_chains, q=None, title="Meshes and Color-coded Kinematic Lines", base_link="base_link", palm_frame="palm_normal"):
    # Build parent map for traversing kinematic chains
    parent_map = {}
    for chain in kinematic_chains:
        for i in range(1, len(chain)):
            parent_map[chain[i]] = chain[i - 1]
    
    # Function to compute transformation from base to a joint
    def compute_transform_to_base(joint_name):
        if joint_name == base_link:
            return np.eye(4)
        transforms = []
        current_joint = joint_name
        while current_joint != base_link:
            xyz = joint_info[current_joint][0]  # Position
            quat = joint_info[current_joint][1]  # Quaternion
            joint_type = joint_info[current_joint][2]  # Joint type
            T = tf.quaternion_matrix(quat)  # Convert quaternion to 4x4 matrix
            T[:3, 3] = xyz  # Set translation
            if q is not None and joint_type == "revolute":
                q_val = q.get(current_joint, 0.0)  # Joint angle or default to 0
                R_z = tf.rotation_matrix(q_val, [0, 0, 1])  # Rotation around z-axis
                T = T @ R_z  # Apply joint rotation
            transforms.append(T)
            current_joint = parent_map[current_joint]
        T_base_to_joint = np.eye(4)
        for T in reversed(transforms):
            T_base_to_joint = T_base_to_joint @ T  # Chain transformations
        return T_base_to_joint
    
    # Initialize PyVista plotter
    plotter = pv.Plotter()
    legend_data = []
    
    # Define default colors for joints
    default_colors = default_colors = [
    '#8B0000',  # Dark Red
    '#006400',  # Dark Green
    '#00008B',  # Dark Blue
    '#FFD700',  # Gold (kept for contrast, still dark enough)
    '#4B0082',  # Indigo
    '#008B8B',  # Dark Cyan
    '#8A2BE2',  # Blue Violet
    '#228B22',  # Forest Green
    '#4169E1',  # Royal Blue
    '#A52A2A',  # Brown
    '#9932CC',  # Dark Orchid
    '#2F4F4F',  # Dark Slate Gray
    '#9400D3',  # Dark Violet
    '#006666',  # Deep Teal
    '#8B008B',  # Dark Magenta
    '#556B2F',  # Dark Olive Green
    '#483D8B',  # Dark Slate Blue
    '#8B4513',  # Saddle Brown
    '#2E8B57',  # Sea Green
    '#4682B4'   # Steel Blue
    ]
    # Create color map for joints based on all joints in kinematic chains (excluding chain[0])
    all_joints = set()
    for chain in kinematic_chains:
        all_joints.update(chain[1:])  # Skip chain[0]
    if palm_frame in joint_info:
        all_joints.add(palm_frame)
    color_map = {joint: default_colors[i % len(default_colors)] for i, joint in enumerate(sorted(all_joints))}
    
    # Visualize meshes in gray with transparency
    for joint_name in meshes_dict.keys():
        if joint_name not in joint_info:
            continue
        T_base_to_joint = compute_transform_to_base(joint_name)
        for mesh in meshes_dict[joint_name]:
            # Convert trimesh to PyVista PolyData
            vertices = mesh.vertices
            faces = mesh.faces
            # PyVista requires faces to be prefixed with the number of vertices per face (3 for triangles)
            pv_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()
            pv_mesh = pv.PolyData(vertices, faces=pv_faces)
            # Apply transformation with explicit inplace=True
            pv_mesh.transform(T_base_to_joint, inplace=True)
            # Add to plotter in gray with transparency
            plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.3, show_edges=True)
    
    # Get the position of the palm_normal frame origin
    if palm_frame not in joint_info:
        print(f"Warning: {palm_frame} not found in joint_info. Skipping kinematic lines.")
        palm_pos = None
    else:
        palm_pos = compute_transform_to_base(palm_frame)[:3, 3]
    
    if palm_pos is not None:
        # Visualize line segments for each kinematic chain starting from palm_normal to chain[1]
        for idx, chain in enumerate(kinematic_chains):
            if len(chain) < 2:  # Need at least chain[1] to draw a line from palm_normal
                print(f"Skipping chain {idx}: insufficient joints (need at least 2)")
                continue
            # Start with palm_normal to chain[1]
            points = [palm_pos]
            joint_names = [palm_frame]
            for joint in chain[1:]:  # Start from chain[1], skip chain[0]
                if joint not in joint_info:
                    print(f"Skipping joint {joint}: not in joint_info")
                    continue
                joint_pos = compute_transform_to_base(joint)[:3, 3]
                points.append(joint_pos)
                joint_names.append(joint)
            
            if len(points) < 2:
                print(f"Skipping chain {idx}: insufficient points after processing")
                continue
            
            # Create individual line segments with colors based on starting joint
            for i in range(len(points) - 1):
                segment_points = np.array([points[i], points[i + 1]])
                start_joint = joint_names[i]
                color = color_map.get(start_joint, 'black')
                # Create a PolyData for the line segment
                line = pv.PolyData()
                line.points = segment_points
                # Define the line connectivity (single line with 2 points)
                line.lines = np.array([2, 0, 1])
                # Add to plotter
                plotter.add_mesh(line, color=color, line_width=20, label=start_joint)
                if (start_joint, color) not in legend_data:
                    legend_data.append((start_joint, color))
    
    # Add legend if there are entries
    # if legend_data:
    #     plotter.add_legend(legend_data, loc="lower right", size=[0.3, 0.3])
    
    # Add title and display
    plotter.add_title(title)
    plotter.show()

def visualize_meshes_and_fingerprints_screenshot(
    meshes_dict,
    joint_info,
    kinematic_chains,
    finger_print_dict,
    q=None,
    title="Meshes and Color-coded Fingerprints",
    base_link="base_link",
    theta_values=None,
    phi_values=None,
    camera_position=None,
    focal_point=None,
    view_up=None,
    screenshot_path=None
):
    # Build parent map for traversing kinematic chains
    parent_map = {}
    for chain in kinematic_chains:
        for i in range(1, len(chain)):
            parent_map[chain[i]] = chain[i - 1]
    
    # Function to compute transformation from base to a joint
    def compute_transform_to_base(joint_name):
        if joint_name == base_link:
            return np.eye(4)
        transforms = []
        current_joint = joint_name
        while current_joint != base_link:
            xyz = joint_info[current_joint][0]  # Position
            quat = joint_info[current_joint][1]  # Quaternion
            joint_type = joint_info[current_joint][2]  # Joint type
            T = tf.quaternion_matrix(quat)  # Convert quaternion to 4x4 matrix
            T[:3, 3] = xyz  # Set translation
            if q is not None and joint_type == "revolute":
                q_val = q.get(current_joint, 0.0)  # Joint angle or default to 0
                R_z = tf.rotation_matrix(q_val, [0, 0, 1])  # Rotation around z-axis
                T = T @ R_z  # Apply joint rotation
            transforms.append(T)
            current_joint = parent_map[current_joint]
        T_base_to_joint = np.eye(4)
        for T in reversed(transforms):
            T_base_to_joint = T_base_to_joint @ T  # Chain transformations
        return T_base_to_joint
    
    # Initialize PyVista plotter
    plotter = pv.Plotter()
    
    # Visualize meshes in gray with transparency
    for joint_name in meshes_dict.keys():
        if joint_name not in joint_info:
            continue
        T_base_to_joint = compute_transform_to_base(joint_name)
        for mesh in meshes_dict[joint_name]:
            # Convert trimesh to PyVista PolyData
            vertices = mesh.vertices
            faces = mesh.faces
            pv_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()
            pv_mesh = pv.PolyData(vertices, faces=pv_faces)
            pv_mesh.transform(T_base_to_joint, inplace=True)
            plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.8, show_edges=True)
    
    # Collect all fingerprint points and colors in base frame
    all_base_points = []
    all_rgbs = []
    for joint_name, data in finger_print_dict.items():
        if "points" not in data or data["points"].size == 0 or "theta" not in data or "phi" not in data:
            print(f"Skipping {joint_name}: missing or empty 'points', 'theta', or 'phi'")
            continue
        points = data["points"]
        theta = (data["theta"]) % (2 * np.pi)
        phi = data["phi"] % (np.pi)
        if points.shape[0] != theta.shape[0] or points.shape[0] != phi.shape[0]:
            print(f"Skipping {joint_name}: mismatched array sizes (points: {points.shape[0]}, theta: {theta.shape[0]}, phi: {phi.shape[0]})")
            continue
        T_base_to_joint = compute_transform_to_base(joint_name)
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        base_points = (T_base_to_joint @ homogeneous_points.T).T[:, :3]
        hues = theta / (2 * np.pi)
        values = 1 - (phi / np.pi) * 0.5
        delta_theta = np.deg2rad(1)
        delta_phi = np.deg2rad(1)
        if theta_values is not None:
            theta_values = np.asarray(theta_values)
            theta_diff = np.abs(theta[:, np.newaxis] - theta_values[np.newaxis, :])
            theta_diff = np.minimum(theta_diff, 2 * np.pi - theta_diff)
            min_dists = np.min(theta_diff, axis=1)
            is_close_theta = min_dists <= delta_theta
            values[~is_close_theta] /= 5.0
        rgbs = np.array([colorsys.hsv_to_rgb(h, 1.0, v) for h, v in zip(hues, values)])
        if phi_values is not None:
            phi_values = np.asarray(phi_values)
            phi_diff = np.abs(phi[:, np.newaxis] - phi_values[np.newaxis, :])
            min_phi_dists = np.min(phi_diff, axis=1)
            is_close_phi = min_phi_dists <= delta_phi
            mask = is_close_theta & is_close_phi if theta_values is not None else is_close_phi
            rgbs[mask] = [1, 1, 1]
        all_base_points.append(base_points)
        all_rgbs.append(rgbs)
    
    # Add all points with their colors if any
    if all_base_points and all_rgbs:
        all_points = np.vstack(all_base_points)
        all_colors = np.vstack(all_rgbs)
        print(f"Plotting {all_points.shape[0]} points with colors shape {all_colors.shape}")
        if all_colors.shape[1] != 3:
            raise ValueError(f"Expected RGB colors with shape (n, 3), got shape {all_colors.shape}")
        cloud = pv.PolyData(all_points)
        plotter.add_points(cloud, scalars=all_colors, point_size=5, rgb=True)
    else:
        print("No valid fingerprint points to plot")
    
    # Set camera position if provided
    if camera_position is not None and focal_point is not None and view_up is not None:
        plotter.camera_position = (camera_position, focal_point, view_up)
    
    # Add title
    plotter.add_title(title)

    plotter.show(auto_close = True)
    
    # Save screenshot if path is provided
    if screenshot_path is not None:
        plotter.screenshot(screenshot_path)
    
    # Display plot
    

def visualize_meshes_in_base_frame(meshes_dict, joint_info, kinematic_chains, q=None, title="Gripper Mesh Visualization", base_link="base_link", color_map=None):
    # Build parent map for traversing kinematic chains
    parent_map = {}
    for chain in kinematic_chains:
        for i in range(1, len(chain)):
            parent_map[chain[i]] = chain[i - 1]

    # Function to compute transformation from base to a joint
    def compute_transform_to_base(joint_name):
        if joint_name == base_link:
            return np.eye(4)
        transforms = []
        current_joint = joint_name
        while current_joint != base_link:
            xyz = joint_info[current_joint][0]  # Position
            quat = joint_info[current_joint][1]  # Quaternion
            joint_type = joint_info[current_joint][2]  # Joint type
            T = tf.quaternion_matrix(quat)  # Convert quaternion to 4x4 matrix
            T[:3, 3] = xyz  # Set translation
            if q is not None and joint_type == "revolute":
                q_val = q.get(current_joint, 0.0)  # Joint angle or default to 0
                R_z = tf.rotation_matrix(q_val, [0, 0, 1])  # Rotation around z-axis
                T = T @ R_z  # Apply joint rotation
            transforms.append(T)
            current_joint = parent_map[current_joint]
        T_base_to_joint = np.eye(4)
        for T in reversed(transforms):
            T_base_to_joint = T_base_to_joint @ T  # Chain transformations
        return T_base_to_joint

    # Initialize PyVista plotter
    plotter = pv.Plotter()
    legend_data = []

    # Define default colors for joints
    default_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan',
                      'magenta', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'coral', 'gold', 'indigo']

    # Create color map if not provided
    if color_map is None:
        joint_names = list(meshes_dict.keys())
        color_map = {joint: default_colors[i % len(default_colors)] for i, joint in enumerate(joint_names)}

    # Add each mesh with its corresponding color
    for joint_name in meshes_dict.keys():
        if joint_name not in joint_info:
            continue
        T_base_to_joint = compute_transform_to_base(joint_name)
        for mesh in meshes_dict[joint_name]:
            # Convert trimesh to PyVista PolyData
            vertices = mesh.vertices
            faces = mesh.faces
            # PyVista requires faces to be prefixed with the number of vertices per face (3 for triangles)
            pv_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()
            pv_mesh = pv.PolyData(vertices, faces=pv_faces)
            # Apply transformation with explicit inplace=True
            pv_mesh.transform(T_base_to_joint, inplace=True)
            # Add to plotter
            plotter.add_mesh(pv_mesh, color=color_map[joint_name], show_edges=True)
            legend_data.append((joint_name, color_map[joint_name]))

    # Add legend if there are entries
    if legend_data:
        plotter.add_legend(legend_data, loc="lower right", size=[0.3, 0.3])

    # Add title and display
    plotter.add_title(title)
    plotter.show()

def visualize_vertices_in_base_frame(vertices_dict, joint_info, kinematic_chains, q=None, title="Gripper at q_0 configuration", base_link="base_link", color_map = None):
    # Build parent map for traversing kinematic chains
    parent_map = {}
    for chain in kinematic_chains:
        for i in range(1, len(chain)):
            parent_map[chain[i]] = chain[i - 1]

    # Function to compute transformation from base to a joint
    def compute_transform_to_base(joint_name):
        if joint_name == base_link:
            return np.eye(4)
        transforms = []
        current_joint = joint_name
        while current_joint != base_link:
            xyz = joint_info[current_joint][0]  # Position
            quat = joint_info[current_joint][1]  # Quaternion
            joint_type = joint_info[current_joint][2]  # Joint type
            T = tf.quaternion_matrix(quat)  # Convert quaternion to 4x4 matrix
            T[:3, 3] = xyz  # Set translation
            if q is not None and joint_type == "revolute":
                q_val = q.get(current_joint, 0.0)  # Joint angle or default to 0
                R_z = tf.rotation_matrix(q_val, [0, 0, 1])  # Rotation around z-axis
                T = T @ R_z  # Apply joint rotation
            transforms.append(T)
            current_joint = parent_map[current_joint]
        T_base_to_joint = np.eye(4)
        for T in reversed(transforms):
            T_base_to_joint = T_base_to_joint @ T  # Chain transformations
        return T_base_to_joint

    # Initialize PyVista plotter
    plotter = pv.Plotter()
    legend_data = []

    # Define default colors for joints
    default_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan',
                      'magenta', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'coral', 'gold', 'indigo']

    # Create color map if not provided
    if color_map is None:
        joint_names = list(vertices_dict.keys())
        color_map = {joint: default_colors[i % len(default_colors)] for i, joint in enumerate(joint_names)}

    # Add each vertex set with its corresponding color
    for joint_name, vertices in vertices_dict.items():
        if vertices.size == 0:
            continue
        T_base_to_joint = compute_transform_to_base(joint_name)
        homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        base_vertices = (T_base_to_joint @ homogeneous_vertices.T).T[:, :3]
        cloud = pv.PolyData(base_vertices)
        plotter.add_points(cloud, color=color_map[joint_name], point_size=5)
        legend_data.append((joint_name, color_map[joint_name]))

    # Add legend if there are entries
    if legend_data:
        plotter.add_legend(legend_data, loc="lower right",size =[0.3,0.3])

    # Add title and display
    plotter.add_title(title)
    plotter.show()

def visualize_7point_fingerprint(
    meshes_dict,
    joint_info,
    kinematic_chains,
    finger_print_dict,           # Rich dict: {joint: {'points': array, 'theta': array, 'phi': array}}
    q=None,
    title="7-Point Fingerprint Sample",
    base_link="base_link",
    highlight_joint="joint_0",           # Chain containing this joint will be highlighted
    highlight_mesh_opacity=0.70,
    theta_values=None,
    phi_values=None,
    show_meshes=True,
    mesh_opacity=0.15,
    point_size=2000,
    render_as_spheres=True,
    point_opacity=1.0
):
    """
    Specialized visualization for 7-point fingerprint samples.
    - Only shows the LAST point of the last joint from the highlighted chain
    - That chain's meshes are more opaque
    - All other chains stay very transparent
    - base_link is always omitted
    """

    # === Find the chain that contains the highlight joint ===
    highlight_chain = None
    for chain in kinematic_chains:
        if highlight_joint in chain:
            highlight_chain = set(chain)
            break

    if highlight_chain is None:
        print(f"Warning: No chain found containing '{highlight_joint}'. Showing nothing.")
        highlight_chain = set()

    # Build parent map
    parent_map = {}
    for chain in kinematic_chains:
        for i in range(1, len(chain)):
            parent_map[chain[i]] = chain[i - 1]

    def compute_transform_to_base(joint_name):
        if joint_name == base_link:
            return np.eye(4)
        transforms = []
        current_joint = joint_name
        while current_joint != base_link:
            xyz = joint_info[current_joint][0]
            quat = joint_info[current_joint][1]
            joint_type = joint_info[current_joint][2]
            T = tf.quaternion_matrix(quat)
            T[:3, 3] = xyz
            if q is not None and joint_type == "revolute":
                q_val = q.get(current_joint, 0.0)
                R_z = tf.rotation_matrix(q_val, [0, 0, 1])
                T = T @ R_z
            transforms.append(T)
            current_joint = parent_map[current_joint]
        T_base_to_joint = np.eye(4)
        for T in reversed(transforms):
            T_base_to_joint = T_base_to_joint @ T
        return T_base_to_joint

    plotter = pv.Plotter()

    # === Visualize meshes (only highlighted chain is more opaque) ===
    if show_meshes:
        for joint_name in meshes_dict.keys():
            if joint_name not in joint_info:
                continue

            current_opacity = highlight_mesh_opacity if joint_name in highlight_chain else mesh_opacity

            T_base_to_joint = compute_transform_to_base(joint_name)
            for mesh in meshes_dict[joint_name]:
                vertices = mesh.vertices
                faces = mesh.faces
                pv_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()
                pv_mesh = pv.PolyData(vertices, faces=pv_faces)
                pv_mesh.transform(T_base_to_joint, inplace=True)
                plotter.add_mesh(
                    pv_mesh,
                    color='lightgray',
                    opacity=current_opacity,
                    show_edges=True
                )

    # === Collect fingerprint points (only from highlighted chain) ===
    all_base_points = []
    all_rgbs = []

    for joint_name, data in finger_print_dict.items():
        if joint_name == base_link:
            continue
        if joint_name not in highlight_chain:
            continue

        if "points" not in data or data["points"].size == 0 or "theta" not in data or "phi" not in data:
            continue

        points = data["points"]
        theta = (data["theta"]) % (2 * np.pi)
        phi = data["phi"] % (np.pi)

        if points.shape[0] != theta.shape[0] or points.shape[0] != phi.shape[0]:
            continue

        T_base_to_joint = compute_transform_to_base(joint_name)
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        base_points = (T_base_to_joint @ homogeneous_points.T).T[:, :3]

        # Color computation
        hues = theta / (2 * np.pi)
        values = 1 - (phi / np.pi) * 0.5

        delta_theta = np.deg2rad(1)
        delta_phi = np.deg2rad(1)

        if theta_values is not None:
            theta_values = np.asarray(theta_values)
            theta_diff = np.abs(theta[:, np.newaxis] - theta_values[np.newaxis, :])
            theta_diff = np.minimum(theta_diff, 2 * np.pi - theta_diff)
            min_dists = np.min(theta_diff, axis=1)
            is_close_theta = min_dists <= delta_theta
            values[~is_close_theta] /= 5.0

        rgbs = np.array([colorsys.hsv_to_rgb(0, 1.0, 1.0) for h, v in zip(hues, values)])

        if phi_values is not None:
            phi_values = np.asarray(phi_values)
            phi_diff = np.abs(phi[:, np.newaxis] - phi_values[np.newaxis, :])
            min_phi_dists = np.min(phi_diff, axis=1)
            is_close_phi = min_phi_dists <= delta_phi
            if theta_values is not None:
                mask = is_close_theta & is_close_phi
            else:
                mask = is_close_phi
            rgbs[mask] = [1, 1, 1]

        all_base_points.append(base_points)
        all_rgbs.append(rgbs)

    # === Show ONLY the last point of the last joint ===
    if all_base_points and all_rgbs:
        all_points = np.vstack(all_base_points)
        all_colors = np.vstack(all_rgbs)

        # Keep only the very last point (last point of the last joint in the chain)
        last_point = all_points[-1:]
        last_color = all_colors[-1:]

        print(f"Plotting 1 fingerprint point (last point of chain with '{highlight_joint}')")

        cloud = pv.PolyData(last_point)
        plotter.add_points(
            cloud,
            scalars=last_color,
            point_size=point_size,
            rgb=True,
            render_points_as_spheres=render_as_spheres,
            opacity=point_opacity
        )
    else:
        print("No valid fingerprint points found for the highlighted chain")

    plotter.add_title(title)
    plotter.show()

def visualize_meshes_and_fingerprints(meshes_dict, joint_info, kinematic_chains, finger_print_dict, q=None, title="Meshes and Color-coded Fingerprints", base_link="base_link", theta_values=None, phi_values=None):

    # Build parent map for traversing kinematic chains
    parent_map = {}
    for chain in kinematic_chains:
        for i in range(1, len(chain)):
            parent_map[chain[i]] = chain[i - 1]
    # Function to compute transformation from base to a joint
    def compute_transform_to_base(joint_name):
        if joint_name == base_link:
            return np.eye(4)
        transforms = []
        current_joint = joint_name
        while current_joint != base_link:
            xyz = joint_info[current_joint][0] # Position
            quat = joint_info[current_joint][1] # Quaternion
            joint_type = joint_info[current_joint][2] # Joint type
            T = tf.quaternion_matrix(quat) # Convert quaternion to 4x4 matrix
            T[:3, 3] = xyz # Set translation
            if q is not None and joint_type == "revolute":
                q_val = q.get(current_joint, 0.0) # Joint angle or default to 0
                R_z = tf.rotation_matrix(q_val, [0, 0, 1]) # Rotation around z-axis
                T = T @ R_z # Apply joint rotation
            transforms.append(T)
            current_joint = parent_map[current_joint]
        T_base_to_joint = np.eye(4)
        for T in reversed(transforms):
            T_base_to_joint = T_base_to_joint @ T # Chain transformations
        return T_base_to_joint
    # Initialize PyVista plotter
    plotter = pv.Plotter()
    # Visualize meshes in gray with transparency
    for joint_name in meshes_dict.keys():
        if joint_name not in joint_info:
            continue
        T_base_to_joint = compute_transform_to_base(joint_name)
        for mesh in meshes_dict[joint_name]:
            # Convert trimesh to PyVista PolyData
            vertices = mesh.vertices
            faces = mesh.faces
            # PyVista requires faces to be prefixed with the number of vertices per face (3 for triangles)
            pv_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()
            pv_mesh = pv.PolyData(vertices, faces=pv_faces)
            # Apply transformation with explicit inplace=True
            pv_mesh.transform(T_base_to_joint, inplace=True)
            # Add to plotter in gray with transparency
            plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.8, show_edges=True)
    # Collect all fingerprint points and colors in base frame
    all_base_points = []
    all_rgbs = []
    for joint_name, data in finger_print_dict.items():
        if "points" not in data or data["points"].size == 0 or "theta" not in data or "phi" not in data:
            print(f"Skipping {joint_name}: missing or empty 'points', 'theta', or 'phi'")
            continue
        points = data["points"]
        theta = (data["theta"]) % (2*np.pi)
        phi = data["phi"] % (np.pi)
        # Ensure arrays are aligned
        if points.shape[0] != theta.shape[0] or points.shape[0] != phi.shape[0]:
            print(f"Skipping {joint_name}: mismatched array sizes (points: {points.shape[0]}, theta: {theta.shape[0]}, phi: {phi.shape[0]})")
            continue
        T_base_to_joint = compute_transform_to_base(joint_name)
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        base_points = (T_base_to_joint @ homogeneous_points.T).T[:, :3]
        # Compute hues and values
        hues = theta / (2 * np.pi) # Normalize theta to [0, 1] for hue
        values = 1 - (phi / np.pi) * 0.5 # Normalize phi to [0, 1] for intensity
        # Apply theta filter if provided
        delta_theta = np.deg2rad(1)
        delta_phi = np.deg2rad(1)  # Assuming same delta for phi as for theta
        if theta_values is not None:
            theta_values = np.asarray(theta_values)
            theta_diff = np.abs(theta[:, np.newaxis] - theta_values[np.newaxis, :])
            theta_diff = np.minimum(theta_diff, 2 * np.pi - theta_diff)
            min_dists = np.min(theta_diff, axis=1)
            is_close_theta = min_dists <= delta_theta
            values[~is_close_theta] /= 5.0
        # Compute RGB colors
        rgbs = np.array([colorsys.hsv_to_rgb(h, 1.0, v) for h, v in zip(hues, values)])
        # Apply phi filter if provided, turning qualifying points white
        if phi_values is not None:
            phi_values = np.asarray(phi_values)
            phi_diff = np.abs(phi[:, np.newaxis] - phi_values[np.newaxis, :])
            min_phi_dists = np.min(phi_diff, axis=1)
            is_close_phi = min_phi_dists <= delta_phi
            if theta_values is not None:
                mask = is_close_theta & is_close_phi
            else:
                mask = is_close_phi
            rgbs[mask] = [1, 1, 1]
        all_base_points.append(base_points)
        all_rgbs.append(rgbs)
    # Add all points with their colors if any
    if all_base_points and all_rgbs:
        all_points = np.vstack(all_base_points)
        all_colors = np.vstack(all_rgbs)
        print(f"Plotting {all_points.shape[0]} points with colors shape {all_colors.shape}")
        if all_colors.shape[1] != 3:
            raise ValueError(f"Expected RGB colors with shape (n, 3), got shape {all_colors.shape}")
        cloud = pv.PolyData(all_points)
        plotter.add_points(cloud, scalars=all_colors, point_size=10, rgb=True)
    else:
        print("No valid fingerprint points to plot")
    # Add title and display
    plotter.add_title(title)
    plotter.show()

def load_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def axis_to_z_transform(axis):
    """
    Compute a 4x4 transformation matrix to align a given axis with [0, 0, 1]. 
    Get the transform to move the +z axis to the current joint axis
    
    Parameters:
    - axis (list or np.ndarray): 3-element vector representing the joint axis (e.g., [1, 0, 0]).
    
    Returns:
    - np.ndarray: 4x4 transformation matrix aligning the axis to [0, 0, 1].
    """
    # Convert axis to numpy array and ensure it's a 3-element vector
    axis = np.array(axis, dtype=np.float64)
    if len(axis) != 3:
        raise ValueError("Axis must have exactly 3 elements")

    # Normalize axis
    mag = np.linalg.norm(axis)
    if mag == 0:
        raise ValueError("Axis vector cannot be zero")
    axis = axis / mag  # Now axis is a unit vector

    if np.allclose(axis, [0, 0, 1]):
        T_z = np.eye(4, dtype=np.float64) # +z
    elif np.allclose(axis, [0, 0, -1]): # -z = +180 in x
        T_z = np.array([[1,0,0,0],
                         [0,-1,0,0],
                         [0,0,-1,0],
                         [0,0,0,1]], dtype=np.float64)
    elif np.allclose(axis, [0, 1, 0]): # + y = -90 in x
        T_z = np.array([[1,0,0,0],
                         [0,0,1,0],
                         [0,-1,0,0],
                         [0,0,0,1]], dtype=np.float64)
    elif np.allclose(axis, [0, -1, 0]): # -y = +90 in x
        T_z = np.array([[1, 0, 0,0],
                         [0, 0, -1,0],
                         [0,1, 0,0],
                         [0, 0, 0,1]], dtype=np.float64)
    elif np.allclose(axis, [1, 0, 0]): # + x = +90 in y
        T_z = np.array([[0, 0, 1,0],
                         [0, 1, 0,0],
                         [-1, 0, 0,0],
                         [0, 0, 0,1]],dtype=np.float64)
    elif np.allclose(axis, [-1, 0, 0]): # -x = -90 in y
        T_z = np.array([[0, 0, -1,0],
                         [0, 1, 0,0],
                         [1,0, 0,0],
                         [0, 0, 0,1]],dtype=np.float64)
    else: # Perform 2 axis rotation torwards z
        min_args = np.argsort(axis)
        print(f" Initial axis {axis}")

        # First alling z-axis to axis z-y projection 
        proj = np.array([axis[1], axis[2]])
        proj = proj/ np.linalg.norm(proj)
        angle = np.arctan2(-proj[0], proj[1])
        R = np.array([[1,             0,              0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle), np.cos(angle)]],dtype=np.float64)
        
        # Update new location of axis 
        R_inv = np.linalg.inv(R)
        new_axis = R_inv @ axis # y will always be 0 
        print(f"{new_axis} intermediate axis {angle} rotation" )
        
        # Perform a subsequent rotation on y
        angle = np.arctan2(new_axis[0], new_axis[2])
        R2 = np.array([[np.cos(angle),  0, np.sin(angle)],
                       [0,              1,             0],
                       [-np.sin(angle), 0, np.cos(angle)]],dtype=np.float64)
        R = R @ R2

        T_z = np.eye(4)
        T_z[:3,:3] = R

    R = T_z[:3, :3]
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    rotated_z_axis = R @ z_axis

    assert(np.allclose(axis, rotated_z_axis)), f"Joint axis transformation to z-axis failed  {rotated_z_axis} == {axis}"

    return T_z

def load_urdf(robot_dir, verbose = False):
    """ Load urdf using urdf parser"""
    # Load the URDF
    robot = URDF.from_xml_file(robot_dir)

    # Iterating over joints to get their initial transforms
    if (verbose):
        for joint in robot.joints:
            parent = joint.parent
            child = joint.child
            transform = joint.origin
            
            print(f"Joint {joint.name}:")
            print(f"  Parent: {parent}")
            print(f"  Child: {child}")
            print(f"  Transform:\n{transform}")
            #print(robot.joint_map)
            print()
    return robot

def visualize_vertices_in_base_frame_old(vertices_dict, joint_info, kinematic_chains, q = None, title= "Gripper at q_0 configuration", base_link = "base_link"):
    """
    Plots all vertices from vertices_dict in the base_link frame, assuming joints are at zero configuration.

    Parameters:
    - vertices_dict (dict): Maps joint names (or 'base_link') to vertex arrays (Nx3).
    - joint_info (dict): Maps joint names to (xyz, quat, joint_type, axis, lower, upper, length).
    - kinematic_chains (list of lists): Defines the hierarchy of joints from base_link to end effectors.
    - q (dict): Dictionary containing the joint values to plot the hand with.
    """
    # Build a parent map from kinematic chains
    parent_map = {}
    for chain in kinematic_chains:
        for i in range(1, len(chain)):
            parent_map[chain[i]] = chain[i - 1]

    # I q is provided, ensure all joints in kinematic_chains are included
    if q is not None:
        all_joints = set()
        for chain in kinematic_chains:
            all_joints.update(chain[1:-1])  # Exclude base_link
        missing_joints = all_joints - set(q.keys())
        if missing_joints:
            raise ValueError(f"Missing q_0 values for joints: {missing_joints}")
        q_vis = q.copy()
    else:
        print("No joint values given, showing robot hand at q = 0.")

    # Compute cumulative transformation from base_link to a joint
    def compute_transform_to_base(joint_name):
        """Returns the 4x4 transformation matrix from base_link to the specified joint."""
        if joint_name == base_link:
            return np.eye(4)  # Identity matrix for base_link

        # Trace back from joint to base_link, collecting transformations
        transforms = []
        current_joint = joint_name
        while current_joint != base_link:
            # Get transformation data from joint_info
            xyz = joint_info[current_joint][0]  # Translation vector
            quat = joint_info[current_joint][1]  # Quaternion (x, y, z, w)
            joint_type = joint_info[current_joint][2]  # Joint type (e.g., "revolute")
            # Create 4x4 transformation matrix from quaternion and translation
            T = tf.quaternion_matrix(quat)  # Rotation matrix
            T[:3, 3] = xyz  # Add translation
            # If q_0_dict is provided and joint is revolute, add z-axis rotation by q_0
            if q_vis is not None and joint_type == "revolute":
                upper = joint_info[current_joint][5]
                lower = joint_info[current_joint][4]
                q_0 = q_vis.get(current_joint, 0.0)  # Default to 0 if not specified
                q_0 = np.clip(q_0,lower,upper)
                q_vis[current_joint] = q_0
                # Create rotation matrix around z-axis by q_0
                R_z = tf.rotation_matrix(q_0, [0, 0, 1])
                # Apply rotation after the joint's base transformation
                T = T @ R_z
            transforms.append(T)
            current_joint = parent_map[current_joint]

        # Multiply transformations from base_link to joint (in reverse order)
        T_base_to_joint = np.eye(4)
        for T in reversed(transforms):
            T_base_to_joint = T_base_to_joint @ T
        return T_base_to_joint

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab20(np.linspace(0, 1, len(vertices_dict)))  # Unique colors for each joint

    all_base_vertices = []
    for i, (joint_name, vertices) in enumerate(vertices_dict.items()):
        if vertices.size == 0:
            continue  # Skip empty vertex arrays

        # Compute the transformation from base_link to this joint's frame
        T_base_to_joint = compute_transform_to_base(joint_name)

        # Transform vertices to the base_link frame
        # Convert vertices to homogeneous coordinates (Nx4) by adding a column of ones
        homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        # Apply transformation and extract the 3D coordinates
        base_vertices = (T_base_to_joint @ homogeneous_vertices.T).T[:, :3]
        all_base_vertices.append(base_vertices)

        # Plot the transformed vertices
        ax.scatter(base_vertices[:, 0], base_vertices[:, 1], base_vertices[:, 2],
                   color=colors[i], label=joint_name, s=10)

    print(f"Joint values: {q_vis}")
    # Visualization ranges for equal scaling
    all_base_vertices = np.vstack(all_base_vertices)  # Stack all vertices into a single array
    x_min, y_min, z_min = all_base_vertices.min(axis=0)
    x_max, y_max, z_max = all_base_vertices.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    # Visualization options
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.view_init(elev=90, azim=0)  # Adjust view angle for better visualization
    plt.show()

def visualize_meshes_in_base_frame_old(meshes_dict, joint_info, kinematic_chains, q=None, title="Gripper Mesh Visualization", base_link = "base_link"):
    """
    Visualize all meshes from link_meshes in the base_link frame, with optional joint configurations.

    Parameters:
    - meshes_dict (dict).
    - joint_info (dict): Maps joint names to (xyz, quat, joint_type, axis, lower, upper, length).
    - kinematic_chains (list of lists): Defines the hierarchy of joints from base_link to end effectors.
    - q (dict, optional): Dictionary containing the joint values to plot the hand with.
    - title (str): Title of the plot.
    """

    # Build a parent map from kinematic chains
    parent_map = {}
    for chain in kinematic_chains:
        for i in range(1, len(chain)):
            parent_map[chain[i]] = chain[i - 1]

    # Validate q if provided
    if q is not None:
        all_joints = set()
        for chain in kinematic_chains:
            all_joints.update(chain[1:-1])  # Exclude base_link
        missing_joints = all_joints - set(q.keys())
        if missing_joints:
            raise ValueError(f"Missing q values for joints: {missing_joints}")
        q_vis = q.copy()
    else:
        print("No joint values given, showing robot hand at zero configuration.")
        q_vis = {}

    # Compute cumulative transformation from base_link to a joint
    def compute_transform_to_base(joint_name):
        """Returns the 4x4 transformation matrix from base_link to the specified joint."""
        if joint_name == base_link:
            return np.eye(4)

        transforms = []
        current_joint = joint_name
        while current_joint != base_link:
            xyz = joint_info[current_joint][0]
            quat = joint_info[current_joint][1]
            joint_type = joint_info[current_joint][2]
            T = tf.quaternion_matrix(quat)
            T[:3, 3] = xyz
            if q_vis is not None and joint_type == "revolute":
                upper = joint_info[current_joint][5]
                lower = joint_info[current_joint][4]
                q_val = q_vis.get(current_joint, 0.0)
                q_val = np.clip(q_val, lower, upper)
                q_vis[current_joint] = q_val
                R_z = tf.rotation_matrix(q_val, [0, 0, 1])
                T = T @ R_z
            transforms.append(T)
            current_joint = parent_map[current_joint]

        T_base_to_joint = np.eye(4)
        for T in reversed(transforms):
            T_base_to_joint = T_base_to_joint @ T
        return T_base_to_joint

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab20(np.linspace(0, 1, len(meshes_dict.keys())))  # Unique colors per link

    all_vertices = []
    for i, joint_name in enumerate(meshes_dict.keys()):
        if joint_name not in joint_info:
            continue  # Skip links not in joint_info

        # Compute transformation to base frame
        T_base_to_link = compute_transform_to_base(joint_name)

        # Process all meshes for this link
        # print(joint_name)
        # print(meshes_dict[joint_name])
        label_set = False
        for mesh in meshes_dict[joint_name]:
            # Copy mesh to avoid modifying the original
            mesh_copy = mesh.copy()
            # Apply transformation to base frame
            mesh_copy.apply_transform(T_base_to_link)
            # Collect vertices for bounding box
            all_vertices.append(mesh_copy.vertices)
            # Plot triangular mesh
            ax.plot_trisurf(
                mesh_copy.vertices[:, 0],
                mesh_copy.vertices[:, 1],
                mesh_copy.vertices[:, 2],
                triangles=mesh_copy.faces,
                color=colors[i],
                label=joint_name if not label_set else None,
                alpha = 0.3
            )
            label_set = True
        
    # Compute bounding box for equal scaling
    if all_vertices:
        all_vertices = np.vstack(all_vertices)
        x_min, y_min, z_min = all_vertices.min(axis=0)
        x_max, y_max, z_max = all_vertices.max(axis=0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        z_mid = (z_max + z_min) / 2
        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.view_init(elev=90, azim=0)  # Top-down view
    plt.show()

def circular_midpoint(start, end, num_points=1000):
    """Compute the circular midpoint between start and end angles."""
    angles = np.linspace(start, end, num_points)
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return np.arctan2(sin_sum, cos_sum)


def compute_interpolated_offsets_at_angles(driving_angles, driving_offsets, target_angles):
    """
    Compute interpolated offsets at specified target angles using linear interpolation.
    Handles periodicity of angles (0 to 2π) by extending the driving angles and offsets.

    Parameters:
    - driving_angles: 1D array of shape (n_driving_planes,), sorted angles in radians.
    - driving_offsets: 2D array of shape (n_driving_planes, n_spheres), offset values for each driving plane.
    - target_angles: 1D array of shape (n_targets,), target angles in radians where offsets are desired.

    Returns:
    - offsets_at_targets: 2D array of shape (n_targets, n_spheres), interpolated offsets at target angles.
    """
    # Ensure inputs are numpy arrays
    driving_angles = np.asarray(driving_angles)
    driving_offsets = np.asarray(driving_offsets)
    target_angles = np.asarray(target_angles)

    # Reduce target_angles modulo 2π to normalize them within [0, 2π)
    target_angles = np.mod(target_angles, 2 * np.pi)

    # Extend driving angles and offsets to handle periodicity
    angles_extended = np.concatenate([driving_angles - 2 * np.pi, driving_angles, driving_angles + 2 * np.pi])
    offsets_extended = np.concatenate([driving_offsets, driving_offsets, driving_offsets], axis=0)

    # Find insertion points for target angles in the extended angle array
    i = np.searchsorted(angles_extended, target_angles, side='right')

    # Ensure indices are within valid bounds for interpolation
    i = np.clip(i, 1, len(angles_extended) - 1)

    # Get left and right neighboring indices
    left_idx = i - 1
    right_idx = i

    # Extract corresponding angles and offsets
    angle_a = angles_extended[left_idx]  # Shape: (n_targets,)
    angle_b = angles_extended[right_idx]  # Shape: (n_targets,)
    offset_a = offsets_extended[left_idx, :]  # Shape: (n_targets, n_spheres)
    offset_b = offsets_extended[right_idx, :]  # Shape: (n_targets, n_spheres)

    # Compute interpolation fraction
    fraction = (target_angles - angle_a) / (angle_b - angle_a)  # Shape: (n_targets,)
    # Handle edge case where angle_a == angle_b (should be rare with unique driving_angles)
    fraction = np.where(angle_a == angle_b, 0, fraction)

    # Expand fraction for broadcasting with offsets
    fraction_expanded = fraction[:, np.newaxis]  # Shape: (n_targets, 1)

    # Perform linear interpolation
    offsets_at_targets = offset_a + fraction_expanded * (offset_b - offset_a)  # Shape: (n_targets, n_spheres)

    return offsets_at_targets

# Helper function to compute theta for a given q_A
def compute_theta_for_q_A(joint, q_A, q_0, fingertip, joint_info, joint_type_info, kinematic_chains, T_sphere_to_base, T_palm_to_base, base_link="base_link"):
    """Compute the theta angle for a given q_A value."""
    q = q_0.copy()
    q[joint] = q_A
    T_base_to_fingertip = compute_transform_to_joint(fingertip, kinematic_chains, joint_info, q, base_link)
    p_fingertip_base = T_base_to_fingertip[:3, 3]
    p_fingertip_sphere = (T_sphere_to_base @ np.append(p_fingertip_base, 1))[:3]
    theta = np.arctan2(p_fingertip_sphere[1], p_fingertip_sphere[0])
    # Check if fingertip is above palm_normal (z > 0)
    p_fingertip_palm = (T_palm_to_base @ np.append(p_fingertip_base, 1))[:3] 
    if joint_type_info[joint]["azimuthal_flag"]==True:
        return theta
    elif p_fingertip_palm[2] > 0:
        return theta
    return None  # Invalid if fingertip is below or on palm plane


def build_lookup_dict_with_fp_and_phi_lookups(
        finger_print_dict,
        joint,
        q_0,
        joint_info,
        kinematic_chains,
        joint_type_info,
        resolution=0.005,
        N=200,
        base_link="base_link"):
    """Build a lookup dictionary for a Type A joint including FP and phi lookups."""

    lower, upper = joint_info[joint][4], joint_info[joint][5]
    print(joint)
    print(joint)

    # Identify the finger chain for this joint
    chain = next((c for c in kinematic_chains if joint in c), None)
    if chain is None:
        raise ValueError(f"No kinematic chain found for joint {joint}")
    fingertip = chain[-1]
    parent = chain[-2]
    root = chain[1]

    # Compute fixed position of root joint in sphere frame using initial configuration
    q_init = {j: q_0[j] for j in joint_list}  # Use q_0 as dict
    T_base_to_root = compute_transform_to_joint(root, kinematic_chains, joint_info, q_init, base_link=base_link)
    p_base_root = T_base_to_root[:3, 3]
    p_sphere_root = (T_sphere_to_base @ np.append(p_base_root, 1))[:3]
    r_root = np.linalg.norm(p_sphere_root)
    theta_root = np.arctan2(p_sphere_root[1], p_sphere_root[0])

    # Sphere transform
    xyz_sphere = joint_info["sphere_frame"][0]
    quat_sphere = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(quat_sphere)
    T_base_to_sphere[:3, 3] = xyz_sphere
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)

    # Palm transform
    xyz_palm = joint_info["palm_normal"][0]
    quat_palm = joint_info["palm_normal"][1]
    T_base_to_palm = tf.quaternion_matrix(quat_palm)
    T_base_to_palm[:3, 3] = xyz_palm
    T_palm_to_base = np.linalg.inv(T_base_to_palm)

    # Prepare for IK solve
    joint_list = sorted(q_0.keys())
    joint_index_dict = {j: i for i, j in enumerate(joint_list)}
    base_q_array = np.array([q_0[j] for j in joint_list])
    main_idx = joint_index_dict[joint]

    # Get perfect sphere points
    radius = joint_info["sphere_frame"][6]
    p = sample_fibonacci_points(1000)
    theta, phi = p[:,0], p[:,1]
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    points_h_t = points_h.T  # (4, n_points)
    print("points", points_h_t.shape)

    # Combined q_main_joint from lower to upper
    q_main_joint = np.linspace(lower, upper, N + 1)
    q_all_joints = np.tile(base_q_array, (N + 1, 1))
    q_all_joints[:, main_idx] = q_main_joint
    print("q", q_all_joints.shape)

    # Solve IK for all B joints across all q_all_joints
    for i in range(len(chain) - 2):
        j = chain[i + 1]
        if joint_type_info[j]["type"] == "B":
            T_sphere_to_j = compute_sphere_to_single_joint_transforms(j, q_all_joints, joint_index_dict, joint_info, chain)
            T_j_to_sphere = np.linalg.inv(T_sphere_to_j)
            transformed_points_h = np.matmul(T_j_to_sphere, points_h_t).transpose(2, 0, 1)  # (n_points, n_spheres, 4)
            joint_points = transformed_points_h[:, :, :3]
            # print("transformed_points_h", transformed_points_h.shape)
            centroids = np.mean(joint_points, axis=0)
            # print("centroids", centroids.shape)
            # print("joint_points", joint_points.shape)
            box_min = np.array(joint_type_info[j]["box_min"])
            box_max = np.array(joint_type_info[j]["box_max"])
            z_values = joint_points[:, :, 2]
            point_mask = (z_values >= box_min[2]) & (z_values <= box_max[2])
            # point_mask = (z_values >= -100000) 
            l_ft = np.full(N + 1, 1000.0) 
            # print("point_mask", point_mask.shape)
            # print("l_ft", l_ft.shape)
            new_q = solve_for_B_joint(q_all_joints, j, joint_index_dict, centroids, joint_points, joint_info, point_mask, l_ft, joint_type_info)
            q_all_joints[:, joint_index_dict[j]] = new_q
        
        elif joint_type_info[j]["type"] == "D":
                # Solve for type D
                print("Solving for:", j)
                T_sphere_to_j = compute_sphere_to_single_joint_transforms(j, q_all_joints, joint_index_dict, joint_info, chain)
                T_j_to_sphere = np.linalg.inv(T_sphere_to_j)
                transformed_points_h = np.matmul(T_j_to_sphere, points_h_t).transpose(2, 0, 1)  # (n_points, n_spheres, 4)
                joint_points = transformed_points_h[:-1,:,:3]
                centroids = transformed_points_h[-1,:,:3]

                #Get ft position (Describes mode of joint)
                T_joint_to_fts = compute_joint_to_fingertip_transforms(
                    j, 
                    q_all_joints, 
                    joint_index_dict, 
                    joint_info, 
                    chain
                )
                # print("T_joint to fts \n", T_joint_to_fts)

                solve_for_D_joint(
                    q_all_joints, j, joint_index_dict, centroids, joint_points,
                    joint_info, chain, joint_type_info, T_joint_to_fts
                )
            

    # Compute p_ft_sphere_all
    T_sphere_to_ft_all = compute_sphere_to_single_joint_transforms(fingertip, q_all_joints, joint_index_dict, joint_info, chain)
    p_ft_sphere_all = T_sphere_to_ft_all[:, :3, 3]
    r_all = np.linalg.norm(p_ft_sphere_all, axis=1)
    phi_all = np.arccos(p_ft_sphere_all[:, 2] / r_all)

    # Compute theta_all
    theta_all = np.arctan2(p_ft_sphere_all[:, 1], p_ft_sphere_all[:, 0])

    # Compute p_ft_palm for theta check
    # p_ft_base_all = np.matmul(T_base_to_sphere[:3, :3], p_ft_sphere_all.T).T + T_base_to_sphere[:3, 3]
    # p_ft_palm_all = np.matmul(T_palm_to_base[:3, :3], p_ft_base_all.T).T + T_palm_to_base[:3, 3]
    # valid_theta_mask = p_ft_palm_all[:, 2] > 0

    # Create mask
    phi_mask =  phi_all >= 3 * np.pi/ 8
    valid_mask = phi_mask 

    # Find q_mid_idx
    q_mid = q_0[joint]
    q_mid_idx = np.argmin(np.abs(q_main_joint - q_mid))
    # print("q_mid", q_mid)
    # print("q_main_joint", q_main_joint)
    # print("phi_all", phi_all)
    # print("phi_mask", phi_mask)
    # print("theta", valid_theta_mask)

    # Find the bounding False indices around the target_idx to define the contiguous valid range
    left_false_indices = np.where((np.arange(len(valid_mask)) < q_mid_idx) & ~valid_mask)[0]
    if len(left_false_indices) == 0:
        selected_start = 0
    else:
        selected_start = np.max(left_false_indices) + 1

    right_false_indices = np.where((np.arange(len(valid_mask)) > q_mid_idx) & ~valid_mask)[0]
    if len(right_false_indices) == 0:
        selected_end = len(valid_mask) - 1
    else:
        selected_end = np.min(right_false_indices) - 1

    # Verify the range is valid and contiguous
    if selected_start > selected_end or not np.all(valid_mask[selected_start:selected_end + 1]):
        raise ValueError(f"Non-contiguous or invalid range for joint {joint}")

    valid_idx = slice(selected_start, selected_end + 1)
    valid_q_main_joint = q_main_joint[valid_idx]
    valid_theta = theta_all[valid_idx]
    valid_q_all_joints = q_all_joints[valid_idx]
    
    if not valid_q_main_joint.size:
        raise ValueError(f"No valid theta values for joint {joint}")

    # Compute anchor as center of theta range
    min_theta = np.min(valid_theta)
    max_theta = np.max(valid_theta)
    anchor = min_theta + (max_theta - min_theta) / 2

    # Find index closest to anchor
    idx_anchor = np.argmin(np.abs(valid_theta - anchor))
    print("anchor", anchor)
    print("idx_anchor", idx_anchor)
    print("theta max & min", max_theta, min_theta)


    # Compute theta offsets from anchor and wrap to [-pi, pi]
    theta_offsets = valid_theta - anchor
    theta_offsets = (theta_offsets + np.pi) % (2 * np.pi) - np.pi

    # Determine min and max offsets
    min_offset = np.min(theta_offsets)
    max_offset = np.max(theta_offsets)

    # Sort theta_offsets and valid_q_main_joint if necessary
    sort_idx = np.argsort(theta_offsets)
    theta_offsets = theta_offsets[sort_idx]
    valid_q_main_joint = valid_q_main_joint[sort_idx]

    # Generate offset steps
    offsets = np.arange(min_offset, max_offset + resolution / 2, resolution)
    
    # Find indices for min and max offsets
    idx_min_offset = np.argmin(theta_offsets)
    idx_max_offset = np.argmax(theta_offsets)
    # Get corresponding q values at min and max offsets
    q_min = valid_q_main_joint[idx_min_offset]
    q_max = valid_q_main_joint[idx_max_offset]
    # Create interpolated q values linearly spaced between q_min and q_max over the length of offsets
    q_list = np.linspace(q_min, q_max, len(offsets))
    zero_idx = np.argmin(np.abs(offsets))
    # print("theta offsets", offsets)
    print("q max & min", q_max, q_min)
    print("theta offset at q max & min", theta_offsets[idx_max_offset], theta_offsets[idx_min_offset])


    # Now create phi_lookup and fp_lookup based on fp_7
    fractions = [1.0, 5.0/6, 4.0/6, 3.0/6, 2.0/6, 1.0/6, 0.0]

    # Compute closest joints for intermediate fractions at q_ref_dict
    relevant_joints = [j for j in chain[1:] if j in finger_print_dict and finger_print_dict[j].get('points', np.array([])).size > 0]
    if not relevant_joints:
        raise ValueError(f"No relevant joints for chain {chain}")
    all_points_fingertip = []
    all_joints = []
    for j in relevant_joints:
        T_j_to_ft = compute_transform_from_joint_to_fingertip(chain, j, joint_info, q_0)
        T_ft_to_j = np.linalg.inv(T_j_to_ft)
        points_j = finger_print_dict[j]['points']
        points_j_h = np.hstack((points_j, np.ones((len(points_j), 1))))
        points_fingertip_h = (T_ft_to_j @ points_j_h.T).T
        points_fingertip = points_fingertip_h[:, :3]
        all_points_fingertip.append(points_fingertip)
        all_joints.extend([j] * len(points_j))
    all_points_fingertip = np.vstack(all_points_fingertip)

    # v at q_ref_dict
    T_root_to_ft = compute_transform_from_joint_to_fingertip(chain, root, joint_info, q_0)
    T_ft_to_root = np.linalg.inv(T_root_to_ft)
    p_root_in_ft = T_ft_to_root[:3, 3]
    v = p_root_in_ft

    # Closest for intermediate
    closest_joints = {}
    for frac in fractions[1:-1]:
        pos = frac * v
        distances = np.linalg.norm(all_points_fingertip - pos, axis=1)
        idx = np.argmin(distances)
        closest_joints[frac] = all_joints[idx]

    # Initialize new_fp_sample and mappings
    new_fp_sample = {j: [] for j in chain}
    frac_to_joint = {}
    frac_to_index = {}
    for i_frac, frac in enumerate(fractions):
        if frac == 1.0:
            joint_s = root
        elif frac == 0.0:
            joint_s = parent
        else:
            joint_s = closest_joints[frac]
        frac_to_joint[frac] = joint_s
        index = len(new_fp_sample[joint_s])
        frac_to_index[frac] = index
        new_fp_sample[joint_s].append([])

    # Add two additional samples for parent at ft with phi +5° and +10°
    parent_joint = parent
    index_plus5 = len(new_fp_sample[parent_joint])
    new_fp_sample[parent_joint].append([])
    index_plus10 = len(new_fp_sample[parent_joint])
    new_fp_sample[parent_joint].append([])

    # Loop over q_list to compute phi and new fp
    phi_lookup = []
    chain_joints = chain[1:-1] # Joints in the chain order, excluding base and fingertip
    q_control_lookup = []
    joint_phis_lookup = []
    for offset_idx in range(len(q_list)):
        q_A = q_list[offset_idx]
        idx_closest = np.argmin(np.abs(valid_q_main_joint - q_A))
        full_q = valid_q_all_joints[idx_closest]
        q = {joint_list[i]: full_q[i] for i in range(len(joint_list))}
        # Fingertip theta
        T_base_to_ft = compute_transform_to_joint(fingertip, kinematic_chains, joint_info, q, base_link=base_link)
        p_ft_base = T_base_to_ft[:3, 3]
        p_ft_sphere = (T_sphere_to_base @ np.append(p_ft_base, 1))[:3]
        fingertip_theta = np.arctan2(p_ft_sphere[1], p_ft_sphere[0])
        phis = []
        # Recompute v
        T_root_to_ft = compute_transform_from_joint_to_fingertip(chain, root, joint_info, q)
        T_ft_to_root = np.linalg.inv(T_root_to_ft)
        p_root_in_ft = T_ft_to_root[:3, 3]
        v = p_root_in_ft
        for frac in fractions:
            joint_s = frac_to_joint[frac]
            if frac == 1.0:
                local_p = np.array([0, 0, 0])
            elif frac == 0.0:
                local_p = joint_info[fingertip][0]
            else:
                pos = frac * v
                T_closest_to_ft = compute_transform_from_joint_to_fingertip(chain, joint_s, joint_info, q)
                pos_h = np.append(pos, 1)
                local_p = (T_closest_to_ft @ pos_h)[:3]
            T_base_to_joint_s = compute_transform_to_joint(joint_s, kinematic_chains, joint_info, q, base_link=base_link)
            p_base = (T_base_to_joint_s @ np.append(local_p, 1))[:3]
            p_sphere = (T_sphere_to_base @ np.append(p_base, 1))[:3]
            r_current = np.linalg.norm(p_sphere)
            phi = np.arccos(p_sphere[2] / r_current)
            phis.append(phi)
            # New fp
            new_p_sphere = radius * np.array([np.sin(phi) * np.cos(fingertip_theta), np.sin(phi) * np.sin(fingertip_theta), np.cos(phi)])
            p_base_new = (T_base_to_sphere @ np.append(new_p_sphere, 1))[:3]
            T_joint_to_base = np.linalg.inv(T_base_to_joint_s)
            p_local_new = (T_joint_to_base @ np.append(p_base_new, 1))[:3]
            index = frac_to_index[frac]
            new_fp_sample[joint_s][index].append(p_local_new)
            if frac == 0.0:
                T_base_to_parent = T_base_to_joint_s

        # Now add the additional points for parent
        phi_ft = phis[-1]
        deltas = [np.deg2rad(7.5), np.deg2rad(15)]
        for i, delta in enumerate(deltas):
            new_phi = phi_ft + delta
            new_p_sphere = radius * np.array([np.sin(new_phi) * np.cos(fingertip_theta), np.sin(new_phi) * np.sin(fingertip_theta), np.cos(new_phi)])
            p_base_new = (T_base_to_sphere @ np.append(new_p_sphere, 1))[:3]
            p_local_new = (np.linalg.inv(T_base_to_parent) @ np.append(p_base_new, 1))[:3]
            if i == 0:
                new_fp_sample[parent][index_plus5].append(p_local_new)
                phis.append(phi_ft + np.deg2rad(5))
            else:
                new_fp_sample[parent][index_plus10].append(p_local_new)
                phis.append(phi_ft + np.deg2rad(10))
            
        phi_lookup.append(phis)
        q_control_lookup.append([full_q[joint_index_dict[j]] for j in chain_joints])

        joint_phis = []
        for j in chain_joints:
            T_base_to_j = compute_transform_to_joint(j, kinematic_chains, joint_info, q, base_link=base_link)
            p_base = T_base_to_j[:3, 3]
            p_sphere = (T_sphere_to_base @ np.append(p_base, 1))[:3]
            r = np.linalg.norm(p_sphere)
            phi_j = np.arccos(p_sphere[2] / r) if r != 0 else 0.0
            joint_phis.append(phi_j)
        joint_phis_lookup.append(joint_phis)

    fingertip_phis = [phis[-3] for phis in phi_lookup]
    max_phi_idx = np.argmax(fingertip_phis)
    joints_phi = {chain_joints[i]: joint_phis_lookup[max_phi_idx][i] for i in range(len(chain_joints))}
    # Add fingertip phi to joints_phi
    joints_phi[chain[-1]] = phi_lookup[max_phi_idx][-1]
    next_joint_dict = {}
    for j in chain_joints:
        j_idx = chain.index(j)
        next_joint = chain[j_idx + 1]
        if joint_type_info[j]["type"] == "D":
            mj_idx = chain.index(joint)
            next_joint = chain[mj_idx]
        nj_i = chain.index(next_joint)
        if next_joint != chain[-1]:
            while abs(joints_phi[j] - joints_phi[next_joint]) < 0.0872665:
                nj_i += 1
                next_joint = chain[nj_i]
                if next_joint == chain[-1]:
                    break
        next_joint_dict[j] = next_joint
    print(joint)
    print()
    print()
    a_dict = {
        "type": "A",
        "anchor": anchor,
        "resolution": resolution,
        "upper": max_offset,
        "lower": min_offset,
        "q_list": q_list,
        "zero_idx": zero_idx,
        "og_type": joint_type_info[joint]["type"]
    }
    joint_type_info[joint].update(a_dict)
    

    # Save sphere control 
    sphere_control_dict = {
        joint: {"fp_phis": phi_lookup,
                "q_control": q_control_lookup,
                "joint_phis": joint_phis_lookup,
                "next_joint": next_joint_dict}}

    # Get the full q_ref_all for this index 
    q_ref_all = valid_q_all_joints[idx_anchor]
    q_ref_dict = {joint_list[i]: q_ref_all[i] for i in range(len(joint_list))}

    return new_fp_sample, sphere_control_dict, q_ref_dict

def solve_for_A_joints(type_A_joints, joint_type_info, d_planes_theta, d_plane_offsets, joint_index_dict, q_0_all):
    """
    Solves inverse kinematics for the specified type A joints.

    Parameters:
    - type_A_joints: list of joint names to solve for
    - joint_type_info: dictionary containing information about each joint
    - d_planes_theta: theta values for interpolation
    - d_plane_offsets: offset values for interpolation
    - joint_index_dict: dictionary mapping joint names to their indices in q_0_all
    - q_0_all: 2D array to update with the solved joint values

    This function computes the joint values for the specified type A joints based on the provided interpolation data and updates q_0_all in place.
    """
    # Gather joint-specific data: anchors, resolutions, and offset lists
    q_A_anchors = [joint_type_info[q_A]["anchor"] for q_A in type_A_joints]
    q_A_res = [joint_type_info[q_A]["resolution"] for q_A in type_A_joints]
    q_A_lists = [np.array(joint_type_info[q_A]["q_list"]) for q_A in type_A_joints]

    # Set min and max index bounds for each joint
    q_A_min = np.zeros(len(q_A_anchors)) 
    q_A_max = np.array([len(joint_type_info[q_A]["q_list"])-1 for q_A in type_A_joints])

    # Get zero indices for offset calculations
    q_A_zero_idx = np.array([joint_type_info[q_A]["zero_idx"] for q_A in type_A_joints])
    q_A_anchor_dist = np.array([joint_type_info[q_A]["anchor_dist"] for q_A in type_A_joints])

    # Interpolate offsets at anchor angles
    anchor_offsets = compute_interpolated_offsets_at_angles(d_planes_theta, d_plane_offsets, q_A_anchors).T

    # Compute offset indices from scaled and shifted offsets
    offset_idx = np.round(np.divide(anchor_offsets, q_A_res) + q_A_zero_idx + np.divide(q_A_anchor_dist, q_A_res))

    # Ensure indices stay within valid bounds
    offset_idx = np.clip(offset_idx, q_A_min, q_A_max).astype(int)

    # Retrieve offset values using computed indices
    result = np.empty_like(offset_idx, dtype=float)
    for joint_idx in range(offset_idx.shape[1]):
        result[:, joint_idx] = q_A_lists[joint_idx][offset_idx[:, joint_idx]]

    # Map joint names to indices in q_0_all
    q_A_indices = [joint_index_dict[x] for x in type_A_joints]

    # Update q_0_all with the computed values
    q_0_all[:, q_A_indices] = result

    return 

def solve_for_B_joint(q_0_all, joint_tag, joint_index_dict, centroids,
                      joint_points, joint_info, point_mask, l_ft, joint_type_info):
    """
    Compute updated joint angles for a B joint across all spheres based on filtered points and centroids.
    Parameters:
    - q_0_all (np.ndarray): Shape (n_spheres, n_joints), current joint angles.
    - joint_tag (str): Name of the B joint to solve for.
    - joint_index_dict (dict): Maps joint names to column indices in q_0_all.
    - centroids (np.ndarray): Shape (n_spheres, 3), centroids in the joint frame (x, y, z).
    - joint_points (np.ndarray): Shape (n_points, n_spheres, 3), points in the joint frame (x, y, z).
    - joint_info (dict): General joint information.
    - point_mask: Prefiltered points to follow (n_points, n_spheres)
    - l_ft: magnitude in x-y axis to filter with (n_spheres,)
    - joint_type_info (dict): Type joint information of joint.
    Returns:
    - np.ndarray: Shape (n_spheres,), updated joint angles for the B joint.
    """
    # Get the column index of the B joint
    B_joint_idx = joint_index_dict[joint_tag]
    # Initialize dimensions and limits
    n_spheres = q_0_all.shape[0]
    lower_limit = joint_info[joint_tag][4]
    upper_limit = joint_info[joint_tag][5]
    # Optionally include centroid for og_type A
    if "og_type" in joint_type_info[joint_tag].keys():
        if joint_type_info[joint_tag]["og_type"] == "A":
            centroids_reshaped = centroids[np.newaxis, :, :]
            joint_points = np.concatenate([joint_points, centroids_reshaped], axis=0)
            centroid_mask = np.ones((1, n_spheres), dtype=bool)
            point_mask = np.concatenate([point_mask, centroid_mask], axis=0)
    # Extract x and y
    x = joint_points[..., 0]  # Shape: (n_points, n_spheres)
    y = joint_points[..., 1]  # Shape: (n_points, n_spheres)
    ref_dir = joint_type_info[joint_tag]["ref_dir"]
    # Additional filter based on ref_dir
    if ref_dir[1] >= 0:
        exclude = (x < 0) & (y < 0)
    else:
        exclude = (x < 0) & (y > 0)
    additional_mask = ~exclude  # Shape: (n_points, n_spheres)
    # Effective mask
    effective_mask = point_mask & additional_mask
    # Compute arctan2 angles for all points
    angles = np.arctan2(y, x)  # Shape: (n_points, n_spheres)
    # Determine min or max based on ref_dir
    if ref_dir[1] >= 0:
        masked_angles = np.where(effective_mask, angles, np.inf)
        result = np.min(masked_angles, axis=0)
        default_result = upper_limit
    else:
        masked_angles = np.where(effective_mask, angles, -np.inf)
        result = np.max(masked_angles, axis=0)
        default_result = lower_limit
    # Check for valid points per sphere
    has_valid_points = np.any(effective_mask, axis=0)  # Shape: (n_spheres,)
    # Update joint angles
    new_q = result + q_0_all[:, B_joint_idx]
    # Apply default where no valid points
    new_q = np.where(has_valid_points, new_q, default_result)
    # Clip to joint limits
    new_q = np.clip(new_q, lower_limit, upper_limit)
    return new_q

def solve_for_B_joint_eigenvectors(q_0_all, joint_tag, joint_index_dict, centroids,
                      joint_points, joint_info, point_mask, l_ft, joint_type_info):
    """
    Compute updated joint angles for a B joint across all spheres based on filtered points and centroids.
    Parameters:
    - q_0_all (np.ndarray): Shape (n_spheres, n_joints), current joint angles.
    - joint_tag (str): Name of the B joint to solve for.
    - joint_index_dict (dict): Maps joint names to column indices in q_0_all.
    - centroids (np.ndarray): Shape (n_spheres, 3), centroids in the joint frame (x, y, z).
    - joint_points (np.ndarray): Shape (n_points, n_spheres, 3), points in the joint frame (x, y, z).
    - joint_info (dict): General joint information.
    - point_mask: Prefiltered points to follow (n_points, n_spheres)
    - l_ft: magnitude in x-y axis to filter with (n_spheres,)
    Returns:
    - np.ndarray: Shape (n_spheres,), updated joint angles for the B joint.
    """
    # Get the column index of the B joint
    B_joint_idx = joint_index_dict[joint_tag]
    # Handle special case for A-type joints by including centroids as additional points
    if "og_type" in joint_type_info[joint_tag].keys():
        if joint_type_info[joint_tag]["og_type"] == "A":
            centroids_reshaped = centroids[np.newaxis, :, :]
            # Concatenate joint_points and centroids along the points axis (axis=0)
            joint_points = np.concatenate([joint_points, centroids_reshaped], axis=0)
            # Extend point_mask to include True for the centroid point
            centroid_mask = np.ones((1, point_mask.shape[1]), dtype=bool)
            point_mask = np.concatenate([point_mask, centroid_mask], axis=0)
    # Extract joint limits from joint_info
    lower_limit = joint_info[joint_tag][4]
    upper_limit = joint_info[joint_tag][5]
    # Get number of spheres
    n_spheres = q_0_all.shape[0]
    # Extract y coordinates for filtering
    y_values = joint_points[..., 1]  # Shape: (n_points, n_spheres)
    # Fetch ref_dir early for both filtering and defaults
    ref_dir = joint_type_info[joint_tag]["ref_dir"]
    # Create masked array for y_values based on point_mask
    masked_y = ma.masked_where(~point_mask, y_values)
    # Compute reference y value per sphere based on ref_dir[1]
    if ref_dir[1] > 0:
        y_ref = ma.min(masked_y, axis=0).filled(np.inf)  # Shape: (n_spheres,)
    else:
        y_ref = ma.max(masked_y, axis=0).filled(-np.inf)  # Shape: (n_spheres,)
    # Compute absolute y differences from reference
    y_diff = np.abs(y_values - y_ref[np.newaxis, :])  # Shape: (n_points, n_spheres)
    # Create new mask based on y distance <= l_ft (per sphere)
    new_y_mask = y_diff <= l_ft[np.newaxis, :]  # Shape: (n_points, n_spheres)
    # Combine with original point_mask
    final_mask = point_mask & new_y_mask  # Shape: (n_points, n_spheres)
    # Extract x and y coordinates for all points across spheres
    points_xy = joint_points[..., :2]  # Shape: (n_points, n_spheres, 2)
    # Convert mask to float for element-wise multiplication in sums
    mask_float = final_mask.astype(np.float64)  # Shape: (n_points, n_spheres)
    # Compute components of the 2x2 scatter matrix per sphere using masked sums
    sum_xx = np.sum(points_xy[..., 0]**2 * mask_float, axis=0)  # Shape: (n_spheres,)
    sum_yy = np.sum(points_xy[..., 1]**2 * mask_float, axis=0)  # Shape: (n_spheres,)
    sum_xy = np.sum(points_xy[..., 0] * points_xy[..., 1] * mask_float, axis=0)  # Shape: (n_spheres,)
    # Construct batched 2x2 scatter matrices for all spheres
    S = np.zeros((n_spheres, 2, 2))  # Shape: (n_spheres, 2, 2)
    S[:, 0, 0] = sum_xx
    S[:, 1, 1] = sum_yy
    S[:, 0, 1] = sum_xy
    S[:, 1, 0] = sum_xy
    # Compute eigenvalues and eigenvectors for batched symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(S)  # eigenvalues: (n_spheres, 2), eigenvectors: (n_spheres, 2, 2)
    # Identify index of the largest eigenvalue per sphere
    idx = np.argmax(eigenvalues, axis=1)  # Shape: (n_spheres,)
    # Extract the corresponding eigenvector (direction vector) per sphere
    v = eigenvectors[np.arange(n_spheres), :, idx]  # Shape: (n_spheres, 2)
    # Compute mean xy per sphere to orient the direction vector
    num_valid = np.sum(final_mask, axis=0).astype(np.float64)  # Shape: (n_spheres,)
    num_valid = np.maximum(num_valid, 1e-6)  # Avoid division by zero
    mean_xy = np.sum(points_xy * mask_float[:, :, np.newaxis], axis=0) / num_valid[:, np.newaxis]  # Shape: (n_spheres, 2)
    # Compute projection of v onto mean_xy
    proj = np.sum(v * mean_xy, axis=1)  # Shape: (n_spheres,)
    # Flip sign of v if projection is negative
    sign_flip = proj < 0  # Shape: (n_spheres,)
    v = np.where(sign_flip[:, np.newaxis], -v, v)
    # Compute the angle of the direction vector with respect to the x-axis per sphere
    angles = np.arctan2(v[:, 1], v[:, 0])  # Shape: (n_spheres,)
    # Set angles to zero if their absolute value is below the threshold (th = 0.0)
    th = 0.0
    if th > 0.0:
        angles = np.where(np.abs(angles) < th, 0.0, angles)
    # Determine default result based on reference direction
    if ref_dir[1] > 0:
        default_result = upper_limit
    else:
        default_result = lower_limit
    # Check for valid points per sphere
    has_valid_points = np.any(final_mask, axis=0)  # Shape: (n_spheres,)
    # Set new joint angles to (original + computed angles) or default limit where no valid points
    new_q = np.where(has_valid_points, q_0_all[:, B_joint_idx] + angles, default_result)
    # Clip the updated angles to stay within joint limits
    new_q = np.clip(new_q, lower_limit, upper_limit)
    return new_q

def solve_for_C_joint(q_0_all, joint_tag, joint_index_dict, centroids, joint_info, ft_ps, ft_normals):
    """
    Compute updated joint angles for a C joint across all spheres based on centroids and fingertip position.

    Parameters:
    - q_0_all (np.ndarray): Shape (n_spheres, n_joints), current joint angles.
    - joint_tag (str): Name of the B joint to solve for.
    - joint_index_dict (dict): Maps joint names to column indices in q_0_all.
    - centroids (np.ndarray): Shape (n_spheres, 3), centroids in the joint frame (x, y, z).
    - joint_info (dict): General joint information (unused here).
    - ft_ps:
    - ft_normals:

    Returns:
    - np.ndarray: Shape (n_spheres,), updated joint angles for the B joint.
    """
    # Get the column index of the B joint
    C_joint_idx = joint_index_dict[joint_tag]
    
    # Initialize result array
    n_spheres = q_0_all.shape[0]
    result = np.zeros(n_spheres)

    # Compute the angle to allign the fingertip normal to the center of the sphere
    d_c_ft = centroids - ft_ps
    theta_d = np.arctan2(d_c_ft[:,1], d_c_ft[:,0])
    theta_v = np.arctan2(ft_normals[:,1], ft_normals[:,0])
    print("theta_d",theta_d)
    print("theta_v",theta_v)
    result = theta_d - theta_v
    
    # Update joint angles
    new_q = q_0_all[:, C_joint_idx] + result

    lower_limit = joint_info[joint_tag][4]
    upper_limit = joint_info[joint_tag][5]
    
    # Clip the values in the specified joint column to the [lower, upper] range
    new_q = np.clip(new_q, lower_limit, upper_limit)
    return new_q

def compute_joint_to_fingertip_transforms(joint_tag, q_0_all, joint_index_dict, joint_info, chain):
    """
    Compute transformations from the child link of the specified joint to the fingertip for all configurations in q_0_all.

    Parameters:
    - joint_tag (str): Name of the joint from which to compute the transformation.
    - q_0_all (np.ndarray): Array of shape (n_spheres, n_joints) with joint angles for each configuration.
    - joint_index_dict (dict): Dictionary mapping joint names to their indices in q_0_all.
    - joint_info (dict): Dictionary mapping joint names to (xyz, quat, type, axis, lower, upper, length).
    - chain (list): List defining the kinematic chain from base_link to the fingertip.

    Returns:
    - np.ndarray: Array of shape (n_spheres, 4, 4) with transformations from the child link of joint_tag to the fingertip.

    Raises:
    - ValueError: If joint_tag is not in the chain or is the fingertip itself.
    """
    # Number of configurations
    n_spheres = q_0_all.shape[0]

    # Validate joint_tag
    try:
        idx = chain.index(joint_tag)
    except ValueError:
        raise ValueError(f"Joint '{joint_tag}' not found in the chain")
    if idx == len(chain) - 1:
        # If joint_tag is the fingertip, return identity matrices
        raise ValueError(f"Error: Attempting to solve Fingertip as Type D joint.")

    # Define subchain from the next joint to the last joint before the fingertip
    subchain = chain[idx + 1 :]  

    # Initialize cumulative transformation as identity for all spheres
    T_cumulative = np.eye(4)[np.newaxis].repeat(n_spheres, axis=0)  # Shape: (n_spheres, 4, 4)

    # Accumulate transformations along the subchain
    for joint in subchain:
        # Extract joint properties
        xyz, quat, joint_type, _, _, _, _ = joint_info[joint]
        
        # Compute fixed transformation
        T_fixed = tf.quaternion_matrix(quat)  # Shape: (4, 4)
        T_fixed[:3, 3] = xyz

        if joint_type == "revolute" and joint in joint_index_dict:
            # Get joint angles for this joint across all configurations
            q_values = q_0_all[:, joint_index_dict[joint]]  # Shape: (n_spheres,)
            
            # Compute rotation matrices around z-axis vectorized
            cos_q = np.cos(q_values)
            sin_q = np.sin(q_values)
            R_z = np.array([
                [cos_q, -sin_q, np.zeros(n_spheres)],
                [sin_q, cos_q, np.zeros(n_spheres)],
                [np.zeros(n_spheres), np.zeros(n_spheres), np.ones(n_spheres)]
            ]).transpose(2, 0, 1)  # Shape: (n_spheres, 3, 3)
            
            # Extend to 4x4 matrices
            T_rot = np.eye(4)[np.newaxis].repeat(n_spheres, axis=0)  # Shape: (n_spheres, 4, 4)
            T_rot[:, :3, :3] = R_z
            
            # Combine fixed and rotation transformations
            T_joint = T_fixed @ T_rot  # Shape: (n_spheres, 4, 4), T_fixed broadcasts
        else:
            # For fixed joints or if joint not in q_0_all, use fixed transformation
            T_joint = T_fixed  # Shape: (4, 4), will be broadcasted

        # Update cumulative transformation
        T_cumulative = T_cumulative @ T_joint  # Shape: (n_spheres, 4, 4)

    return T_cumulative

def solve_for_D_joint_dual_behavior(q_0_all, joint, joint_index_dict, centroids, joint_points, joint_info, chain, joint_type_info, T_joint_to_fts):
    """
    Solve for type "D" joints by handling spheres based on fingertip position and original joint type.

    This function processes type "D" joints by:
    1. Computing a mask (`ft_mask`) based on fingertip position relative to a bounding box.
    2. Solving spheres where `ft_mask` is True as type "C" joints.
    3. Solving spheres where `ft_mask` is False as type "B" joints with modifications based on the original type.

    Parameters:
    - q_0_all (np.ndarray): Shape (n_spheres, n_joints), current joint angles to be updated in place.
    - joint (str): Name of the joint to solve for.
    - joint_index_dict (dict): Maps joint names to column indices in q_0_all.
    - centroids (np.ndarray): Shape (n_spheres, 3), centroids in the joint frame.
    - joint_points (np.ndarray): Shape (n_points, n_spheres, 3), points in the joint frame.
    - joint_info (dict): General joint information.
    - chain (list): Kinematic chain containing the joint.
    - joint_type_info (dict): Type-specific information for joints.
    - T_joint_to_fts (np.ndarray): Shape (n_spheres, 4, 4), transformations from joint to fingertip.

    Returns:
    - None: Updates q_0_all in place.
    """
    # Compute ft_mask based on fingertip position
    ft_x = T_joint_to_fts[:, 0, 3]  # Shape: (n_spheres,)
    ft_y = T_joint_to_fts[:, 1, 3]  # Shape: (n_spheres,)
    box_min = joint_type_info[chain[-1]]["box_min"]
    box_max = joint_type_info[chain[-1]]["box_max"]
    ft_mask = (ft_x >= box_min[2]) & (ft_x <= box_max[2]) & (ft_y >= box_min[2]) & (ft_y <= box_max[2])  # Shape: (n_spheres,)
    print("Solve as C type (ft_mask):", ft_mask)

    # Extract fingertip position and normal for all spheres
    ft_ps = T_joint_to_fts[:, :3, 3]       # Shape: (n_spheres, 3)
    ft_normals = T_joint_to_fts[:, :3, 2]  # Shape: (n_spheres, 3)
    l_ft = np.linalg.norm(ft_ps[:, :2], axis=1)  # Shape: (n_spheres,)

    # **Handle spheres where ft_mask is True (solve as Type C)**
    if np.any(ft_mask):
        c_mask = ft_mask
        q_0_c = q_0_all[c_mask]              # Shape: (n_c_spheres, n_joints)
        centroids_c = centroids[c_mask]      # Shape: (n_c_spheres, 3)
        ft_ps_c = ft_ps[c_mask]              # Shape: (n_c_spheres, 3)
        ft_normals_c = ft_normals[c_mask]    # Shape: (n_c_spheres, 3)

        new_q_c = solve_for_C_joint(
            q_0_c,
            joint,
            joint_index_dict,
            centroids_c,
            joint_info,
            ft_ps_c,
            ft_normals_c
        )
        q_0_all[c_mask, joint_index_dict[joint]] = new_q_c
        print(f"Type D (as C) result for {joint}:", new_q_c)

    # **Handle spheres where ft_mask is False (solve as Type B with modifications)**
    if np.any(~ft_mask):
        b_mask = ~ft_mask
        n_b_spheres = np.sum(b_mask)
        joint_points_b = joint_points[:, b_mask, :]  # Shape: (n_points, n_b_spheres, 3)
        centroids_b = centroids[b_mask]              # Shape: (n_b_spheres, 3)
        T_joint_to_fts_b = T_joint_to_fts[b_mask]    # Shape: (n_b_spheres, 4, 4)
        l_ft_b = l_ft[b_mask]                        # Shape: (n_b_spheres,)
        q_0_b = q_0_all[b_mask]                      # Shape: (n_b_spheres, n_joints)

        # Initialize point_mask for B spheres
        point_mask_b = np.zeros((joint_points.shape[0], n_b_spheres), dtype=bool)

        # Homogeneous points
        joint_points_h_b = np.concatenate(
            [joint_points_b, np.ones((joint_points_b.shape[0], n_b_spheres, 1))],
            axis=2
        )  # Shape: (n_points, n_b_spheres, 4)

        # Get original joint type
        og_type = joint_type_info[joint]["og_type"]
        ft_pos_b = T_joint_to_fts_b[:, :3, 3]  # Shape: (n_b_spheres, 3)

        if og_type == "C":
            # Compute rotation angles (alpha) for all spheres
            alpha = np.arctan2(-ft_pos_b[:, 1], ft_pos_b[:, 2])  # Shape: (n_b_spheres,)

            # Create rotation matrices around x-axis
            cos_alpha = np.cos(alpha)
            sin_alpha = np.sin(alpha)
            R_x = np.array([
                [np.ones(n_b_spheres), np.zeros(n_b_spheres), np.zeros(n_b_spheres)],
                [np.zeros(n_b_spheres), cos_alpha, -sin_alpha],
                [np.zeros(n_b_spheres), sin_alpha, cos_alpha]
            ]).transpose(2, 0, 1)  # Shape: (n_b_spheres, 3, 3)
            # print("R_x\n", R_x)

            # Extend to 4x4 transformation matrices
            T_rot = np.tile(np.eye(4), (n_b_spheres, 1, 1))
            T_rot[:, :3, :3] = R_x
            # print("T_rot\n", T_rot)
            T_rot = np.linalg.inv(T_rot)

            # Apply rotations to all points for all spheres
            rotated_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h_b)  # Shape: (n_points, n_b_spheres, 4)
            rotated_points = rotated_points_h[:, :, :3]  # Shape: (n_points, n_b_spheres, 3)

            # Compute point mask based on y-component (rotated y-range)
            point_mask_b = (rotated_points[:, :, 1] >= box_min[2]) & (rotated_points[:, :, 1] <= box_max[2])  # Shape: (n_points, n_b_spheres)

        elif og_type == "B":
            # Compute rotation angles (beta) for all spheres
            beta = np.arctan2(-ft_pos_b[:, 2], ft_pos_b[:, 0])  # Shape: (n_b_spheres,)

            # Create rotation matrices around y-axis
            cos_beta = np.cos(beta)
            sin_beta = np.sin(beta)
            R_y = np.array([
                [cos_beta, np.zeros(n_b_spheres), sin_beta],
                [np.zeros(n_b_spheres), np.ones(n_b_spheres), np.zeros(n_b_spheres)],
                [-sin_beta, np.zeros(n_b_spheres), cos_beta]
            ]).transpose(2, 0, 1)  # Shape: (n_b_spheres, 3, 3)
            # print("R_y\n", R_y)

            # Extend to 4x4 transformation matrices
            T_rot = np.tile(np.eye(4), (n_b_spheres, 1, 1))
            T_rot[:, :3, :3] = R_y
            # print("T_rot\n", T_rot)
            T_rot = np.linalg.inv(T_rot)

            # Apply rotations to all points for all spheres
            rotated_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h_b)  # Shape: (n_points, n_b_spheres, 4)
            rotated_points = rotated_points_h[:, :, :3]  # Shape: (n_points, n_b_spheres, 3)

            # Compute point mask based on z-component (rotated z-range)
            point_mask_b = (rotated_points[:, :, 2] >= box_min[1]) & (rotated_points[:, :, 2] <= box_max[1])  # Shape: (n_points, n_b_spheres)

        else:
            raise ValueError(f"Unknown og_type '{og_type}' for joint {joint}")

        # Check for any collision of joint with points
        j_idx = chain.index(joint) + 1  # Index of next joint
        l_nj = joint_info[chain[j_idx]][0]  # Next joint's reach
        mag = np.linalg.norm(l_nj[:2])
        nj_box_min = joint_type_info[joint]["box_min"] if joint_type_info[joint]["box_min"] is not None else np.zeros(3)
        nj_box_max = joint_type_info[joint]["box_max"] if joint_type_info[joint]["box_max"] is not None else np.zeros(3)
        magnitudes = np.linalg.norm(joint_points_b[..., :2], axis=2) # Apply transforms to next joints max and min
        new_mask = (joint_points_b[:, :, 2] >= nj_box_min[2]) & (joint_points_b[:, :, 2] <= nj_box_max[2])
        mag_mask = magnitudes < mag
        new_mask = new_mask & mag_mask
        point_mask_b = new_mask | point_mask_b

        # Align ft to x-axis and transform points for calculating q
        sigma = np.arctan2(ft_pos_b[:, 1], ft_pos_b[:, 0])  # Shape: (n_b_spheres,)

        # Create rotation matrices around z-axis
        cos_sigma = np.cos(sigma)
        sin_sigma = np.sin(sigma)
        R_z = np.array([
            [cos_sigma, -sin_sigma, np.zeros(n_b_spheres)],
            [sin_sigma, cos_sigma, np.zeros(n_b_spheres)],
            [np.zeros(n_b_spheres), np.zeros(n_b_spheres), np.ones(n_b_spheres)]
        ]).transpose(2, 0, 1)  # Shape: (n_b_spheres, 3, 3)
        # print("R_z\n", R_z)

        # Extend to 4x4 transformation matrices
        T_rot = np.tile(np.eye(4), (n_b_spheres, 1, 1))
        T_rot[:, :3, :3] = R_z
        # print("T_rot\n", T_rot)
        T_rot = np.linalg.inv(T_rot)

        new_joint_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h_b)
        joint_points_b = new_joint_points_h[:, :, :3]  # Shape: (n_points, n_b_spheres, 3)

        # # Count and print the number of points that passed the filter for each sphere
        # num_passed_points = np.sum(point_mask_b, axis=0)  # Shape: (n_b_spheres,)
        # sphere_indices = np.where(b_mask)[0]  # Shape: (n_b_spheres,)
        # for idx, count in zip(sphere_indices, num_passed_points):
        #     print(f"Sphere {idx}: {count} points passed the point_mask_b filter")

        # Solve as Type B using transformed joint points
        new_q_b = solve_for_B_joint(
            q_0_b,
            joint,
            joint_index_dict,
            centroids_b,
            joint_points_b,
            joint_info,
            point_mask_b,
            l_ft_b
        )
        q_0_all[b_mask, joint_index_dict[joint]] = new_q_b
        print(f"Type D (as B, og_type={og_type}) result for {joint}:", new_q_b)
    return

def solve_for_D_joint(q_0_all, joint, joint_index_dict, centroids, joint_points, joint_info, chain, joint_type_info, T_joint_to_fts):
    """
    Solve for type "D" joints by handling spheres based on fingertip position and original joint type.

    This function processes type "D" joints by:
    1. Computing a mask (`ft_mask`) based on fingertip position relative to a bounding box.
    2. Solving spheres where `ft_mask` is True as type "C" joints.
    3. Solving spheres where `ft_mask` is False as type "B" joints with modifications based on the original type.

    Parameters:
    - q_0_all (np.ndarray): Shape (n_spheres, n_joints), current joint angles to be updated in place.
    - joint (str): Name of the joint to solve for.
    - joint_index_dict (dict): Maps joint names to column indices in q_0_all.
    - centroids (np.ndarray): Shape (n_spheres, 3), centroids in the joint frame.
    - joint_points (np.ndarray): Shape (n_points, n_spheres, 3), points in the joint frame.
    - joint_info (dict): General joint information.
    - chain (list): Kinematic chain containing the joint.
    - joint_type_info (dict): Type-specific information for joints.
    - T_joint_to_fts (np.ndarray): Shape (n_spheres, 4, 4), transformations from joint to fingertip.

    Returns:
    - None: Updates q_0_all in place.
    """
    # Extract fingertip position and normal for all spheres
    ft_pos = T_joint_to_fts[:, :3, 3]       # Shape: (n_spheres, 3)
    l_ft = np.linalg.norm(ft_pos[:, :2], axis=1)  # Shape: (n_spheres,)
    box_min = joint_type_info[chain[-1]]["box_min"]
    box_max = joint_type_info[chain[-1]]["box_max"]
    n_spheres = centroids.shape[0]

    # Initialize point_mask for B spheres
    point_mask = np.zeros((joint_points.shape[0], n_spheres), dtype=bool)

    # Homogeneous points
    joint_points_h = np.concatenate(
        [joint_points, np.ones((joint_points.shape[0], n_spheres, 1))],
        axis=2
    )  # Shape: (n_points, n_b_spheres, 4)

    # Get original joint type
    og_type = joint_type_info[joint]["og_type"]

    if og_type == "C":
        # Compute rotation angles (alpha) for all spheres
        alpha = np.arctan2(-ft_pos[:, 1], ft_pos[:, 2])  # Shape: (n_b_spheres,)

        # Create rotation matrices around x-axis
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        R_x = np.array([
            [np.ones(n_spheres), np.zeros(n_spheres), np.zeros(n_spheres)],
            [np.zeros(n_spheres), cos_alpha, -sin_alpha],
            [np.zeros(n_spheres), sin_alpha, cos_alpha]
        ]).transpose(2, 0, 1)  # Shape: (n_b_spheres, 3, 3)
        # print("R_x\n", R_x)

        # Extend to 4x4 transformation matrices
        T_rot = np.tile(np.eye(4), (n_spheres, 1, 1))
        T_rot[:, :3, :3] = R_x
        # print("T_rot\n", T_rot)
        T_rot = np.linalg.inv(T_rot)

        # Apply rotations to all points for all spheres
        rotated_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h)  # Shape: (n_points, n_b_spheres, 4)
        rotated_points = rotated_points_h[:, :, :3]  # Shape: (n_points, n_b_spheres, 3)

        # Compute point mask based on y-component (rotated y-range)
        point_mask = (rotated_points[:, :, 1] >= box_min[1]) & (rotated_points[:, :, 1] <= box_max[1])  # Shape: (n_points, n_b_spheres)

    elif og_type == "B":
        # Compute rotation angles (beta) for all spheres
        beta = np.arctan2(-ft_pos[:, 2], ft_pos[:, 0])  # Shape: (n_b_spheres,)

        # Create rotation matrices around y-axis
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        R_y = np.array([
            [cos_beta, np.zeros(n_spheres), sin_beta],
            [np.zeros(n_spheres), np.ones(n_spheres), np.zeros(n_spheres)],
            [-sin_beta, np.zeros(n_spheres), cos_beta]
        ]).transpose(2, 0, 1)  # Shape: (n_b_spheres, 3, 3)
        # print("R_y\n", R_y)

        # Extend to 4x4 transformation matrices
        T_rot = np.tile(np.eye(4), (n_spheres, 1, 1))
        T_rot[:, :3, :3] = R_y
        # print("T_rot\n", T_rot)
        T_rot = np.linalg.inv(T_rot)

        # Apply rotations to all points for all spheres
        rotated_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h)  # Shape: (n_points, n_spheres, 4)
        rotated_points = rotated_points_h[:, :, :3]  # Shape: (n_points, n_spheres, 3)

        # Compute point mask based on z-component (rotated z-range)
        point_mask = (rotated_points[:, :, 2] >= box_min[1]) & (rotated_points[:, :, 2] <= box_max[1])  # Shape: (n_points, n_spheres)

    else:
        raise ValueError(f"Unknown og_type '{og_type}' for joint {joint}")

    # Check for any collision of joint with points
    l_nj =  joint_type_info[joint]["ft"]  # Next joint's reach 
    mag = np.linalg.norm(l_nj[:2])
    nj_box_min = joint_type_info[joint]["box_min"] if joint_type_info[joint]["box_min"] is not None else np.zeros(3)
    nj_box_max = joint_type_info[joint]["box_max"] if joint_type_info[joint]["box_max"] is not None else np.zeros(3)
    magnitudes = np.linalg.norm(joint_points[..., :2], axis=2) 
    new_mask = (joint_points[:, :, 2] >= nj_box_min[2]) & (joint_points[:, :, 2] <= nj_box_max[2])
    mag_mask = magnitudes < mag
    new_mask = new_mask & mag_mask
    point_mask = new_mask | point_mask

    # Align ft to x-axis and transform points for calculating q
    sigma = np.arctan2(ft_pos[:, 1], ft_pos[:, 0])  # Shape: (n_spheres,)

    # Create rotation matrices around z-axis
    cos_sigma = np.cos(sigma)
    sin_sigma = np.sin(sigma)
    R_z = np.array([
        [cos_sigma, -sin_sigma, np.zeros(n_spheres)],
        [sin_sigma, cos_sigma, np.zeros(n_spheres)],
        [np.zeros(n_spheres), np.zeros(n_spheres), np.ones(n_spheres)]
    ]).transpose(2, 0, 1)  # Shape: (n_spheres, 3, 3)
    # print("R_z\n", R_z)

    # Extend to 4x4 transformation matrices
    T_rot = np.tile(np.eye(4), (n_spheres, 1, 1))
    T_rot[:, :3, :3] = R_z
    # print("T_rot\n", T_rot)
    T_rot = np.linalg.inv(T_rot)

    new_joint_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h)
    # joint_points = new_joint_points_h[:, :, :3]  # Shape: (n_points, n_b_spheres, 3)

    # Solve as Type B using transformed joint points
    new_q= solve_for_B_joint(
        q_0_all,
        joint,
        joint_index_dict,
        centroids,
        joint_points,
        joint_info,
        point_mask,
        l_ft,
        joint_type_info
    )
    q_0_all[:, joint_index_dict[joint]] = new_q
    # print(f"Type D (as B, og_type={og_type}) result for {joint}:", new_q)

    return

def solve_for_D_joint_ft_correction(q_0_all, joint, joint_index_dict, centroids, joint_points, joint_info, chain, joint_type_info, T_joint_to_fts):
    """
    Solve for type "D" joints by handling spheres based on fingertip position and original joint type.

    This function processes type "D" joints by:
    1. Computing a mask (`ft_mask`) based on fingertip position relative to a bounding box.
    2. Solving spheres where `ft_mask` is True as type "C" joints.
    3. Solving spheres where `ft_mask` is False as type "B" joints with modifications based on the original type.

    Parameters:
    - q_0_all (np.ndarray): Shape (n_spheres, n_joints), current joint angles to be updated in place.
    - joint (str): Name of the joint to solve for.
    - joint_index_dict (dict): Maps joint names to column indices in q_0_all.
    - centroids (np.ndarray): Shape (n_spheres, 3), centroids in the joint frame.
    - joint_points (np.ndarray): Shape (n_points, n_spheres, 3), points in the joint frame.
    - joint_info (dict): General joint information.
    - chain (list): Kinematic chain containing the joint.
    - joint_type_info (dict): Type-specific information for joints.
    - T_joint_to_fts (np.ndarray): Shape (n_spheres, 4, 4), transformations from joint to fingertip.

    Returns:
    - None: Updates q_0_all in place.
    """
    # Extract fingertip position and normal for all spheres
    ft_pos = T_joint_to_fts[:, :3, 3]       # Shape: (n_spheres, 3)
    l_ft = np.linalg.norm(ft_pos[:, :2], axis=1)  # Shape: (n_spheres,)
    box_min = joint_type_info[chain[-1]]["box_min"]
    box_max = joint_type_info[chain[-1]]["box_max"]
    n_spheres = centroids.shape[0]

    # Initialize point_mask for B spheres
    point_mask = np.zeros((joint_points.shape[0], n_spheres), dtype=bool)

    # Homogeneous points
    joint_points_h = np.concatenate(
        [joint_points, np.ones((joint_points.shape[0], n_spheres, 1))],
        axis=2
    )  # Shape: (n_points, n_b_spheres, 4)

    # Get original joint type
    og_type = joint_type_info[joint]["og_type"]

    if og_type == "C":
        # Compute rotation angles (alpha) for all spheres
        alpha = np.arctan2(-ft_pos[:, 1], ft_pos[:, 0])  # Shape: (n_b_spheres,)

        # Create rotation matrices around x-axis
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        R_x = np.array([
            [np.ones(n_spheres), np.zeros(n_spheres), np.zeros(n_spheres)],
            [np.zeros(n_spheres), cos_alpha, -sin_alpha],
            [np.zeros(n_spheres), sin_alpha, cos_alpha]
        ]).transpose(2, 0, 1)  # Shape: (n_b_spheres, 3, 3)
        # print("R_x\n", R_x)

        # Extend to 4x4 transformation matrices
        T_rot = np.tile(np.eye(4), (n_spheres, 1, 1))
        T_rot[:, :3, :3] = R_x
        # print("T_rot\n", T_rot)
        T_rot = np.linalg.inv(T_rot)

        # Apply rotations to all points for all spheres
        rotated_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h)  # Shape: (n_points, n_b_spheres, 4)
        rotated_points = rotated_points_h[:, :, :3]  # Shape: (n_points, n_b_spheres, 3)

        # Compute point mask based on y-component (rotated y-range)
        point_mask = (rotated_points[:, :, 1] >= box_min[1]) & (rotated_points[:, :, 1] <= box_max[1])  # Shape: (n_points, n_b_spheres)

    elif og_type == "B":
        # Compute rotation angles (beta) for all spheres
        beta = np.arctan2(-ft_pos[:, 1], ft_pos[:, 0])  # Shape: (n_b_spheres,)

        # Create rotation matrices around y-axis
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        R_y = np.array([
            [cos_beta, np.zeros(n_spheres), sin_beta],
            [np.zeros(n_spheres), np.ones(n_spheres), np.zeros(n_spheres)],
            [-sin_beta, np.zeros(n_spheres), cos_beta]
        ]).transpose(2, 0, 1)  # Shape: (n_b_spheres, 3, 3)
        # print("R_y\n", R_y)

        # Extend to 4x4 transformation matrices
        T_rot = np.tile(np.eye(4), (n_spheres, 1, 1))
        T_rot[:, :3, :3] = R_y
        # print("T_rot\n", T_rot)
        T_rot = np.linalg.inv(T_rot)

        # Apply rotations to all points for all spheres
        rotated_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h)  # Shape: (n_points, n_spheres, 4)
        rotated_points = rotated_points_h[:, :, :3]  # Shape: (n_points, n_spheres, 3)

        # Compute point mask based on z-component (rotated z-range)
        point_mask = (rotated_points[:, :, 2] >= box_min[1]) & (rotated_points[:, :, 2] <= box_max[1])  # Shape: (n_points, n_spheres)

    else:
        raise ValueError(f"Unknown og_type '{og_type}' for joint {joint}")

    # Check for any collision of joint with points
    l_nj =  joint_type_info[joint]["ft"]  # Next joint's reach 
    mag = np.linalg.norm(l_nj[:2])
    nj_box_min = joint_type_info[joint]["box_min"] if joint_type_info[joint]["box_min"] is not None else np.zeros(3)
    nj_box_max = joint_type_info[joint]["box_max"] if joint_type_info[joint]["box_max"] is not None else np.zeros(3)
    magnitudes = np.linalg.norm(joint_points[..., :2], axis=2) 
    new_mask = (joint_points[:, :, 2] >= nj_box_min[2]) & (joint_points[:, :, 2] <= nj_box_max[2])
    mag_mask = magnitudes < mag
    new_mask = new_mask & mag_mask
    point_mask = new_mask | point_mask

    # Align ft to x-axis and transform points for calculating q
    sigma = np.arctan2(ft_pos[:, 1], ft_pos[:, 0])  # Shape: (n_spheres,)

    # Create rotation matrices around z-axis
    cos_sigma = np.cos(sigma)
    sin_sigma = np.sin(sigma)
    R_z = np.array([
        [cos_sigma, -sin_sigma, np.zeros(n_spheres)],
        [sin_sigma, cos_sigma, np.zeros(n_spheres)],
        [np.zeros(n_spheres), np.zeros(n_spheres), np.ones(n_spheres)]
    ]).transpose(2, 0, 1)  # Shape: (n_spheres, 3, 3)
    # print("R_z\n", R_z)

    # Extend to 4x4 transformation matrices
    T_rot = np.tile(np.eye(4), (n_spheres, 1, 1))
    T_rot[:, :3, :3] = R_z
    # print("T_rot\n", T_rot)
    T_rot = np.linalg.inv(T_rot)

    new_joint_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h)
    joint_points = new_joint_points_h[:, :, :3]  # Shape: (n_points, n_b_spheres, 3)

    # Solve as Type B using transformed joint points
    new_q= solve_for_B_joint(
        q_0_all,
        joint,
        joint_index_dict,
        centroids,
        joint_points,
        joint_info,
        point_mask,
        l_ft,
        joint_type_info
    )
    q_0_all[:, joint_index_dict[joint]] = new_q
    # print(f"Type D (as B, og_type={og_type}) result for {joint}:", new_q)

    return

def solve_for_D_joint_eigenvectors(q_0_all, joint, joint_index_dict, centroids, joint_points, joint_info, chain, joint_type_info, T_joint_to_fts):
    """
    Solve for type "D" joints by handling spheres based on fingertip position and original joint type.

    This function processes type "D" joints by:
    1. Computing a mask (`ft_mask`) based on fingertip position relative to a bounding box.
    2. Solving spheres where `ft_mask` is True as type "C" joints.
    3. Solving spheres where `ft_mask` is False as type "B" joints with modifications based on the original type.

    Parameters:
    - q_0_all (np.ndarray): Shape (n_spheres, n_joints), current joint angles to be updated in place.
    - joint (str): Name of the joint to solve for.
    - joint_index_dict (dict): Maps joint names to column indices in q_0_all.
    - centroids (np.ndarray): Shape (n_spheres, 3), centroids in the joint frame.
    - joint_points (np.ndarray): Shape (n_points, n_spheres, 3), points in the joint frame.
    - joint_info (dict): General joint information.
    - chain (list): Kinematic chain containing the joint.
    - joint_type_info (dict): Type-specific information for joints.
    - T_joint_to_fts (np.ndarray): Shape (n_spheres, 4, 4), transformations from joint to fingertip.

    Returns:
    - None: Updates q_0_all in place.
    """
    # Extract fingertip position and normal for all spheres
    ft_pos = T_joint_to_fts[:, :3, 3]       # Shape: (n_spheres, 3)
    l_ft = np.linalg.norm(ft_pos[:, :2], axis=1)  # Shape: (n_spheres,)
    box_min = joint_type_info[chain[-1]]["box_min"]
    box_max = joint_type_info[chain[-1]]["box_max"]
    n_spheres = centroids.shape[0]

    # Initialize point_mask for B spheres
    point_mask = np.zeros((joint_points.shape[0], n_spheres), dtype=bool)

    # Homogeneous points
    joint_points_h = np.concatenate(
        [joint_points, np.ones((joint_points.shape[0], n_spheres, 1))],
        axis=2
    )  # Shape: (n_points, n_b_spheres, 4)

    # Get original joint type
    og_type = joint_type_info[joint]["og_type"]

    if og_type == "C":
        # Compute rotation angles (alpha) for all spheres
        alpha = np.arctan2(-ft_pos[:, 1], ft_pos[:, 2])  # Shape: (n_b_spheres,)

        # Create rotation matrices around x-axis
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        R_x = np.array([
            [np.ones(n_spheres), np.zeros(n_spheres), np.zeros(n_spheres)],
            [np.zeros(n_spheres), cos_alpha, -sin_alpha],
            [np.zeros(n_spheres), sin_alpha, cos_alpha]
        ]).transpose(2, 0, 1)  # Shape: (n_b_spheres, 3, 3)
        # print("R_x\n", R_x)

        # Extend to 4x4 transformation matrices
        T_rot = np.tile(np.eye(4), (n_spheres, 1, 1))
        T_rot[:, :3, :3] = R_x
        # print("T_rot\n", T_rot)
        T_rot = np.linalg.inv(T_rot)

        # Apply rotations to all points for all spheres
        rotated_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h)  # Shape: (n_points, n_b_spheres, 4)
        rotated_points = rotated_points_h[:, :, :3]  # Shape: (n_points, n_b_spheres, 3)

        # Compute point mask based on y-component (rotated y-range)
        point_mask = (rotated_points[:, :, 1] >= box_min[1]) & (rotated_points[:, :, 1] <= box_max[1])  # Shape: (n_points, n_b_spheres)

    elif og_type == "B":
        # Compute rotation angles (beta) for all spheres
        beta = np.arctan2(-ft_pos[:, 2], ft_pos[:, 0])  # Shape: (n_b_spheres,)

        # Create rotation matrices around y-axis
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        R_y = np.array([
            [cos_beta, np.zeros(n_spheres), sin_beta],
            [np.zeros(n_spheres), np.ones(n_spheres), np.zeros(n_spheres)],
            [-sin_beta, np.zeros(n_spheres), cos_beta]
        ]).transpose(2, 0, 1)  # Shape: (n_b_spheres, 3, 3)
        # print("R_y\n", R_y)

        # Extend to 4x4 transformation matrices
        T_rot = np.tile(np.eye(4), (n_spheres, 1, 1))
        T_rot[:, :3, :3] = R_y
        # print("T_rot\n", T_rot)
        T_rot = np.linalg.inv(T_rot)

        # Apply rotations to all points for all spheres
        rotated_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h)  # Shape: (n_points, n_spheres, 4)
        rotated_points = rotated_points_h[:, :, :3]  # Shape: (n_points, n_spheres, 3)

        # Compute point mask based on z-component (rotated z-range)
        point_mask = (rotated_points[:, :, 2] >= box_min[1]) & (rotated_points[:, :, 2] <= box_max[1])  # Shape: (n_points, n_spheres)

    else:
        raise ValueError(f"Unknown og_type '{og_type}' for joint {joint}")

    # Check for any collision of joint with points
    l_nj =  joint_type_info[joint]["ft"]  # Next joint's reach 
    mag = np.linalg.norm(l_nj[:2])
    nj_box_min = joint_type_info[joint]["box_min"] if joint_type_info[joint]["box_min"] is not None else np.zeros(3)
    nj_box_max = joint_type_info[joint]["box_max"] if joint_type_info[joint]["box_max"] is not None else np.zeros(3)
    magnitudes = np.linalg.norm(joint_points[..., :2], axis=2) 
    new_mask = (joint_points[:, :, 2] >= nj_box_min[2]) & (joint_points[:, :, 2] <= nj_box_max[2])
    mag_mask = magnitudes < mag
    new_mask = new_mask & mag_mask
    point_mask = new_mask | point_mask

    # Align ft to x-axis and transform points for calculating q
    sigma = np.arctan2(ft_pos[:, 1], ft_pos[:, 0])  # Shape: (n_spheres,)

    # Create rotation matrices around z-axis
    cos_sigma = np.cos(sigma)
    sin_sigma = np.sin(sigma)
    R_z = np.array([
        [cos_sigma, -sin_sigma, np.zeros(n_spheres)],
        [sin_sigma, cos_sigma, np.zeros(n_spheres)],
        [np.zeros(n_spheres), np.zeros(n_spheres), np.ones(n_spheres)]
    ]).transpose(2, 0, 1)  # Shape: (n_spheres, 3, 3)
    # print("R_z\n", R_z)

    # Extend to 4x4 transformation matrices
    T_rot = np.tile(np.eye(4), (n_spheres, 1, 1))
    T_rot[:, :3, :3] = R_z
    # print("T_rot\n", T_rot)
    T_rot = np.linalg.inv(T_rot)

    new_joint_points_h = np.einsum('sij,psj->psi', T_rot, joint_points_h)
    joint_points = new_joint_points_h[:, :, :3]  # Shape: (n_points, n_b_spheres, 3)

    # Solve as Type B using transformed joint points
    new_q= solve_for_B_joint_eigenvectors(
        q_0_all,
        joint,
        joint_index_dict,
        centroids,
        joint_points,
        joint_info,
        point_mask,
        l_ft,
        joint_type_info
    )
    q_0_all[:, joint_index_dict[joint]] = new_q
    # print(f"Type D (as B, og_type={og_type}) result for {joint}:", new_q)

    return

def solve_for_D_joint_ohne_fingertip(q_0_all, joint, joint_index_dict, centroids, joint_points, joint_info, chain, joint_type_info):
    """
    Solve for type "D" joints by handling spheres based on fingertip position and original joint type.

    This function processes type "D" joints by:
    1. Computing a mask (`ft_mask`) based on fingertip position relative to a bounding box.
    2. Solving spheres where `ft_mask` is True as type "C" joints.
    3. Solving spheres where `ft_mask` is False as type "B" joints with modifications based on the original type.

    Parameters:
    - q_0_all (np.ndarray): Shape (n_spheres, n_joints), current joint angles to be updated in place.
    - joint (str): Name of the joint to solve for.
    - joint_index_dict (dict): Maps joint names to column indices in q_0_all.
    - centroids (np.ndarray): Shape (n_spheres, 3), centroids in the joint frame.
    - joint_points (np.ndarray): Shape (n_points, n_spheres, 3), points in the joint frame.
    - joint_info (dict): General joint information.
    - chain (list): Kinematic chain containing the joint.
    - joint_type_info (dict): Type-specific information for joints.
    - T_joint_to_fts (np.ndarray): Shape (n_spheres, 4, 4), transformations from joint to fingertip.

    Returns:
    - None: Updates q_0_all in place.
    """
    # Extract fingertip position and normal for all spheres
    nj_pos = np.array(joint_type_info[joint]["nj"])
    l_nj = np.linalg.norm(nj_pos[:2])

    box_min = joint_type_info[joint]["box_min"]
    box_max = joint_type_info[joint]["box_max"]

    magnitudes = np.linalg.norm(joint_points[..., :2], axis=2) 
    box_mask = (joint_points[:, :, 2] >= box_min[2]) & (joint_points[:, :, 2] <= box_max[2])
    mag_mask = magnitudes < l_nj
    point_mask = box_mask & mag_mask

    # Solve as Type B using transformed joint points
    new_q= solve_for_B_joint(
        q_0_all,
        joint,
        joint_index_dict,
        centroids,
        joint_points,
        joint_info,
        point_mask,
        l_nj,
        joint_type_info
    )
    q_0_all[:, joint_index_dict[joint]] = new_q
    print(f"Type D result for {joint}:", new_q)

    return



def save_to_hdf5(file_path, data_dict, dataset_prefix=""):
    """Save numerical data from a dictionary to an HDF5 file."""
    with h5py.File(file_path, 'w') as f:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(f"{dataset_prefix}{key}", data=value)
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                # Handle lists of arrays (e.g., meshes)
                for i, arr in enumerate(value):
                    f.create_dataset(f"{dataset_prefix}{key}_{i}", data=arr)
            else:
                print(f"Skipping {key} in {file_path}: not a NumPy array or list of arrays")

def save_to_json(file_path, data_dict):
    """Save structured data to a JSON file, ensuring serializability."""
    # Convert non-serializable types (e.g., NumPy arrays) to lists
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        elif isinstance(obj, (np.float64, np.int64)):
            return float(obj) if isinstance(obj, np.float64) else int(obj)
        return obj
    
    serializable_data = convert_to_serializable(data_dict)
    with open(file_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)

def get_main_joints_q(joints, joint_info, joint_type_info, q_dict, kin_chains, base_link):
    """
    Adjusts joint values in q_dict for main joints to align the sphere center with their x-axis.
    
    Parameters:
    - joints (list): List of main joint names (or None).
    - joint_info (dict): Maps joint names to (xyz, quat, joint_type, axis, lower, upper, length).
    - joint_type_info (dict): Type-specific joint information.
    - q_dict (dict): Current joint angles to be updated in place.
    - kin_chains (list): Kinematic chains defining joint hierarchy.
    
    Returns:
    - dict: Updated q_dict with adjusted joint values.
    """
    # Get sphere center in base frame
    p_s = joint_info["sphere_frame"][0]
    
    for joint in joints:
        if joint is None:
            continue
            
        # Initial joint value and limits
        q_initial = q_dict[joint]
        lower = joint_info[joint][4]
        upper = joint_info[joint][5]
        
        # Compute transformation from base to joint with current q_dict
        T_base_to_j = compute_transform_to_joint(joint, kin_chains, joint_info, q_dict, base_link)
        T_j_to_base = np.linalg.inv(T_base_to_j)
        
        # Transform sphere center to joint frame
        p_s_hom = np.append(p_s, 1)
        p_s_j = (T_j_to_base @ p_s_hom)[:3]
        
        # Compute rotation angle to align sphere center with x-axis
        delta_q = np.arctan2(p_s_j[1], p_s_j[0])
        
        # Adjust delta_q to preserve the sign of the initial x-coordinate
        print("q To adjust to", delta_q)
        print("center", p_s_j)
        print("initial", q_initial)
        print("bounds", upper, lower)
        if p_s_j[0] < 0 and p_s_j[1] < 0:
            delta_q += np.pi
        elif p_s_j[0] < 0 and p_s_j[1] > 0:
            delta_q -= np.pi

        q_j = np.clip(q_initial + delta_q,lower,upper)
        # Update q_dict
        q_dict[joint] = q_j
        print(f"Adjusted {joint} from {q_initial:.4f} to {q_j:.4f}")
    return 


def compute_deformed_sphere_points(
        d_planes_theta, d_plane_offsets, d_vectors_phi, 
        d_vector_offsets, n_pole_values, s_pole_values, points, base_radius=1.0):
    """
    Compute (x, y, z) coordinates for points on deformed spheres, incorporating pole values into the interpolation.

    Parameters:
    - d_planes_theta: 1D array, driving plane angles (radians).
    - d_plane_offsets: 2D array, theta offsets (n_driving_planes, n_spheres).
    - d_vectors_phi: 1D array, driving vector phi angles (radians), assumed sorted.
    - d_vector_offsets: 3D array, radius offsets (n_driving_planes, n_driving_vectors, n_spheres).
    - n_pole_values: 1D array, north pole radius offsets (n_spheres,).
    - s_pole_values: 1D array, south pole radius offsets (n_spheres,).
    - points: 2D array, points to sample in sphere (theta, phi) (n_points,2).
    - base_radius: float, base radius of the sphere (default: 1.0).

    Returns:
    - xyz: 3D array, (n_points, n_spheres, 3), (x, y, z) coordinates.
    """
    # Extract dimensions from input arrays
    n_spheres = d_plane_offsets.shape[1]
    n_driving_planes = len(d_planes_theta)

    # Extend driving vectors to include poles (phi=0 and phi=pi)
    extended_d_vectors_phi = np.concatenate(([0], d_vectors_phi, [np.pi]))  # Shape: (n_driving_vectors + 2,)

    # Extend radius offsets to include pole values, replicated across all driving planes
    n_pole_offsets = np.tile(n_pole_values[None, None, :], (n_driving_planes, 1, 1))  # Shape: (n_driving_planes, 1, n_spheres)
    s_pole_offsets = np.tile(s_pole_values[None, None, :], (n_driving_planes, 1, 1))  # Shape: (n_driving_planes, 1, n_spheres)
    extended_d_vector_offsets = np.concatenate([n_pole_offsets, d_vector_offsets, s_pole_offsets], axis=1)
    # Shape: (n_driving_planes, n_driving_vectors + 2, n_spheres)

    # Get s
    n_points = points.shape[0]
    theta = points[:,0]
    phi = points[:,1]

    # Adjust driving plane angles with offsets and handle periodicity
    adjusted_d_planes_theta = (d_planes_theta[:, None] + d_plane_offsets) % (2 * np.pi)
    # Extend theta to handle periodicity: [-2π, 0], [0, 2π], [2π, 4π]
    extended_theta = np.concatenate([adjusted_d_planes_theta - 2 * np.pi, adjusted_d_planes_theta, adjusted_d_planes_theta + 2 * np.pi], axis=0)
    sort_idx = np.argsort(extended_theta, axis=0)  # Sort for each sphere
    extended_theta_sorted = np.take_along_axis(extended_theta, sort_idx, axis=0)

    # Extend and sort radius offsets to match extended theta
    extended_offsets = np.tile(extended_d_vector_offsets, (3, 1, 1))  # Shape: (3*n_driving_planes, n_driving_vectors + 2, n_spheres)
    extended_offsets_sorted = np.take_along_axis(extended_offsets, sort_idx[:, None, :], axis=0)

    # Find indices for theta interpolation
    idx = np.zeros((n_points, n_spheres), dtype=int)
    for s in range(n_spheres):
        idx[:, s] = np.searchsorted(extended_theta_sorted[:, s], theta, side='right')
    idx = np.clip(idx, 1, 3 * n_driving_planes - 1)  # Ensure valid range for interpolation

    # Get surrounding theta values for interpolation
    theta_a = extended_theta_sorted[idx - 1, np.arange(n_spheres)]
    theta_b = extended_theta_sorted[idx, np.arange(n_spheres)]

    # Compute interpolation fraction between theta_a and theta_b
    fraction = (theta[:, None] - theta_a) / (theta_b - theta_a)  # Should be in [0, 1] due to periodicity handling

    # Compute phi interpolation indices for all points
    idx_phi = np.searchsorted(extended_d_vectors_phi, phi, side='right') - 1
    idx_phi = np.clip(idx_phi, 0, len(extended_d_vectors_phi) - 2)  # Ensure valid range

    # Compute phi interpolation weights
    phi_a = extended_d_vectors_phi[idx_phi]
    phi_b = extended_d_vectors_phi[idx_phi + 1]
    weight_phi = (phi - phi_a) / (phi_b - phi_a)  # Should be in [0, 1] since phi is within [0, π]

    # Extract indices for surrounding planes
    row_indices_a = idx - 1  # Indices for plane a
    row_indices_b = idx      # Indices for plane b

    # Interpolate along phi for each surrounding plane
    offset_a_at_phi = np.zeros((n_points, n_spheres))
    offset_b_at_phi = np.zeros((n_points, n_spheres))

    for s in range(n_spheres):
        # Extract offsets for surrounding planes for this sphere
        offsets_a = extended_offsets_sorted[row_indices_a[:, s], :, s]  # Shape: (n_points, n_driving_vectors + 2)
        offsets_b = extended_offsets_sorted[row_indices_b[:, s], :, s]  # Shape: (n_points, n_driving_vectors + 2)

        # Linear interpolation along phi
        offset_a_at_phi[:, s] = (1 - weight_phi) * offsets_a[np.arange(n_points), idx_phi] + weight_phi * offsets_a[np.arange(n_points), idx_phi + 1]
        offset_b_at_phi[:, s] = (1 - weight_phi) * offsets_b[np.arange(n_points), idx_phi] + weight_phi * offsets_b[np.arange(n_points), idx_phi + 1]

    # Interpolate between surrounding planes using theta fraction
    radius_offsets = offset_a_at_phi + fraction * (offset_b_at_phi - offset_a_at_phi)


    # Compute final radius and convert to Cartesian coordinates
    r = base_radius*(1 + radius_offsets)  # Total radius per point and sphere
    x = r * np.sin(phi)[:, None] * np.cos(theta)[:, None]
    y = r * np.sin(phi)[:, None] * np.sin(theta)[:, None]
    z = r * np.cos(phi)[:, None]
    xyz = np.stack([x, y, z], axis=2)  # Shape: (n_points, n_spheres, 3)

    return xyz, phi, theta

def find_type_A_joint_in_chain(chain, type_A_joints):
    """Find the Type A joint in the given chain."""
    for joint in chain:
        if joint in type_A_joints:
            return joint
    return None  # If no Type A joint is found in the chain

def compute_sphere_to_single_joint_transforms(joint_tag, q_0_all, joints_idx, joint_info, chain):
    """
    Compute transformations from sphere_frame to a single target joint for all configurations in q_0_all.

    Parameters:
    - joint_tag (str): Name of the target joint.
    - q_0_all (np.ndarray): Array of shape (n_spheres, n_joints) with joint angles for each configuration.
    - joints_idx (dictionary): Dictionary of joint:idx corresponding to columns in q_0_all.
    - joint_info (dict): Dictionary mapping joint names to (xyz, quat, type, axis, lower, upper, length).
    - chain (list): List defining the joint hierarchy from base_link.

    Returns:
    - np.ndarray: Array of shape (n_spheres, 4, 4) with transformations from sphere_frame to joint_tag.
    """
    # Extract the path from base_link to joint_tag
    path = chain[:chain.index(joint_tag) + 1]  # From base_link to joint_tag inclusive

    # Number of configurations
    n_spheres = q_0_all.shape[0]

    # Compute fixed transformation from sphere_frame to base_link # can be precomputed
    T_base_to_sphere = tf.quaternion_matrix(joint_info["sphere_frame"][1])
    T_base_to_sphere[:3, 3] = joint_info["sphere_frame"][0]
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)  # Shape: (4, 4)
    # print("Sphere Transform \n", T_sphere_to_base)

    # Initialize cumulative transformation as identity for all spheres
    T_cumulative = np.eye(4)[np.newaxis].repeat(n_spheres, axis=0)  # Shape: (n_spheres, 4, 4)

    # Compute transformations along the kinematic path
    # print("path", path)

    for joint in path[1:]:  # Skip base_link
        # Get joint angles; use zeros if joint not in q_0_all
        if joint in joints_idx:
            q_values = q_0_all[:, joints_idx[joint]]  # Shape: (n_spheres,)
        else:
            q_values = np.zeros(n_spheres)

        # Extract joint info
        xyz, quat, joint_type = joint_info[joint][:3]

        # Fixed transformation for the joint
        T_fixed = tf.quaternion_matrix(quat)
        T_fixed[:3, 3] = xyz  # Shape: (4, 4)
        # print(xyz, quat, joint_type)
        

        if joint_type == "revolute":
            # Vectorized rotation matrices around z-axis
            cos_q = np.cos(q_values)
            sin_q = np.sin(q_values)
            R_z = np.array([
                [cos_q, -sin_q, np.zeros(n_spheres)],
                [sin_q, cos_q, np.zeros(n_spheres)],
                [np.zeros(n_spheres), np.zeros(n_spheres), np.ones(n_spheres)]
            ]).transpose(2, 0, 1)  # Shape: (n_spheres, 3, 3)

            # Extend to 4x4 matrices
            T_rot = np.eye(4)[np.newaxis].repeat(n_spheres, axis=0)
            T_rot[:, :3, :3] = R_z  # Shape: (n_spheres, 4, 4)

            # Combine fixed and rotation transformations
            T_joint = T_fixed @ T_rot  # Shape: (n_spheres, 4, 4)
        else: 
            # For fixed joints, tile the fixed transformation
            T_joint = np.tile(T_fixed, (n_spheres, 1, 1))  # Shape: (n_spheres, 4, 4)

        # Update cumulative transformation
        T_cumulative = T_cumulative @ T_joint  # Shape: (n_spheres, 4, 4)
        # print("joint\n",T_joint)
        # print("cummulative\n",T_cumulative)

    # Final transformation from sphere_frame to joint_tag
    T_sphere_to_joint_all = T_sphere_to_base @ T_cumulative  # Shape: (n_spheres, 4, 4)
    # print(T_sphere_to_joint_all)

    return T_sphere_to_joint_all

def solve_for_proto_B_joint(q_0, joint_tag, joint_index_dict, joint_points, centroid, joint_info, joint_type_info):
    """Compute updated joint angle for a B joint (or non-first A) based on palm_normal points."""
    B_joint_idx = joint_index_dict[joint_tag]
    lower_limit = joint_info[joint_tag][4]
    upper_limit = joint_info[joint_tag][5]
    q_0_joint = q_0[B_joint_idx]

    if joint_type_info[joint_tag]["type"] == "A":
        joint_points = np.vstack((joint_points, centroid.reshape(1, -1)))

    angles = np.arctan2(joint_points[:, 1], joint_points[:, 0])

    masked_angles = np.ma.masked_array(angles, mask=np.zeros_like(angles, dtype=bool))

    ref_dir = joint_type_info[joint_tag]["ref_dir"]
    if ref_dir[1] > 0:
        result = np.min(masked_angles)
    else:
        result = np.max(masked_angles)

    has_valid_points = np.any(~masked_angles.mask)
    centroid_angle = np.arctan2(centroid[1], centroid[0])
    result = result if has_valid_points else centroid_angle

    new_q = q_0_joint + result
    new_q = np.clip(new_q, lower_limit, upper_limit)
    return new_q

def solve_for_proto_C_joint(q_0, joint_tag, joint_index_dict, centroid, joint_info, ft_pos, ft_normal):
    """Compute updated joint angle for a C joint based on palm_normal centroid and fingertip position."""
    C_joint_idx = joint_index_dict[joint_tag]
    lower_limit = joint_info[joint_tag][4]
    upper_limit = joint_info[joint_tag][5]
    q_0_joint = q_0[C_joint_idx]

    d_c_ft = centroid - ft_pos
    theta_d = np.arctan2(d_c_ft[1], d_c_ft[0])
    theta_v = np.arctan2(ft_normal[1], ft_normal[0])
    result = theta_d - theta_v

    new_q = q_0_joint + result
    new_q = np.clip(new_q, lower_limit, upper_limit)
    return new_q

def compute_transform_from_joint_to_fingertip(chain, joint, joint_info, q_dict=None):
    """
    Compute the 4x4 transformation matrix from a specific joint to the fingertip in a kinematic chain.

    Parameters:
    - chain (list): A kinematic chain as a list of joint names from base to fingertip 
                    (e.g., ["base_link", "joint1", "joint2", "fingertip"]).
    - specific_joint (str): The joint from which to compute the transformation to the fingertip.
    - joint_info (dict): Maps joint names to (xyz, quat, joint_type, axis, lower, upper, length).
                         Each tuple contains position, orientation, type, etc., relative to the parent.
    - q_dict (dict, optional): Joint angles for revolute joints; if None, assumes zero configuration.

    Returns:
    - np.ndarray: 4x4 transformation matrix from specific_joint to the fingertip.

    Raises:
    - ValueError: If specific_joint is not in the chain or is the fingertip itself.
    """
    # Validate inputs
    if joint not in chain:
        raise ValueError(f"Specific joint '{joint}' not found in the chain")
    
    idx = chain.index(joint)
    if idx == len(chain) - 1:
        return np.eye(4)
    
    # Extract the subchain from the child of specific_joint to fingertip
    subchain = chain[idx + 1:] 

    # Initialize the transformation matrix as identity
    T_joint_to_fingertip = np.eye(4)
    
    # Accumulate transformations from the child of specific_joint to fingertip
    for child in subchain:
        # Retrieve joint properties
        xyz, quat, joint_type, _, _, _, _ = joint_info[child]  # Unpack position, orientation, type
        
        # Build the static transformation from parent to child
        T = tf.quaternion_matrix(quat)  # 4x4 matrix from quaternion
        T[:3, 3] = xyz                  # Set translation
        
        # Apply rotation for revolute joints if angle is provided
        if q_dict is not None and joint_type == "revolute":
            q_val = q_dict.get(child, 0.0)  # Default to 0 if not in q_dict
            R_z = tf.rotation_matrix(q_val, [0, 0, 1])  # Rotation around z-axis (common convention)
            T = T @ R_z  # Compose the static transform with the joint rotation
        
        # Accumulate the transformation
        T_joint_to_fingertip = T_joint_to_fingertip @ T
    
    return T_joint_to_fingertip

def compute_type_A_joints(joint_info, kinematic_chains, q_0, q_max, joint_type_info, base_link = "base_link"):
    """Identify Type 'A' joints and their theta ranges for each finger.

    Parameters:
    - joint_info (dict): Maps joint names to (xyz, quat, joint_type, axis, lower, upper, length).
    - kinematic_chains (list of lists): Defines joint hierarchy.
    - q_0 (dict): Initial joint angles.
    - q_max (dict): .
    - joint_type_info:

    Returns:
    - tuple: (type_A_joints, theta_ranges) where type_A_joints is a list of Type "A" joints
             (None if none), and theta_ranges is a list of (min_theta, max_theta) tuples
             (None if no Type "A" joint), with ranges computed around the circular midpoint.
    """
    # Initialize lists
    type_A_joints = []
    theta_ranges = []

    # Sphere frame transformation
    sphere_xyz = joint_info["sphere_frame"][0]
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3, 3] = sphere_xyz
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)

    # Palm frame transformation
    palm_xyz = joint_info["palm_normal"][0]
    palm_quat = joint_info["palm_normal"][1]
    T_base_to_palm = tf.quaternion_matrix(palm_quat)
    T_base_to_palm[:3, 3] = palm_xyz
    T_palm_to_base = np.linalg.inv(T_base_to_palm)

    # Filter finger chains
    finger_chains = [
        chain for chain in kinematic_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]

    # Sphere radius
    radius = joint_info["sphere_frame"][6]

    # Process each finger chain
    for finger_chain in finger_chains:
        fingertip = finger_chain[-1]

        # Get revolute joints (exclude base_link and fingertip)
        joints = [j for j in finger_chain[1:-1] if joint_info[j][2] == "revolute"]

        # Track best Type "A" candidate
        max_theta_range = 0
        type_A_candidate = None
        min_theta_candidate = None
        max_theta_candidate = None

        # Check if finger has an azimuthal joint alrad
        azimuthal_joint_flag = False
        A_joints = []
        for joint in joints:
            if joint_type_info[joint]["type"] == "A":
                azimuthal_joint_flag = True
                A_joints.append(joint)

        if azimuthal_joint_flag: 
            print(f"Finger Chain {finger_chain} has an A type joints {A_joints}")


            for joint in A_joints:
                lower = joint_info[joint][4]
                upper = joint_info[joint][5]
                angles = np.linspace(lower, upper, 100)

                theta_values = []
                xy_projections = []


                # Sample all joint angles
                for angle in angles:
                    q = q_0.copy() 
                    q[joint] = angle

                    # Get transform to fingertip with the q changed
                    T_base_to_fingertip = compute_transform_to_joint(
                        fingertip, kinematic_chains, joint_info, q, base_link
                    )
                    p_fingertip_base = T_base_to_fingertip[:3, 3]

                    # Check z in palm frame, eliminate any below palm normal  
                    p_fingertip_palm_h = T_palm_to_base @ np.append(p_fingertip_base, 1)

                    # Transform to sphere frame
                    p_fingertip_sphere_h = T_sphere_to_base @ np.append(p_fingertip_base, 1)
                    p_fingertip_sphere = p_fingertip_sphere_h[:3]

                    # Get Azimuthal angle of every q value
                    theta = np.arctan2(p_fingertip_sphere[1], p_fingertip_sphere[0])

                    theta_values.append(theta)
                    xy_projections.append(p_fingertip_sphere[:2]) # For exclusion condition


                if not theta_values:
                    continue

                # Compute circular midpoint and range
                theta_values = np.array(theta_values)
                theta_mid = np.arctan2(np.mean(np.sin(theta_values)), np.mean(np.cos(theta_values)))
                shifted_theta = np.mod(theta_values - theta_mid + np.pi, 2 * np.pi) - np.pi
                min_shift = np.min(shifted_theta)
                max_shift = np.max(shifted_theta)

                # Answer can now be from -3π to 3π, accounts for circularity
                min_theta = theta_mid + min_shift
                max_theta = theta_mid + max_shift
                theta_range = max_theta - min_theta
                print(f"{joint} theta range {theta_range}")

                # Update candidate
                if theta_range > max_theta_range:
                    max_theta_range = theta_range
                    type_A_candidate = joint
                    min_theta_candidate = min_theta
                    max_theta_candidate = max_theta
            
            # If multiple type A's, deactivate one
            if len(A_joints)> 1:
                for joint in A_joints:
                    if joint!= type_A_candidate:
                        joint_type_info[joint]["type"] = None
            joint_type_info[type_A_candidate]["azimuthal_flag"] = True
        else: 
            print(f"Finger Chain {finger_chain} has no A type joints, searching for the best candidate")
            # If there is no joint, evaluate the other BC joints and get the best candidate.
            for joint in joints:
                lower = joint_info[joint][4]
                upper = joint_info[joint][5]
                angles = np.linspace(lower, upper, 100)

                theta_values = []
                xy_projections = []


                # Sample all joint angles
                for angle in angles:
                    if joint_type_info[joint]["type"] == "C":
                        q = q_max.copy()
                    else:
                        q = q_0.copy() 
                    q[joint] = angle

                    # Get transform to fingertip with the q changed
                    T_base_to_fingertip = compute_transform_to_joint(
                        fingertip, kinematic_chains, joint_info, q, base_link
                    )
                    p_fingertip_base = T_base_to_fingertip[:3, 3]

                    # Check z in palm frame, eliminate any below palm normal
                    p_fingertip_palm_h = T_palm_to_base @ np.append(p_fingertip_base, 1)
                    if p_fingertip_palm_h[2] <= 0:
                        continue

                    # Transform to sphere frame
                    p_fingertip_sphere_h = T_sphere_to_base @ np.append(p_fingertip_base, 1)
                    p_fingertip_sphere = p_fingertip_sphere_h[:3]

                    # Get Azimuthal angle of every q value
                    theta = np.arctan2(p_fingertip_sphere[1], p_fingertip_sphere[0])

                    theta_values.append(theta)
                    xy_projections.append(p_fingertip_sphere[:2]) # For exclusion condition

                if not theta_values:
                    continue

                # Compute circular midpoint and range
                theta_values = np.array(theta_values)
                theta_mid = np.arctan2(np.mean(np.sin(theta_values)), np.mean(np.cos(theta_values)))
                shifted_theta = np.mod(theta_values - theta_mid + np.pi, 2 * np.pi) - np.pi
                min_shift = np.min(shifted_theta)
                max_shift = np.max(shifted_theta)

                # Answer can now be from -3π to 3π, accounts for circularity
                min_theta = theta_mid + min_shift
                max_theta = theta_mid + max_shift
                theta_range = max_theta - min_theta
                print(f"{joint} theta range {theta_range}")

                # Update candidate
                xy_projections = np.array(xy_projections)
                if len(xy_projections) > 1: # Check that the finger joint, doesn't pass through the sphere center directly. 
                    centroid = np.mean(xy_projections, axis=0)
                    U, S, Vt = np.linalg.svd(xy_projections - centroid)
                    direction = Vt[0]
                    t = -np.dot(centroid, direction)
                    closest_point = centroid + t * direction
                    distance = np.linalg.norm(closest_point)
                    print(f"{joint} distance {distance} {radius}")
                    if distance < 0.15 * radius: # Threshold
                        continue
                    
                if theta_range > max_theta_range:
                    max_theta_range = theta_range
                    type_A_candidate = joint
                    min_theta_candidate = min_theta
                    max_theta_candidate = max_theta

            joint_type_info[type_A_candidate]["azimuthal_flag"] = False
            

        # Assign Type "A" joint if conditions are met
        if type_A_candidate and max_theta_range > np.deg2rad(10):
            type_A_joints.append(type_A_candidate)
            theta_ranges.append((min_theta_candidate, max_theta_candidate))
        else:
            type_A_joints.append(None)
            theta_ranges.append(None)

    return type_A_joints, theta_ranges

def construct_sphere(vertices_dict, joint_info, kinematic_chains, q_0, q_max, base_link="base_link"):
    """Create the sphere frame based on palm_normal and finger roots.

    Parameters:
    - vertices_dict (dict): Maps joint names to vertex arrays.
    - joint_info (dict): Maps joint names to (xyz, quat, joint_type, axis, lower, upper, length).
    - kinematic_chains (list of lists): Defines the hierarchy of joints from base_link to end effectors.
    - q_0 (dict): Joint values for initial configuration.
    - q_max (dict): Joint values for maximum reach configuration.

    Returns:
    - np.ndarray: 4x4 transformation matrix from base_link to sphere frame.
    """

    # Build a parent map from kinematic chains
    parent_map = {}
    for chain in kinematic_chains:
        for i in range(1, len(chain)):
            parent_map[chain[i]] = chain[i - 1]

    # Validate q_0 if provided
    if q_0 is not None:
        all_joints = set()
        for chain in kinematic_chains:
            all_joints.update(chain[1:-1])  # Exclude base_link
        missing_joints = all_joints - set(q_0.keys())
        if missing_joints:
            raise ValueError(f"Missing q_0 values for joints: {missing_joints}")
        print("Joint values:", q_0)

    palm_chain = next((chain for chain in kinematic_chains if "palm_normal" in chain), None)

    # Identify finger chains and their roots
    finger_chains = [chain for chain in kinematic_chains if chain != palm_chain]
    finger_roots = [chain[1] for chain in finger_chains if len(chain) > 1]

    if not finger_roots:
        raise ValueError("No finger roots found in kinematic chains")

    # Identify finger chains (excluding palm_normal chain)
    fingertips = [chain[-1] for chain in finger_chains]  # Last joint in each chain

    # Compute finger root positions in base_link frame
    root_positions_base = [np.array(joint_info[root][0]) for root in finger_roots]
    root_min = np.min(root_positions_base, axis=0)
    root_max = np.max(root_positions_base, axis=0)
    root_mid = root_min + (root_max - root_min) / 2

    # Compute fingertip positions in base_link frame at q_max
    fingertip_positions_base = []
    for fingertip in fingertips:
        # Compute transformation from base_link to fingertip at q_max
        T_base_to_fingertip = compute_transform_to_joint(fingertip, kinematic_chains, joint_info, q=q_max, base_link=base_link)
        # Extract position in base frame
        p_fingertip_base = T_base_to_fingertip[:3, 3]
        fingertip_positions_base.append(p_fingertip_base)

    ft_min = np.min(fingertip_positions_base, axis=0)
    ft_max = np.max(fingertip_positions_base, axis=0)
    ft_mid = ft_min + (ft_max - ft_min) / 2

    # Calculate the midpoint between root_mean and ft_mean as the sphere frame origin
    origin = (root_mid + ft_mid) / 2

    # Calculate the direction from ft_mean to root_mean and normalize to get z-axis
    direction = root_mid - ft_mid
    z_axis = direction / np.linalg.norm(direction)

    # Construct the rotation matrix for the sphere frame with z-axis aligned to the direction
    # Choose a reference vector to define the orthogonal basis
    ref = np.array([1, 0, 0]) if np.abs(np.dot(z_axis, [1, 0, 0])) < 0.999 else np.array([0, 1, 0])
    # Compute y-axis as cross product of z and reference, normalized
    y_axis = np.cross(z_axis, ref)
    y_axis /= np.linalg.norm(y_axis)
    # Compute x-axis as cross product of y and z, normalized
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    # Stack to form rotation matrix
    R = np.stack((x_axis, y_axis, z_axis), axis=1)

    # Compute radius as half the distance between root_mean and ft_mean
    radius = (np.linalg.norm(direction) / 2)
    print(f"Sphere Radius: {radius}")

    # Construct T_base_to_sphere directly
    T_base_to_sphere = np.eye(4)
    T_base_to_sphere[:3, :3] = R
    T_base_to_sphere[:3, 3] = origin
    print("Sphere center in base_link", origin)

    # Print the results (you can modify this to return or store them instead)
    for i, pos in enumerate(fingertip_positions_base):
        print(f"Fingertip {fingertips[i]} position in base_link frame at q_max: {pos}")

    # Extract translation and quaternion from T_base_to_sphere
    xyz_sphere = origin
    quat_sphere = tf.quaternion_from_matrix(T_base_to_sphere)
    # Add to joint_info
    joint_info["sphere_frame"] = (
        xyz_sphere,  # Translation
        quat_sphere,  # Quaternion
        "fixed",  # Type (fixed frame)
        None,  # Axis (not applicable)
        None,  # Lower limit
        None,  # Upper limit
        radius  # Radius in length field
    )

    # Add the sphere_frame chain
    kinematic_chains.append([base_link, "sphere_frame"])

    # Generate sphere points in sphere frame
    p = sample_fibonacci_points(500)
    theta, phi = p[:,0], p[:,1]
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    sphere_points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    # Generate sphere x-axis points in sphere frame
    radius_offset = np.linspace(1, 1.5, 5) * radius  # 20 samples for polar angle
    phi = np.linspace(0, np.pi, 20)  # 40 samples for polar angle
    phi, radius_offset = np.meshgrid(phi, radius_offset)
    x = radius_offset * np.sin(phi)
    z = radius_offset * np.cos(phi)
    axis_points = np.stack((x.flatten(), np.zeros(z.flatten().shape), z.flatten()), axis=1)

    # Add center point
    center_point = np.array([[0, 0, 0]])
    all_points = np.vstack((center_point, sphere_points, axis_points))

    # Add to vertices_dict
    vertices_dict["sphere_frame"] = all_points

    return len(axis_points)


def allign_sphere_frame(joint_info, kinematic_chains, type_A_joints, theta_ranges):
    """Align the sphere frame x-axis to the circular average of candidate finger root thetas.

    Parameters:
    - joint_info (dict): Joint information dictionary.
    - kinematic_chains (list of lists): Kinematic chains for fingers.
    - type_A_joints (list): Type "A" joints from compute_type_A_joints.
    - theta_ranges (list): Theta ranges (min_theta, max_theta) for each Type "A" joint.

    Updates:
    - joint_info["sphere_frame"]: With the new transformation.
    """
    # Sphere frame transformation
    sphere_xyz = joint_info["sphere_frame"][0]
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3, 3] = sphere_xyz
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)

    # Identify finger chains
    finger_chains = [
        chain for chain in kinematic_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]

    # Compute non-overlapping theta ranges
    non_overlapping_sizes = []
    non_overlapping_midpoints = []
    original_midpoints = []

    for i, (joint, theta_range) in enumerate(zip(type_A_joints, theta_ranges)):
        if joint is None or theta_range is None:
            non_overlapping_sizes.append(0)
            non_overlapping_midpoints.append(None)
            original_midpoints.append(None)
            continue

        min_theta, max_theta = theta_range
        other_ranges = [
            r for j, r in enumerate(theta_ranges)
            if j != i and r is not None
        ]

        # Compute original midpoint
        original_midpoint = circular_midpoint(min_theta, max_theta)
        original_midpoints.append(original_midpoint)

        if not other_ranges:
            size = max_theta - min_theta
            non_overlapping_sizes.append(size)
            non_overlapping_midpoints.append(original_midpoint)
            continue

        # Create union with ±2π, ±4π shifts of other ranges
        union_points = []
        for r_min, r_max in other_ranges:
            union_points.append((r_min, 'start'))
            union_points.append((r_max, 'end'))
            union_points.append((r_min - 2 * np.pi, 'start'))
            union_points.append((r_max - 2 * np.pi, 'end'))
            union_points.append((r_min + 2 * np.pi, 'start'))
            union_points.append((r_max + 2 * np.pi, 'end'))
            union_points.append((r_min - 4 * np.pi, 'start'))
            union_points.append((r_max - 4 * np.pi, 'end'))
            union_points.append((r_min + 4 * np.pi, 'start'))
            union_points.append((r_max + 4 * np.pi, 'end'))

        # Sort points for sweep-line algorithm
        union_points.sort()
        active = 0
        union_segments = []
        current_start = None
        for point, event in union_points:
            if event == 'start':
                if active == 0:
                    current_start = point
                active += 1
            else:
                active -= 1
                if active == 0:
                    union_segments.append((current_start, point))

        # Find non-overlapping segments
        non_overlap = []
        current_pos = min_theta
        for u_min, u_max in union_segments:
            if u_max <= min_theta or u_min >= max_theta:
                continue
            if current_pos < u_min and u_min > min_theta:
                non_overlap.append((current_pos, u_min))
            current_pos = max(current_pos, u_max)
        if current_pos < max_theta:
            non_overlap.append((current_pos, max_theta))

        # Compute size and midpoint
        if non_overlap:
            size = sum(end - start for start, end in non_overlap)
            non_overlap_angles = []
            for start, end in non_overlap:
                non_overlap_angles.extend(np.linspace(start, end, 100))
            midpoint = circular_midpoint(non_overlap[0][0], non_overlap[-1][1])
        else:
            size = 0
            midpoint = None

        non_overlapping_sizes.append(size)
        non_overlapping_midpoints.append(midpoint)

    # Find candidate fingers
    min_size = min(non_overlapping_sizes)
    candidate_indices = [
        i for i, size in enumerate(non_overlapping_sizes) if size == min_size
    ]
    print("Non-overlapping range sizes:", non_overlapping_sizes)
    print("Non-overlapping range midpoints:", non_overlapping_midpoints)
    print("Candidate indices:", candidate_indices)

    # Align sphere frame to circular average of candidate root thetas
    if candidate_indices:
        root_thetas = []
        for idx in candidate_indices:
            finger = finger_chains[idx]
            root_joint = finger[1]  # First joint after base_link
            p_root_base = joint_info[root_joint][0]
            p_root_sphere_h = T_sphere_to_base @ np.append(p_root_base, 1)
            p_root_sphere = p_root_sphere_h[:3]
            theta = np.arctan2(p_root_sphere[1], p_root_sphere[0])
            root_thetas.append(theta)
        
        if root_thetas:
            # Compute circular average
            mean_theta = np.arctan2(np.mean(np.sin(root_thetas)), np.mean(np.cos(root_thetas)))
            print("Mean theta of candidate roots:", mean_theta)
            
            # Rotate sphere frame around z-axis by mean_theta
            R_align = tf.rotation_matrix(mean_theta, [0, 0, 1])
            T_base_to_sphere = T_base_to_sphere @ R_align
            
            # Update joint_info["sphere_frame"]
            xyz_sphere = T_base_to_sphere[:3, 3]
            quat_sphere = tf.quaternion_from_matrix(T_base_to_sphere)
            joint_info["sphere_frame"] = (
                xyz_sphere,
                quat_sphere,
                "fixed",
                None,
                None,
                None,
                joint_info["sphere_frame"][6]  # Preserve radius
            )
            return mean_theta, non_overlapping_midpoints
        else:
            return 0, non_overlapping_midpoints
    else:
        print("No candidate fingers found; sphere frame alignment unchanged.")
        return 0, non_overlapping_midpoints
    
def normalize_finger_chains(joint_info, kinematic_chains, q_0, base_link="base_link"):
    """Normalize the length of each finger chain to pi * radius by adjusting the fingertip position along its local x-axis direction."""

    # Extract sphere information (assuming construct_sphere has been called prior)
    if "sphere_frame" not in joint_info:
        raise ValueError("Sphere frame not found in joint_info. Call construct_sphere first.")
    
    radius = joint_info["sphere_frame"][6]
    target_length = np.pi * radius

    # Compute T_base_to_sphere
    xyz_sphere = joint_info["sphere_frame"][0]
    quat_sphere = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(quat_sphere)
    T_base_to_sphere[:3, 3] = xyz_sphere

    # North pole in base frame
    north_pole_sphere = np.array([0, 0, radius, 1])
    north_pole_base = (T_base_to_sphere @ north_pole_sphere)[:3]

    # Identify finger chains (excluding palm_normal and sphere_frame chains)
    palm_chain = next((chain for chain in kinematic_chains if "palm_normal" in chain), None)
    sphere_chain = next((chain for chain in kinematic_chains if "sphere_frame" in chain), None)
    finger_chains = [chain for chain in kinematic_chains if chain not in [palm_chain, sphere_chain]]

    # Process each finger chain
    for chain in finger_chains:
        # Compute root position in base frame at q_0
        root = chain[1]
        # T_base_to_root = compute_transform_to_joint(root, kinematic_chains, joint_info, q=q_0, base_link=base_link)
        root_base = joint_info[root][0]

        # Distance from north pole to root
        dist_north_root = np.linalg.norm(root_base - north_pole_base)

        # Compute current finger length as sum of segment distances (norm of xyz offsets)
        finger_length = 0.0
        for j in chain[2:]:
            xyz = np.array(joint_info[j][0])
            finger_length += np.linalg.norm(xyz)

        # Current total path length
        current_total = dist_north_root + finger_length

        print("Length for finger", chain)
        print("Length:", current_total)
        print("Length:", finger_length)
        print("Length:", dist_north_root)
        # Delta to adjust
        delta = target_length - current_total

        # Adjust the fingertip along its local x-axis, expressed in parent frame
        fingertip = chain[-1]
        xyz_tip = np.array(joint_info[fingertip][0])
        quat_tip = joint_info[fingertip][1]
        R = tf.quaternion_matrix(quat_tip)[:3, :3]  # Rotation matrix from tip to parent frame
        local_x = np.array([1.0, 0.0, 0.0])
        dir_parent = R @ local_x  # Local x-axis in parent frame
        norm_dir = np.linalg.norm(dir_parent)
        if norm_dir == 0:
            raise ValueError(f"Fingertip {fingertip} has degenerate x-axis direction.")
        unit_dir = dir_parent / norm_dir
        new_xyz = xyz_tip + delta * unit_dir
        new_length = np.linalg.norm(new_xyz)
        # Update joint_info for fingertip
        joint_info[fingertip] = (
            new_xyz.tolist(),  # Updated xyz
            joint_info[fingertip][1],  # quat unchanged
            joint_info[fingertip][2],  # type unchanged
            joint_info[fingertip][3],  # axis unchanged
            joint_info[fingertip][4],  # lower unchanged
            joint_info[fingertip][5],  # upper unchanged
            new_length  # Updated length
        )

    # Optionally return the target length for verification
    return target_length

def compute_finger_chain_order_old(joint_info, kinematic_chains, theta_ranges, q_0, left=False):
    """
    Compute finger chain order based on spherical median and assign indices.

    Parameters:
    - joint_info (dict): Joint information dictionary with sphere_frame and chain root positions.
    - kinematic_chains (list of lists): List of kinematic chains for fingers.

    Returns:
    - indices (list): Ordered indices for each finger chain (e.g., [2] for 1 chain, [1, 2, 3] for 3 chains).
    """
    # Extract sphere frame transformation
    sphere_xyz = joint_info["sphere_frame"][0]
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3, 3] = sphere_xyz
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)

    # Filter finger chains (exclude palm_normal and sphere_frame)
    finger_chains = [
        chain for chain in kinematic_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]
    num_chains = len(finger_chains)

    # Compute theta values for each finger chain
    theta_values = []
    for chain in finger_chains:
        root_xyz = joint_info[chain[1]][0]  # Root position of the chain
        root_xyz_h = np.append(root_xyz, 1)  # Homogeneous coordinate
        root_in_sphere = T_sphere_to_base @ root_xyz_h
        theta = np.arctan2(root_in_sphere[1], root_in_sphere[0])
        theta_values.append(theta)
    
    # Compute circular mean (theta_alignment)
    sum_sin = sum(np.sin(theta) for theta in theta_values)
    sum_cos = sum(np.cos(theta) for theta in theta_values)
    theta_alignment = np.arctan2(sum_sin, sum_cos)

    # Find spherical median (chain closest to theta_alignment)
    min_diff = float('inf')
    median_idx = 0
    for idx, theta in enumerate(theta_values):
        diff = abs((theta - theta_alignment + np.pi) % (2 * np.pi) - np.pi)
        if diff < min_diff:
            min_diff = diff
            median_idx = idx
    
    #! Get mean of 2 if five fingers

    alignment_idx = median_idx

    # Update sphere frame with rotation around z-axis using spherical median's theta
    theta_alignment = theta_values[median_idx]
    
    c = 0
    for theta_range in theta_ranges: # Allgin to directly to any joint without azimuthal joints if any
        if theta_range is None:
            root_xyz = joint_info[finger_chains[c][1]][0]  # Root position of the chain
            root_xyz_h = np.append(root_xyz, 1)  # Homogeneous coordinate
            root_in_sphere = T_sphere_to_base @ root_xyz_h
            theta_alignment = np.arctan2(root_in_sphere[1], root_in_sphere[0])
            alignment_idx = c
        c+=1



    T_new = np.eye(4)
    R = tf.rotation_matrix(theta_alignment, [0, 0, 1])
    T_new = T_base_to_sphere @ R
    xyz_sphere = T_new[:3, 3]
    quat_sphere = tf.quaternion_from_matrix(T_new)
    # quat_sphere = sphere_quat
    joint_info["sphere_frame"] = (
        xyz_sphere,
        quat_sphere,
        "fixed",
        None,
        None,
        None,
        joint_info["sphere_frame"][6]  # Preserve radius
    )

    # Compute shifted theta values relative to theta_alignment
    theta_shifted = (np.array(theta_values) - theta_alignment + np.pi) % (2 * np.pi) - np.pi

    # Assign indices dynamically based on number of chains and their positions
    indices = [None] * num_chains
    print("Theta shifted", theta_shifted)
    sorted_indices = np.argsort(theta_shifted)
    # print("sorted_indices", sorted_indices)
    
    # Assign index 2 to the spherical median (midpoint)
    indices[alignment_idx] = 2
    alignment_pos = sorted_indices.tolist().index(alignment_idx)

    # Assign indices to chains on the negative side (if any)
    j = 0 
    for i in sorted_indices[:alignment_pos].tolist():
        indices[i] = j  # 0, 1, etc., for negative side
        j+= 1

    j = 0
    # Assign indices to chains on the positive side (if any)
    for i in np.flip(sorted_indices[(alignment_pos+1):]).tolist():
        indices[i] = 4 - j # 3, 4, etc., for positive side
        j +=1

    return theta_alignment,  indices

def compute_7_point_sample(finger_print_dict, joint_info, kin_chains, q_0_dict):
    """
    Compute a 7-point sample from finger_print_dict for each finger chain, based on geometric criteria.
    
    Parameters:
    - finger_print_dict (dict): Maps joint names to {'points': array, 'theta': array, 'phi': array}.
    - joint_info (dict): Maps joint names to (xyz, quat, type, axis, lower, upper, length).
    - kin_chains (list): List of kinematic chains (lists of joint names).
    - q_0_dict (dict): Joint angles at initial configuration.
    
    Returns:
    - dict: Maps joint names to lists of selected points in their local frames.
    """
    # Initialize output dictionary
    fp_7_point_sample = {joint: [] for joint in joint_info.keys()}
    fp_7_theta_phi_sample = {joint: [] for joint in joint_info.keys()}
    fp_7_quats = {joint: [] for joint in joint_info.keys()}
    
    # Filter finger chains
    finger_chains = [
        chain for chain in kin_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]
    fp_vector_indices = {chain[1]: {} for chain in finger_chains}

    sphere_xyz = joint_info["sphere_frame"][0]
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3,3] = sphere_xyz
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)

    # Add 9-point sample for base_link
    base_link = kin_chains[0][0]  # base_link is the first joint in all chains
    if base_link in finger_print_dict and finger_print_dict[base_link].get('points', np.array([])).size > 0:
        points = finger_print_dict[base_link]['points']  # Array of [x, y, z] points
        phi = finger_print_dict[base_link]['phi']        # Array of phi values
        theta = finger_print_dict[base_link]['theta']        # Array of theta values
        
        # Sample 1: Point with phi closest to 0
        idx_phi0 = np.argmin(np.abs(phi))
        fp_7_point_sample[base_link].append(points[idx_phi0]) # Palm center point
        fp_7_theta_phi_sample[base_link].append((theta[idx_phi0], phi[idx_phi0])) # Palm center point
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        points_in_sphere = (T_sphere_to_base @ points_h.T).T[:,:3]

        # Compute statistics for x, y, z coordinates
        min_x, min_y, min_z = np.min(points_in_sphere, axis=0)
        max_x, max_y, max_z = np.max(points_in_sphere, axis=0)
        avg_x, avg_y, avg_z = np.mean(points_in_sphere, axis=0)
        
        # Define the 8 target coordinates
        targets = [
            (min_x, min_y, avg_z),
            (min_x, max_y, avg_z),
            (max_x, min_y, avg_z),
            (max_x, max_y, avg_z),
            (min_x, avg_y, avg_z),
            (max_x, avg_y, avg_z),
            (avg_x, min_y, avg_z),
            (avg_x, max_y, avg_z)
        ]
        
        # Samples 2-9: Find closest point to each target
        for target in targets:
            target_array = np.array(target)
            distances = np.linalg.norm(points_in_sphere - target_array, axis=1)
            idx_closest = np.argmin(distances)
            fp_7_point_sample[base_link].append(points[idx_closest])
            fp_7_theta_phi_sample[base_link].append((theta[idx_closest], phi[idx_closest]))
    
    for finger_chain in finger_chains:
        fingertip = finger_chain[-1]    # Last joint in chain
        parent = finger_chain[-2]       # Parent of fingertip
        root = finger_chain[1]          # First joint after base_link
        base_link = finger_chain[0]     # Base link
        
        # **Sample 1: Fingertip origin in parent's frame**
        xyz_fingertip = joint_info[fingertip][0]  # Translation from parent to fingertip
        # fp_7_point_sample[parent].append(xyz_fingertip) Save as last
        # print("first sample:", xyz_fingertip)
        
        # **Sample 2: Root origin in base_link's frame**
        root_xyz = np.array(joint_info[root][0])  # Translation from base_link to root 
        root_quat = np.array(joint_info[root][1])  # Translation from base_link to root 
        T_base_to_root = tf.quaternion_matrix(root_quat)
        T_base_to_root[:3,3] = root_xyz
        q_val = q_0_dict.get(root, 0.0)  # Default to 0 if not in q_dict
        R_z = tf.rotation_matrix(q_val, [0, 0, 1])
        T_base_to_root = T_base_to_root @ R_z # Apply root transform

        sphere_xyz = joint_info["sphere_frame"][0]
        sphere_quat = joint_info["sphere_frame"][1]
        T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
        T_base_to_sphere[:3,3] = sphere_xyz
        T_sphere_to_base = np.linalg.inv(T_base_to_sphere)
        root_xyz_in_sphere = (T_sphere_to_base @ np.append(root_xyz, 1).T).T[:3]
        root_xyz_dir = root_xyz_in_sphere / np.linalg.norm(root_xyz_in_sphere)
        root_phi = np.arccos(root_xyz_dir[2])
        root_theta = np.arctan2(root_xyz_dir[1], root_xyz_dir[0])

        fp_7_point_sample[root].append([0,0,0]) # Add root to root
        fp_7_theta_phi_sample[root].append((root_theta, root_phi))
        
        T_root_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q_0_dict)
        fp_7_quats[root].append(tf.quaternion_from_matrix(T_root_to_fingertip))
        # fp_7_point_sample[base_link].append(xyz_root) 

        # Collect all fingerprint points in fingertip's frame
        relevant_joints = [
            j for j in finger_chain[1:]
            if j in finger_print_dict and finger_print_dict[j].get('points', np.array([])).size > 0
        ]
        if not relevant_joints:
            print(f"No fingerprint points for chain {finger_chain}, skipping.")
            continue
            
        all_points_fingertip = []
        all_points_original = []
        all_joints = []
        all_thetas = []
        all_phis = []
        for j in relevant_joints:
            T_j_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, j, joint_info, q_0_dict)
            T_ft_to_j = np.linalg.inv(T_j_to_fingertip)
            phis_j = finger_print_dict[j]['phi']
            thetas_j = finger_print_dict[j]['theta']
            points_j = finger_print_dict[j]['points']
            points_j_h = np.hstack((points_j, np.ones((points_j.shape[0], 1))))
            points_fingertip_h = (T_ft_to_j @ points_j_h.T).T
            points_fingertip = points_fingertip_h[:, :3]
            all_points_fingertip.append(points_fingertip)
            all_points_original.append(points_j)
            all_joints.extend([j] * points_j.shape[0])
            all_thetas.append(thetas_j)
            all_phis.append(phis_j)
        
        all_points_fingertip = np.vstack(all_points_fingertip)
        all_points_original = np.vstack(all_points_original)
        all_thetas = np.concatenate(all_thetas)
        all_phis = np.concatenate(all_phis)
        
        # **Sample 3: Closest point to midpoint**
        # Root position in fingertip frame
        T_root_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q_0_dict)
        T_ft_to_root = np.linalg.inv(T_root_to_fingertip)
        p_root_in_fingertip = T_ft_to_root[:3, 3]
        # p_root_in_fingertip[2] = 0 
        # Fingertip position is origin
        p_fingertip = np.array([0, 0, 0])
        # Midpoint
        midpoint = (p_root_in_fingertip + p_fingertip) / 2
        # Find closest point
        distances = np.linalg.norm(all_points_fingertip - midpoint, axis=1)
        idx_closest = np.argmin(distances)
        closest_joint = all_joints[idx_closest]
        closest_point_original = all_points_original[idx_closest]

        fp_7_point_sample[closest_joint].append(closest_point_original)
        fp_7_theta_phi_sample[closest_joint].append((all_thetas[idx_closest], all_phis[idx_closest]))
        T_cj_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, closest_joint, joint_info, q_0_dict)
        fp_7_quats[closest_joint].append(tf.quaternion_from_matrix(T_cj_to_fingertip)) # xyzw

        try: 
            fp_vector_indices[root][closest_joint].append(len(fp_7_point_sample[closest_joint])-1)
        except:
            fp_vector_indices[root][closest_joint]=[len(fp_7_point_sample[closest_joint])-1]
        # print("second sample:", closest_point_original)
        
        # Split points along the root-to-fingertip direction
        v = p_root_in_fingertip  # Direction vector
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            print(f"Root and fingertip coincide in chain {finger_chain}, skipping split.")
            continue
        t = (all_points_fingertip @ v) / (v @ v)  # Projection parameter
        group1_idx = t >= 0.5  # Between midpoint and root
        group2_idx = t < 0.5   # Between fingertip and midpoint

        # print(f"{finger_chain[-1] } chian group1 filter", np.sum(group1_idx))
        # print(f"{finger_chain[-1] } chian group2 filter", np.sum(group2_idx))
        
        # **Samples 4 and 5: Midpoint to root group**
        if np.any(group1_idx):
            group1_points = all_points_fingertip[group1_idx]
            avg_z1 = np.mean(group1_points[:, 2])
            max_y1 = np.max(group1_points[:, 1])
            min_y1 = np.min(group1_points[:, 1])
            avg_x1 = (p_root_in_fingertip[0] + midpoint[0]) / 2
            target_max = np.array([avg_x1, max_y1, avg_z1])
            target_min = np.array([avg_x1, min_y1, avg_z1])
            distances_max = np.linalg.norm(group1_points - target_max, axis=1)
            distances_min = np.linalg.norm(group1_points - target_min, axis=1)
            idx_max = np.argmin(distances_max)
            idx_min = np.argmin(distances_min)
            # print("G1 IDX MAX MIN", idx_max, idx_min)
            original_idx_max = np.where(group1_idx)[0][idx_max]
            original_idx_min = np.where(group1_idx)[0][idx_min]
            fp_7_point_sample[all_joints[original_idx_max]].append(all_points_original[original_idx_max])
            fp_7_point_sample[all_joints[original_idx_min]].append(all_points_original[original_idx_min])
            fp_7_theta_phi_sample[all_joints[original_idx_max]].append((all_thetas[original_idx_max], all_phis[original_idx_max]))
            fp_7_theta_phi_sample[all_joints[original_idx_min]].append((all_thetas[original_idx_min], all_phis[original_idx_min]))
            T_max_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, all_joints[original_idx_max], joint_info, q_0_dict)
            fp_7_quats[all_joints[original_idx_max]].append(tf.quaternion_from_matrix(T_max_to_fingertip)) # xyzw
            T_min_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, all_joints[original_idx_min], joint_info, q_0_dict)
            fp_7_quats[all_joints[original_idx_min]].append(tf.quaternion_from_matrix(T_min_to_fingertip)) # xyzw

            # print("sample 3 4:", all_points_original[original_idx_max], all_points_original[original_idx_min])
        
        # **Samples 6 and 7: Fingertip to midpoint group**
        if np.any(group2_idx):
            group2_points = all_points_fingertip[group2_idx]
            avg_z2 = np.mean(group2_points[:, 2])
            max_y2 = np.max(group2_points[:, 1])
            min_y2 = np.min(group2_points[:, 1])
            avg_x2 = (p_fingertip[0] + midpoint[0]) / 2  # p_fingertip[0] = 0
            target_max = np.array([avg_x2, max_y2, avg_z2])
            target_min = np.array([avg_x2, min_y2, avg_z2])
            distances_max = np.linalg.norm(group2_points - target_max, axis=1)
            distances_min = np.linalg.norm(group2_points - target_min, axis=1)
            idx_max = np.argmin(distances_max)
            idx_min = np.argmin(distances_min)
            # print("G2 IDX MAX MIN", idx_max, idx_min)
            original_idx_max = np.where(group2_idx)[0][idx_max]
            original_idx_min = np.where(group2_idx)[0][idx_min]
            fp_7_point_sample[all_joints[original_idx_max]].append(all_points_original[original_idx_max])
            fp_7_point_sample[all_joints[original_idx_min]].append(all_points_original[original_idx_min])
            fp_7_theta_phi_sample[all_joints[original_idx_max]].append((all_thetas[original_idx_max], all_phis[original_idx_max]))
            fp_7_theta_phi_sample[all_joints[original_idx_min]].append((all_thetas[original_idx_min], all_phis[original_idx_min]))
            # print("sample 3 4:", all_points_original[original_idx_max], all_points_original[original_idx_min])
            T_max_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, all_joints[original_idx_max], joint_info, q_0_dict)
            fp_7_quats[all_joints[original_idx_max]].append(tf.quaternion_from_matrix(T_max_to_fingertip)) # xyzw
            T_min_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, all_joints[original_idx_min], joint_info, q_0_dict)
            fp_7_quats[all_joints[original_idx_min]].append(tf.quaternion_from_matrix(T_min_to_fingertip)) # xyzw
        
        # Add ft to be always at the end (easy lookup)
        fp_7_point_sample[parent].append(xyz_fingertip)
        try:
            fp_vector_indices[root][parent].append(len(fp_7_point_sample[parent])-1)
        except:
            fp_vector_indices[root][parent]=[len(fp_7_point_sample[parent])-1]
        p_fingertip_h = np.array([0, 0, 0, 1])
        T_sphere_to_fingertip = T_sphere_to_base @ T_base_to_root @ T_root_to_fingertip
        ft_xyz_in_sphere = (T_sphere_to_fingertip @ p_fingertip_h.T).T[:3]
        ft_xyz_dir = ft_xyz_in_sphere / np.linalg.norm(ft_xyz_in_sphere)
        ft_phi = np.arccos(ft_xyz_dir[2])
        ft_theta = np.arctan2(ft_xyz_dir[1], ft_xyz_dir[0])
        fp_7_theta_phi_sample[parent].append((ft_theta, ft_phi))
        T_parent_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, parent, joint_info, q_0_dict)
        fp_7_quats[parent].append(tf.quaternion_from_matrix(T_parent_to_fingertip)) # xyzw
        
    
    print(f"Total of {len(fp_7_point_sample[base_link])} points for base_link")
    for finger_chain in finger_chains:
        total_points = sum(len(fp_7_point_sample[j]) for j in finger_chain[1:] if j in fp_7_point_sample.keys())
        print(f"Total of {total_points} points for finger chain with tip '{finger_chain[-1]}'")
    
    return fp_7_point_sample, fp_vector_indices, fp_7_theta_phi_sample, fp_7_quats

def compute_3_point_sample_skeleton(finger_print_dict, joint_info, kin_chains, q_opened, q_sphere):
    """
    Compute a 3 point sample for each finger chain using a three-leve.
    
    Parameters:
    - finger_print_dict (dict): Maps joint names to {'points': array, 'theta': array, 'phi': array}.
    - joint_info (dict): Maps joint names to (xyz, quat, type, axis, lower, upper, length).
    - kin_chains (list): List of kinematic chains (lists of joint names).
    - q_0_dict (dict): Joint angles at initial configuration.
    
    Returns:
    - dict: Maps joint names to lists of selected points in their local frames.
    """
    # Initialize output dictionary
    fp_point_sample = {joint: [] for joint in finger_print_dict.keys()}
    fp_theta_phi_sample = {joint: [] for joint in finger_print_dict.keys()}
    
    # Filter finger chains (exclude palm_normal and sphere_frame)
    finger_chains = [
        chain for chain in kin_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]

    sphere_xyz = joint_info["sphere_frame"][0]
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3,3] = sphere_xyz
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)
    
    # Add 9-point sample for base_link
    base_link = kin_chains[0][0]  # base_link is the first joint in all chains
    if base_link in finger_print_dict and finger_print_dict[base_link].get('points', np.array([])).size > 0:
        points = finger_print_dict[base_link]['points']  # Array of [x, y, z] points
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        points = (T_sphere_to_base @ points_h.T).T [:, :3]
        
        radius = joint_info["sphere_frame"][6]
        n_pole = np.array([0.0, 0.0 , radius, 1.0])
        n_pole = (T_base_to_sphere @ n_pole.T).T[:3]
        fp_point_sample[base_link].append(n_pole) # Palm center point
        fp_theta_phi_sample[base_link].append((0.0, 0.0)) # Palm center point
        
        # Compute statistics for x, y, z coordinates
        min_x, min_y, min_z = np.min(points, axis=0)
        max_x, max_y, max_z = np.max(points, axis=0)
        avg_x, avg_y, avg_z = np.mean(points, axis=0)
        
        # Define the 8 target coordinates
        targets = np.array([
            [min_x, min_y, radius, 1.0],
            [min_x, max_y, radius, 1.0],
            [max_x, min_y, radius, 1.0],
            [max_x, max_y, radius, 1.0],
            [min_x, avg_y, radius, 1.0],
            [max_x, avg_y, radius, 1.0],
            [avg_x, min_y, radius, 1.0],
            [avg_x, max_y, radius, 1.0]
        ])
        
        targets_phi = np.arccos(targets[:, 2])
        targets_theta = np.arctan2(targets[:,1], targets[:,0])
        targets_in_base = (T_base_to_sphere @ targets.T).T [:, :3]
        for point in targets_in_base.tolist():
            fp_point_sample[base_link].append(point)
        for i in range(len(targets_phi)):
            fp_theta_phi_sample[base_link].append((targets_theta[i], targets_phi[i]))

    for finger_chain in finger_chains:
        fingertip = finger_chain[-1]  # Last joint in the chain
        parent = finger_chain[-2]     # Parent of fingertip
        root = finger_chain[1]        # First joint after base_link
        base_link = finger_chain[0]   # Base link (not sampled in 10 points)
        
        # Sample 1: Fingertip origin in parent's frame
        xyz_fingertip = joint_info[fingertip][0]
        # fp_10_point_sample[parent].append(xyz_fingertip)
        
        # Sample 2: Root origin in base_link's frame (not part of the 10 points)
        root_xyz = np.array(joint_info[root][0])  # Translation from base_link to root 
        root_quat = np.array(joint_info[root][1])  # Translation from base_link to root 
        T_base_to_root = tf.quaternion_matrix(root_quat)
        T_base_to_root[:3,3] = root_xyz
        q_val = q_opened.get(root, 0.0)  # Default to 0 if not in q_dict
        R_z = tf.rotation_matrix(q_val, [0, 0, 1])
        T_base_to_root = T_base_to_root @ R_z # Apply root transform

        root_xyz_in_sphere = (T_sphere_to_base @ np.append(root_xyz, 1).T).T[:3]
        root_xyz_dir = root_xyz_in_sphere / np.linalg.norm(root_xyz_in_sphere)
        root_phi = np.arccos(root_xyz_dir[2])
        root_theta = np.arctan2(root_xyz_dir[1], root_xyz_dir[0])

        fp_point_sample[root].append([0,0,0]) # Add root to root
        fp_theta_phi_sample[root].append((root_theta, root_phi))

        # fp_10_point_sample[base_link].append(xyz_root) 
        
        # Collect fingerprint points from relevant joints (exclude base_link)
        relevant_joints = [
            j for j in finger_chain[1:]
            if j in finger_print_dict and finger_print_dict[j].get('points', np.array([])).size > 0
        ]
        if not relevant_joints:
            print(f"No fingerprint points for chain {finger_chain}, skipping.")
            continue
        
        all_points_fingertip = []
        all_points_original = []
        all_joints = []
        all_thetas = []
        all_phis = []
        # Collect all fingerprint points
        for j in relevant_joints:
            # Compute transformation from joint j to fingertip
            T_j_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, j, joint_info, q_opened)
            T_ft_to_j = np.linalg.inv(T_j_to_fingertip)
            phis_j = finger_print_dict[j]['phi']
            points_j = finger_print_dict[j]['points']
            thetas_j = finger_print_dict[j]['theta']
            points_j_h = np.hstack((points_j, np.ones((points_j.shape[0], 1))))
            points_fingertip_h = (T_ft_to_j @ points_j_h.T).T
            points_fingertip = points_fingertip_h[:, :3]
            all_points_fingertip.append(points_fingertip)
            all_points_original.append(points_j)
            all_joints.extend([j] * points_j.shape[0])
            all_thetas.append(thetas_j)
            all_phis.append(phis_j)
        
        
        all_points_fingertip = np.vstack(all_points_fingertip)
        all_points_original = np.vstack(all_points_original)
        all_thetas = np.concatenate(all_thetas)
        all_phis = np.concatenate(all_phis)
        
        # Compute positions along the fingertip-to-root axis
        T_root_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q_opened)
        T_ft_to_root = np.linalg.inv(T_root_to_fingertip)
        p_root_in_fingertip = T_ft_to_root[:3, 3]
        p_fingertip = np.array([0, 0, 0])  # Origin in fingertip frame
        v = p_root_in_fingertip - p_fingertip
        pos_1_2 = p_fingertip + (1/2) * v  # 1/2 position
        
        # Sample 3: Closest to 1/3 position
        distances = np.linalg.norm(all_points_fingertip - pos_1_2, axis=1)
        idx_closest = np.argmin(distances)
        closest_joint = all_joints[idx_closest]
        T_closest_joint_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, closest_joint, joint_info, q_opened)
        pos_1_2_h = np.append(pos_1_2, 1.0)
        p12 = (T_closest_joint_to_fingertip @ pos_1_2_h.T).T

        # Get Transform from sphere to cj
        T_base_to_cj =  compute_transform_to_joint(closest_joint, finger_chains, joint_info, q_sphere, base_link)
        T_sphere_to_cj = T_sphere_to_base @ T_base_to_cj
        p12_in_sphere = (T_sphere_to_cj @ p12.T).T[:3]
        p12_dir = p12_in_sphere / np.linalg.norm(p12_in_sphere)
        p12_phi = np.arccos(p12_dir[2])
        p12_theta = np.arctan2(p12_dir[1], p12_dir[0])
        fp_point_sample[closest_joint].append(p12[:3])
        fp_theta_phi_sample[closest_joint].append((p12_theta, p12_phi))
        

        # Add ft to be always at the end 
        fp_point_sample[parent].append(xyz_fingertip)
        T_base_to_parent =  compute_transform_to_joint(parent, finger_chains, joint_info, q_sphere, base_link)
        p_fingertip_h = np.append(xyz_fingertip, 1.0)
        T_sphere_to_parent = T_sphere_to_base @ T_base_to_parent 

        ft_xyz_in_sphere = (T_sphere_to_parent @ p_fingertip_h.T).T[:3]
        ft_xyz_dir = ft_xyz_in_sphere / np.linalg.norm(ft_xyz_in_sphere)
        ft_phi = np.arccos(ft_xyz_dir[2])
        ft_theta = np.arctan2(ft_xyz_dir[1], ft_xyz_dir[0])
        fp_theta_phi_sample[parent].append((ft_theta, ft_phi))

        
    print(f"Total of {len(fp_point_sample[base_link])} points for base_link")
    for finger_chain in finger_chains:
        total_points = sum(len(fp_point_sample[j]) for j in finger_chain[1:] if j in fp_point_sample.keys())
        print(f"Total of {total_points} points for finger chain with tip '{finger_chain[-1]}'")
    
    
    return fp_point_sample, fp_theta_phi_sample

def compute_transform_from_joint_to_fingertip(chain, joint, joint_info, q_dict=None):
    """
    Compute the 4x4 transformation matrix from a specific joint to the fingertip in a kinematic chain.

    Parameters:
    - chain (list): A kinematic chain as a list of joint names from base to fingertip 
                    (e.g., ["base_link", "joint1", "joint2", "fingertip"]).
    - specific_joint (str): The joint from which to compute the transformation to the fingertip.
    - joint_info (dict): Maps joint names to (xyz, quat, joint_type, axis, lower, upper, length).
                         Each tuple contains position, orientation, type, etc., relative to the parent.
    - q_dict (dict, optional): Joint angles for revolute joints; if None, assumes zero configuration.

    Returns:
    - np.ndarray: 4x4 transformation matrix from specific_joint to the fingertip.

    Raises:
    - ValueError: If specific_joint is not in the chain or is the fingertip itself.
    """
    # Validate inputs
    if joint not in chain:
        raise ValueError(f"Specific joint '{joint}' not found in the chain")
    
    idx = chain.index(joint)
    if idx == len(chain) - 1:
        return np.eye(4)
    
    # Extract the subchain from the child of specific_joint to fingertip
    subchain = chain[idx + 1:] 

    # Initialize the transformation matrix as identity
    T_joint_to_fingertip = np.eye(4)
    
    # Accumulate transformations from the child of specific_joint to fingertip
    for child in subchain:
        # Retrieve joint properties
        xyz, quat, joint_type, _, _, _, _ = joint_info[child]  # Unpack position, orientation, type
        
        # Build the static transformation from parent to child
        T = tf.quaternion_matrix(quat)  # 4x4 matrix from quaternion
        T[:3, 3] = xyz                  # Set translation
        
        # Apply rotation for revolute joints if angle is provided
        if q_dict is not None and joint_type == "revolute":
            q_val = q_dict.get(child, 0.0)  # Default to 0 if not in q_dict
            R_z = tf.rotation_matrix(q_val, [0, 0, 1])  # Rotation around z-axis (common convention)
            T = T @ R_z  # Compose the static transform with the joint rotation
        
        # Accumulate the transformation
        T_joint_to_fingertip = T_joint_to_fingertip @ T
    
    return T_joint_to_fingertip

def compute_10_point_sample(finger_print_dict, joint_info, kin_chains, q_0_dict):
    """
    Compute a 10-point sample for each finger chain using a three-level split at 1/3 and 2/3 positions.
    
    Parameters:
    - finger_print_dict (dict): Maps joint names to {'points': array, 'theta': array, 'phi': array}.
    - joint_info (dict): Maps joint names to (xyz, quat, type, axis, lower, upper, length).
    - kin_chains (list): List of kinematic chains (lists of joint names).
    - q_0_dict (dict): Joint angles at initial configuration.
    
    Returns:
    - dict: Maps joint names to lists of selected points in their local frames.
    """
    # Initialize output dictionary
    fp_10_point_sample = {joint: [] for joint in finger_print_dict.keys()}
    fp_10_theta_phi_sample = {joint: [] for joint in finger_print_dict.keys()}
    
    # Filter finger chains (exclude palm_normal and sphere_frame)
    finger_chains = [
        chain for chain in kin_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]
    fp_vector_indices = {chain[1]: {} for chain in finger_chains}

    sphere_xyz = joint_info["sphere_frame"][0]
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3,3] = sphere_xyz
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)

    # Add 9-point sample for base_link
    base_link = kin_chains[0][0]  # base_link is the first joint in all chains
    if base_link in finger_print_dict and finger_print_dict[base_link].get('points', np.array([])).size > 0:
        points = finger_print_dict[base_link]['points']  # Array of [x, y, z] points
        phi = finger_print_dict[base_link]['phi']        # Array of phi values
        theta = finger_print_dict[base_link]['theta']        # Array of theta values
        
        # Sample 1: Point with phi closest to 0
        idx_phi0 = np.argmin(np.abs(phi))
        fp_10_point_sample[base_link].append(points[idx_phi0]) # Palm center point
        fp_10_theta_phi_sample[base_link].append((theta[idx_phi0], phi[idx_phi0])) # Palm center point
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        points_in_sphere = (T_sphere_to_base @ points_h.T).T[:,:3]

        # Compute statistics for x, y, z coordinates
        min_x, min_y, min_z = np.min(points_in_sphere, axis=0)
        max_x, max_y, max_z = np.max(points_in_sphere, axis=0)
        avg_x, avg_y, avg_z = np.mean(points_in_sphere, axis=0)
        
        # Define the 8 target coordinates
        targets = [
            (min_x, min_y, avg_z),
            (min_x, max_y, avg_z),
            (max_x, min_y, avg_z),
            (max_x, max_y, avg_z),
            (min_x, avg_y, avg_z),
            (max_x, avg_y, avg_z),
            (avg_x, min_y, avg_z),
            (avg_x, max_y, avg_z)
        ]
        
        # Samples 2-9: Find closest point to each target
        for target in targets:
            target_array = np.array(target)
            distances = np.linalg.norm(points_in_sphere - target_array, axis=1)
            idx_closest = np.argmin(distances)
            fp_10_point_sample[base_link].append(points[idx_closest])
            fp_10_theta_phi_sample[base_link].append((theta[idx_closest], phi[idx_closest]))

    for finger_chain in finger_chains:
        fingertip = finger_chain[-1]  # Last joint in the chain
        parent = finger_chain[-2]     # Parent of fingertip
        root = finger_chain[1]        # First joint after base_link
        base_link = finger_chain[0]   # Base link (not sampled in 10 points)
        
        # Sample 1: Fingertip origin in parent's frame
        xyz_fingertip = joint_info[fingertip][0]
        # fp_10_point_sample[parent].append(xyz_fingertip)
        
        # Sample 2: Root origin in base_link's frame (not part of the 10 points)
        root_xyz = np.array(joint_info[root][0])  # Translation from base_link to root 
        root_quat = np.array(joint_info[root][1])  # Translation from base_link to root 
        T_base_to_root = tf.quaternion_matrix(root_quat)
        T_base_to_root[:3,3] = root_xyz
        q_val = q_0_dict.get(root, 0.0)  # Default to 0 if not in q_dict
        R_z = tf.rotation_matrix(q_val, [0, 0, 1])
        T_base_to_root = T_base_to_root @ R_z # Apply root transform

        sphere_xyz = joint_info["sphere_frame"][0]
        sphere_quat = joint_info["sphere_frame"][1]
        T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
        T_base_to_sphere[:3,3] = sphere_xyz
        T_sphere_to_base = np.linalg.inv(T_base_to_sphere)
        root_xyz_in_sphere = (T_sphere_to_base @ np.append(root_xyz, 1).T).T[:3]
        root_xyz_dir = root_xyz_in_sphere / np.linalg.norm(root_xyz_in_sphere)
        root_phi = np.arccos(root_xyz_dir[2])
        root_theta = np.arctan2(root_xyz_dir[1], root_xyz_dir[0])

        fp_10_point_sample[root].append([0,0,0]) # Add root to root
        fp_10_theta_phi_sample[root].append((root_theta, root_phi))

        # fp_10_point_sample[base_link].append(xyz_root) 
        
        # Collect fingerprint points from relevant joints (exclude base_link)
        relevant_joints = [
            j for j in finger_chain[1:]
            if j in finger_print_dict and finger_print_dict[j].get('points', np.array([])).size > 0
        ]
        if not relevant_joints:
            print(f"No fingerprint points for chain {finger_chain}, skipping.")
            continue
        
        all_points_fingertip = []
        all_points_original = []
        all_joints = []
        all_thetas = []
        all_phis = []
        for j in relevant_joints:
            # Compute transformation from joint j to fingertip
            T_j_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, j, joint_info, q_0_dict)
            T_ft_to_j = np.linalg.inv(T_j_to_fingertip)
            phis_j = finger_print_dict[j]['phi']
            points_j = finger_print_dict[j]['points']
            thetas_j = finger_print_dict[j]['theta']
            points_j_h = np.hstack((points_j, np.ones((points_j.shape[0], 1))))
            points_fingertip_h = (T_ft_to_j @ points_j_h.T).T
            points_fingertip = points_fingertip_h[:, :3]
            all_points_fingertip.append(points_fingertip)
            all_points_original.append(points_j)
            all_joints.extend([j] * points_j.shape[0])
            all_thetas.append(thetas_j)
            all_phis.append(phis_j)
        
        
        all_points_fingertip = np.vstack(all_points_fingertip)
        all_points_original = np.vstack(all_points_original)
        all_thetas = np.concatenate(all_thetas)
        all_phis = np.concatenate(all_phis)
        
        # Compute positions along the fingertip-to-root axis
        T_root_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q_0_dict)
        T_ft_to_root = np.linalg.inv(T_root_to_fingertip)
        p_root_in_fingertip = T_ft_to_root[:3, 3]
        p_fingertip = np.array([0, 0, 0])  # Origin in fingertip frame
        v = p_root_in_fingertip - p_fingertip
        pos_1_3 = p_fingertip + (1/3) * v  # 1/3 position
        pos_2_3 = p_fingertip + (2/3) * v  # 2/3 position
        
        # Sample 3: Closest to 1/3 position
        distances = np.linalg.norm(all_points_fingertip - pos_1_3, axis=1)
        idx_closest = np.argmin(distances)
        closest_joint = all_joints[idx_closest]
        fp_10_point_sample[all_joints[idx_closest]].append(all_points_original[idx_closest])
        fp_10_theta_phi_sample[closest_joint].append((all_thetas[idx_closest], all_phis[idx_closest]))
        try: 
            fp_vector_indices[root][closest_joint].append(len(fp_10_point_sample[closest_joint])-1)
        except:
            fp_vector_indices[root][closest_joint]=[len(fp_10_point_sample[closest_joint])-1]
        
        # Sample 4: Closest to 2/3 position
        distances = np.linalg.norm(all_points_fingertip - pos_2_3, axis=1)
        idx_closest = np.argmin(distances)
        closest_joint = all_joints[idx_closest]
        fp_10_point_sample[all_joints[idx_closest]].append(all_points_original[idx_closest])
        fp_10_theta_phi_sample[closest_joint].append((all_thetas[idx_closest], all_phis[idx_closest]))
        try: 
            fp_vector_indices[root][closest_joint].append(len(fp_10_point_sample[closest_joint])-1)
        except:
            fp_vector_indices[root][closest_joint]=[len(fp_10_point_sample[closest_joint])-1]
        
        # Compute projection parameter t along the axis
        t = ((all_points_fingertip - p_fingertip) @ v) / (v @ v)
        
        # Define groups based on t
        group1_idx = t < 1/3              # Fingertip to 1/3
        group2_idx = (t >= 1/3) & (t < 2/3)  # 1/3 to 2/3
        group3_idx = t >= 2/3             # 2/3 to root
        
        # Function to sample two points in a group
        def sample_group(group_idx, group_name):
            if np.any(group_idx):
                group_points = all_points_fingertip[group_idx]
                avg_x = np.mean(group_points[:, 0])
                avg_z = np.mean(group_points[:, 2])
                max_y = np.max(group_points[:, 1])
                min_y = np.min(group_points[:, 1])
                target_max = np.array([avg_x, max_y, avg_z])
                target_min = np.array([avg_x, min_y, avg_z])
                distances_max = np.linalg.norm(group_points - target_max, axis=1)
                distances_min = np.linalg.norm(group_points - target_min, axis=1)
                idx_max = np.argmin(distances_max)
                idx_min = np.argmin(distances_min)
                original_idx_max = np.where(group_idx)[0][idx_max]
                original_idx_min = np.where(group_idx)[0][idx_min]
                fp_10_point_sample[all_joints[original_idx_max]].append(all_points_original[original_idx_max])
                fp_10_point_sample[all_joints[original_idx_min]].append(all_points_original[original_idx_min])
                fp_10_theta_phi_sample[all_joints[original_idx_max]].append((all_thetas[original_idx_max], all_phis[original_idx_max]))
                fp_10_theta_phi_sample[all_joints[original_idx_min]].append((all_thetas[original_idx_min], all_phis[original_idx_min]))
        
        # Sample points for each group (6 points total: 2 per group)
        sample_group(group1_idx, "group1")
        sample_group(group2_idx, "group2")
        sample_group(group3_idx, "group3")

        # Add ft to be always at the end 
        fp_10_point_sample[parent].append(xyz_fingertip)
        try:
            fp_vector_indices[root][parent].append(len(fp_10_point_sample[parent])-1)
        except:
            fp_vector_indices[root][parent]=[len(fp_10_point_sample[parent])-1]
        p_fingertip_h = np.array([0, 0, 0, 1])
        T_sphere_to_fingertip = T_sphere_to_base @ T_base_to_root @ T_root_to_fingertip
        ft_xyz_in_sphere = (T_sphere_to_fingertip @ p_fingertip_h.T).T[:3]
        ft_xyz_dir = ft_xyz_in_sphere / np.linalg.norm(ft_xyz_in_sphere)
        ft_phi = np.arccos(ft_xyz_dir[2])
        ft_theta = np.arctan2(ft_xyz_dir[1], ft_xyz_dir[0])
        fp_10_theta_phi_sample[parent].append((ft_theta, ft_phi))

        
    print(f"Total of {len(fp_10_point_sample[base_link])} points for base_link")
    for finger_chain in finger_chains:
        total_points = sum(len(fp_10_point_sample[j]) for j in finger_chain[1:] if j in fp_10_point_sample.keys())
        print(f"Total of {total_points} points for finger chain with tip '{finger_chain[-1]}'")
    
    
    return fp_10_point_sample, fp_vector_indices, fp_10_theta_phi_sample

def compute_4_point_sample_skeleton(finger_print_dict, joint_info, kin_chains, q_opened, q_sphere):
    """
    Compute a 10-point sample for each finger chain using a three-level split at 1/3 and 2/3 positions.
    
    Parameters:
    - finger_print_dict (dict): Maps joint names to {'points': array, 'theta': array, 'phi': array}.
    - joint_info (dict): Maps joint names to (xyz, quat, type, axis, lower, upper, length).
    - kin_chains (list): List of kinematic chains (lists of joint names).
    - q_0_dict (dict): Joint angles at initial configuration.
    
    Returns:
    - dict: Maps joint names to lists of selected points in their local frames.
    """
    # Initialize output dictionary
    fp_point_sample = {joint: [] for joint in joint_info.keys()}
    fp_theta_phi_sample = {joint: [] for joint in joint_info.keys()}
    fp_quats = {joint: [] for joint in joint_info.keys()}
    
    # Filter finger chains (exclude palm_normal and sphere_frame)
    finger_chains = [
        chain for chain in kin_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]

    sphere_xyz = joint_info["sphere_frame"][0]
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3,3] = sphere_xyz
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)
    
    # Add 9-point sample for base_link
    base_link = kin_chains[0][0]  # base_link is the first joint in all chains
    if base_link in finger_print_dict and finger_print_dict[base_link].get('points', np.array([])).size > 0:
        points = finger_print_dict[base_link]['points']  # Array of [x, y, z] points
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        points = (T_sphere_to_base @ points_h.T).T [:, :3]
        
        radius = joint_info["sphere_frame"][6]
        n_pole = np.array([0.0, 0.0 , radius, 1.0])
        n_pole = (T_base_to_sphere @ n_pole.T).T[:3]
        fp_point_sample[base_link].append(n_pole) # Palm center point
        fp_theta_phi_sample[base_link].append((0.0, 0.0)) # Palm center point
        
        # Compute statistics for x, y, z coordinates
        min_x, min_y, min_z = np.min(points, axis=0)
        max_x, max_y, max_z = np.max(points, axis=0)
        avg_x, avg_y, avg_z = np.mean(points, axis=0)
        
        # Define the 8 target coordinates
        targets = np.array([
            [min_x, min_y, radius, 1.0],
            [min_x, max_y, radius, 1.0],
            [max_x, min_y, radius, 1.0],
            [max_x, max_y, radius, 1.0],
            [min_x, avg_y, radius, 1.0],
            [max_x, avg_y, radius, 1.0],
            [avg_x, min_y, radius, 1.0],
            [avg_x, max_y, radius, 1.0]
        ])
        
        targets_phi = np.arccos(targets[:, 2])
        targets_theta = np.arctan2(targets[:,1], targets[:,0])
        targets_in_base = (T_base_to_sphere @ targets.T).T [:, :3]
        for point in targets_in_base.tolist():
            fp_point_sample[base_link].append(point)
        for i in range(len(targets_phi)):
            fp_theta_phi_sample[base_link].append((targets_theta[i], targets_phi[i]))

    for finger_chain in finger_chains:
        fingertip = finger_chain[-1]  # Last joint in the chain
        parent = finger_chain[-2]     # Parent of fingertip
        root = finger_chain[1]        # First joint after base_link
        base_link = finger_chain[0]   # Base link (not sampled in 10 points)
        
        # Sample 1: Fingertip origin in parent's frame
        xyz_fingertip = joint_info[fingertip][0]
        # fp_point_sample[parent].append(xyz_fingertip)
        
        # Sample 2: Root origin in base_link's frame (not part of the 10 points)
        root_xyz = np.array(joint_info[root][0])  # Translation from base_link to root 
        root_quat = np.array(joint_info[root][1])  # Translation from base_link to root 
        T_base_to_root = tf.quaternion_matrix(root_quat)
        T_base_to_root[:3,3] = root_xyz
        q_val = q_opened.get(root, 0.0)  # Default to 0 if not in q_dict
        R_z = tf.rotation_matrix(q_val, [0, 0, 1])
        T_base_to_root = T_base_to_root @ R_z # Apply root transform

        root_xyz_in_sphere = (T_sphere_to_base @ np.append(root_xyz, 1).T).T[:3]
        root_xyz_dir = root_xyz_in_sphere / np.linalg.norm(root_xyz_in_sphere)
        root_phi = np.arccos(root_xyz_dir[2])
        root_theta = np.arctan2(root_xyz_dir[1], root_xyz_dir[0])

        fp_point_sample[root].append([0,0,0]) # Add root to root
        fp_theta_phi_sample[root].append((root_theta, root_phi))

        T_root_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q_opened)
        fp_quats[root].append(tf.quaternion_from_matrix(T_root_to_fingertip))
        # fp_point_sample[base_link].append(xyz_root) 
        
        # Collect fingerprint points from relevant joints (exclude base_link)
        relevant_joints = [
            j for j in finger_chain[1:]
            if (j in finger_print_dict.keys()) & (finger_print_dict.get(j, dict()).get('points', np.array([])).size > 0)
        ]
        if not relevant_joints:
            print(f"No fingerprint points for chain {finger_chain}, skipping.")
            continue
        
        all_points_fingertip = []
        all_points_original = []
        all_joints = []
        all_thetas = []
        all_phis = []
        # Collect all fingerprint points
        for j in relevant_joints:
            # Compute transformation from joint j to fingertip
            T_j_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, j, joint_info, q_opened)
            T_ft_to_j = np.linalg.inv(T_j_to_fingertip)
            phis_j = finger_print_dict[j]['phi']
            points_j = finger_print_dict[j]['points']
            thetas_j = finger_print_dict[j]['theta']
            points_j_h = np.hstack((points_j, np.ones((points_j.shape[0], 1))))
            points_fingertip_h = (T_ft_to_j @ points_j_h.T).T
            points_fingertip = points_fingertip_h[:, :3]
            all_points_fingertip.append(points_fingertip)
            all_points_original.append(points_j)
            all_joints.extend([j] * points_j.shape[0])
            all_thetas.append(thetas_j)
            all_phis.append(phis_j)
        
        
        all_points_fingertip = np.vstack(all_points_fingertip)
        all_points_original = np.vstack(all_points_original)
        all_thetas = np.concatenate(all_thetas)
        all_phis = np.concatenate(all_phis)
        
        # Compute positions along the fingertip-to-root axis
        T_root_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q_opened)
        T_ft_to_root = np.linalg.inv(T_root_to_fingertip)
        p_root_in_fingertip = T_ft_to_root[:3, 3]
        p_fingertip = np.array([0, 0, 0])  # Origin in fingertip frame
        v = p_root_in_fingertip - p_fingertip
        pos_1_3 = p_fingertip + (1/3) * v  # 1/3 position
        pos_2_3 = p_fingertip + (2/3) * v  # 2/3 position

        # Sample 3: Closest to 2/3 position
        distances = np.linalg.norm(all_points_fingertip - pos_2_3, axis=1)
        idx_closest = np.argmin(distances)
        closest_joint = all_joints[idx_closest]
        T_closest_joint_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, closest_joint, joint_info, q_opened)
        pos_2_3_h = np.append(pos_2_3, 1.0)
        p23 = (T_closest_joint_to_fingertip @ pos_2_3_h.T).T
        T_base_to_cj =  compute_transform_to_joint(closest_joint, finger_chains, joint_info, q_sphere, base_link)
        T_sphere_to_cj = T_sphere_to_base @ T_base_to_cj
        p23_in_sphere = (T_sphere_to_cj @ p23.T).T[:3]
        p23_dir = p23_in_sphere / np.linalg.norm(p23_in_sphere)
        p23_phi = np.arccos(p23_dir[2])
        p23_theta = np.arctan2(p23_dir[1], p23_dir[0])
        fp_point_sample[closest_joint].append(p23[:3])
        fp_theta_phi_sample[closest_joint].append((p23_theta, p23_phi))
        fp_quats[closest_joint].append(tf.quaternion_from_matrix(T_closest_joint_to_fingertip))

        # Sample 4: Closest to 1/3 position
        distances = np.linalg.norm(all_points_fingertip - pos_1_3, axis=1)
        idx_closest = np.argmin(distances)
        closest_joint = all_joints[idx_closest]
        T_closest_joint_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, closest_joint, joint_info, q_opened)
        pos_1_3_h = np.append(pos_1_3, 1.0)
        p13 = (T_closest_joint_to_fingertip @ pos_1_3_h.T).T
        T_base_to_cj =  compute_transform_to_joint(closest_joint, finger_chains, joint_info, q_sphere, base_link)
        T_sphere_to_cj = T_sphere_to_base @ T_base_to_cj
        p13_in_sphere = (T_sphere_to_cj @ p13.T).T[:3]
        p13_dir = p13_in_sphere / np.linalg.norm(p13_in_sphere)
        p13_phi = np.arccos(p13_dir[2])
        p13_theta = np.arctan2(p13_dir[1], p13_dir[0])
        fp_point_sample[closest_joint].append(p13[:3])
        fp_theta_phi_sample[closest_joint].append((p13_theta, p13_phi))
        fp_quats[closest_joint].append(tf.quaternion_from_matrix(T_closest_joint_to_fingertip))
        
        # Add ft to be always at the end 
        fp_point_sample[parent].append(xyz_fingertip)
        T_base_to_parent =  compute_transform_to_joint(parent, finger_chains, joint_info, q_sphere, base_link)
        p_fingertip_h = np.append(xyz_fingertip, 1.0)
        T_sphere_to_parent = T_sphere_to_base @ T_base_to_parent 

        ft_xyz_in_sphere = (T_sphere_to_parent @ p_fingertip_h.T).T[:3]
        ft_xyz_dir = ft_xyz_in_sphere / np.linalg.norm(ft_xyz_in_sphere)
        ft_phi = np.arccos(ft_xyz_dir[2])
        ft_theta = np.arctan2(ft_xyz_dir[1], ft_xyz_dir[0])
        fp_theta_phi_sample[parent].append((ft_theta, ft_phi))
        T_parent_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, parent, joint_info, q_opened)
        fp_quats[parent].append(tf.quaternion_from_matrix(T_parent_to_fingertip)) # xyzw
        
        
    print(f"Total of {len(fp_point_sample[base_link])} points for base_link")
    for finger_chain in finger_chains:
        total_points = sum(len(fp_point_sample[j]) for j in finger_chain[1:] if j in fp_point_sample.keys())
        print(f"Total of {total_points} points for finger chain with tip '{finger_chain[-1]}'")
    
    
    return fp_point_sample, fp_theta_phi_sample, fp_quats

def compute_4_point_sample_skeleton_with_lookups(finger_print_dict, joint_info, joint_type_info, kin_chains, q_opened, q_sphere):
    """
    Compute a 10-point sample for each finger chain using a three-level split at 1/3 and 2/3 positions.
    
    Parameters:
    - finger_print_dict (dict): Maps joint names to {'points': array, 'theta': array, 'phi': array}.
    - joint_info (dict): Maps joint names to (xyz, quat, type, axis, lower, upper, length).
    - joint_type_info (dict): Maps joint names to type info (dictionaries)
    - kin_chains (list): List of kinematic chains (lists of joint names).
    - q_0_dict (dict): Joint angles at initial configuration.
    
    Returns:
    - dict: Maps joint names to lists of selected points in their local frames.
    """
    # Initialize output dictionary
    fp_point_sample = {joint: [] for joint in joint_info.keys()}
    fp_theta_phi_sample = {joint: [] for joint in joint_info.keys()}
    fp_quats = {joint: [] for joint in joint_info.keys()}
    main_joints = [j for j in joint_type_info.keys() if joint_type_info[j]["type"]=="A"]

    # Filter finger chains (exclude palm_normal and sphere_frame)
    finger_chains = [
        chain for chain in kin_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]

    sphere_xyz = joint_info["sphere_frame"][0]
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3,3] = sphere_xyz
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)
    
    # Add 9-point sample for base_link
    base_link = kin_chains[0][0]  # base_link is the first joint in all chains
    if base_link in finger_print_dict and finger_print_dict[base_link].get('points', np.array([])).size > 0:
        points = finger_print_dict[base_link]['points']  # Array of [x, y, z] points
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        points = (T_sphere_to_base @ points_h.T).T [:, :3]
        
        radius = joint_info["sphere_frame"][6]
        n_pole = np.array([0.0, 0.0 , radius, 1.0])
        n_pole = (T_base_to_sphere @ n_pole.T).T[:3]
        fp_point_sample[base_link].append(n_pole) # Palm center point
        fp_theta_phi_sample[base_link].append((0.0, 0.0)) # Palm center point
        
        # Compute statistics for x, y, z coordinates
        min_x, min_y, min_z = np.min(points, axis=0)
        max_x, max_y, max_z = np.max(points, axis=0)
        avg_x, avg_y, avg_z = np.mean(points, axis=0)
        
        # Define the 8 target coordinates
        targets = np.array([
            [min_x, min_y, radius, 1.0],
            [min_x, max_y, radius, 1.0],
            [max_x, min_y, radius, 1.0],
            [max_x, max_y, radius, 1.0],
            [min_x, avg_y, radius, 1.0],
            [max_x, avg_y, radius, 1.0],
            [avg_x, min_y, radius, 1.0],
            [avg_x, max_y, radius, 1.0]
        ])
        
        targets_phi = np.arccos(targets[:, 2])
        targets_theta = np.arctan2(targets[:,1], targets[:,0])
        targets_in_base = (T_base_to_sphere @ targets.T).T [:, :3]
        for point in targets_in_base.tolist():
            fp_point_sample[base_link].append(point)
        for i in range(len(targets_phi)):
            fp_theta_phi_sample[base_link].append((targets_theta[i], targets_phi[i]))

    for finger_chain in finger_chains:
        fingertip = finger_chain[-1]  # Last joint in the chain
        parent = finger_chain[-2]     # Parent of fingertip
        root = finger_chain[1]        # First joint after base_link
        base_link = finger_chain[0]   # Base link (not sampled in 10 points)
        
        main_joint = next((j for j in finger_chain if j in main_joints), None)
        
        if main_joint is None:
            # No main joint in chain: execute original logic with correction for point transformation
            # Sample 1: Fingertip origin in parent's frame
            xyz_fingertip = joint_info[fingertip][0]
            
            # Sample 2: Root origin in base_link's frame (not part of the 10 points)
            root_xyz = np.array(joint_info[root][0])  # Translation from base_link to root 
            root_quat = np.array(joint_info[root][1])  # Translation from base_link to root 
            T_base_to_root = tf.quaternion_matrix(root_quat)
            T_base_to_root[:3,3] = root_xyz
            q_val = q_opened.get(root, 0.0)  # Default to 0 if not in q_dict
            R_z = tf.rotation_matrix(q_val, [0, 0, 1])
            T_base_to_root = T_base_to_root @ R_z # Apply root transform

            root_xyz_in_sphere = (T_sphere_to_base @ np.append(root_xyz, 1).T).T[:3]
            root_xyz_dir = root_xyz_in_sphere / np.linalg.norm(root_xyz_in_sphere)
            root_phi = np.arccos(root_xyz_dir[2])
            root_theta = np.arctan2(root_xyz_dir[1], root_xyz_dir[0])

            fp_point_sample[root].append([0,0,0]) # Add root to root
            fp_theta_phi_sample[root].append((root_theta, root_phi))

            T_root_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q_opened)
            fp_quats[root].append(tf.quaternion_from_matrix(T_root_to_fingertip))
            
            # Collect fingerprint points from relevant joints (exclude base_link)
            relevant_joints = [
                j for j in finger_chain[1:]
                if (j in finger_print_dict.keys()) & (finger_print_dict.get(j, dict()).get('points', np.array([])).size > 0)
            ]
            if not relevant_joints:
                print(f"No fingerprint points for chain {finger_chain}, skipping.")
                continue
            
            all_points_fingertip = []
            all_points_original = []
            all_joints = []
            all_thetas = []
            all_phis = []
            # Collect all fingerprint points
            for j in relevant_joints:
                # Compute transformation from joint j to fingertip
                T_j_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, j, joint_info, q_opened)
                T_ft_to_j = np.linalg.inv(T_j_to_fingertip)
                phis_j = finger_print_dict[j]['phi']
                points_j = finger_print_dict[j]['points']
                thetas_j = finger_print_dict[j]['theta']
                points_j_h = np.hstack((points_j, np.ones((points_j.shape[0], 1))))
                points_fingertip_h = (T_ft_to_j @ points_j_h.T).T
                points_fingertip = points_fingertip_h[:, :3]
                all_points_fingertip.append(points_fingertip)
                all_points_original.append(points_j)
                all_joints.extend([j] * points_j.shape[0])
                all_thetas.append(thetas_j)
                all_phis.append(phis_j)
            
            
            all_points_fingertip = np.vstack(all_points_fingertip)
            all_points_original = np.vstack(all_points_original)
            all_thetas = np.concatenate(all_thetas)
            all_phis = np.concatenate(all_phis)
            
            # Compute positions along the fingertip-to-root axis
            T_root_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q_opened)
            T_ft_to_root = np.linalg.inv(T_root_to_fingertip)
            p_root_in_fingertip = T_ft_to_root[:3, 3]
            p_fingertip = np.array([0, 0, 0])  # Origin in fingertip frame
            v = p_root_in_fingertip - p_fingertip
            pos_1_3 = p_fingertip + (1/3) * v  # 1/3 position
            pos_2_3 = p_fingertip + (2/3) * v  # 2/3 position

            # Sample 3: Closest to 2/3 position
            distances = np.linalg.norm(all_points_fingertip - pos_2_3, axis=1)
            idx_closest = np.argmin(distances)
            closest_joint = all_joints[idx_closest]
            T_closest_joint_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, closest_joint, joint_info, q_opened)
            pos_2_3_h = np.append(pos_2_3, 1.0)
            p23 = (T_closest_joint_to_fingertip @ pos_2_3_h.T).T
            T_base_to_cj =  compute_transform_to_joint(closest_joint, finger_chains, joint_info, q_sphere, base_link)
            T_sphere_to_cj = T_sphere_to_base @ T_base_to_cj
            p23_in_sphere = (T_sphere_to_cj @ p23.T).T[:3]
            p23_dir = p23_in_sphere / np.linalg.norm(p23_in_sphere)
            p23_phi = np.arccos(p23_dir[2])
            p23_theta = np.arctan2(p23_dir[1], p23_dir[0])
            fp_point_sample[closest_joint].append(p23[:3])
            fp_theta_phi_sample[closest_joint].append((p23_theta, p23_phi))
            fp_quats[closest_joint].append(tf.quaternion_from_matrix(T_closest_joint_to_fingertip))

            # Sample 4: Closest to 1/3 position
            distances = np.linalg.norm(all_points_fingertip - pos_1_3, axis=1)
            idx_closest = np.argmin(distances)
            closest_joint = all_joints[idx_closest]
            T_closest_joint_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, closest_joint, joint_info, q_opened)
            pos_1_3_h = np.append(pos_1_3, 1.0)
            p13 = (T_closest_joint_to_fingertip @ pos_1_3_h.T).T
            T_base_to_cj =  compute_transform_to_joint(closest_joint, finger_chains, joint_info, q_sphere, base_link)
            T_sphere_to_cj = T_sphere_to_base @ T_base_to_cj
            p13_in_sphere = (T_sphere_to_cj @ p13.T).T[:3]
            p13_dir = p13_in_sphere / np.linalg.norm(p13_in_sphere)
            p13_phi = np.arccos(p13_dir[2])
            p13_theta = np.arctan2(p13_dir[1], p13_dir[0])
            fp_point_sample[closest_joint].append(p13[:3])
            fp_theta_phi_sample[closest_joint].append((p13_theta, p13_phi))
            fp_quats[closest_joint].append(tf.quaternion_from_matrix(T_closest_joint_to_fingertip))
            
            # Add ft to be always at the end 
            fp_point_sample[parent].append(xyz_fingertip)
            T_base_to_parent =  compute_transform_to_joint(parent, finger_chains, joint_info, q_sphere, base_link)
            p_fingertip_h = np.append(xyz_fingertip, 1.0)
            T_sphere_to_parent = T_sphere_to_base @ T_base_to_parent 

            ft_xyz_in_sphere = (T_sphere_to_parent @ p_fingertip_h.T).T[:3]
            ft_xyz_dir = ft_xyz_in_sphere / np.linalg.norm(ft_xyz_in_sphere)
            ft_phi = np.arccos(ft_xyz_dir[2])
            ft_theta = np.arctan2(ft_xyz_dir[1], ft_xyz_dir[0])
            fp_theta_phi_sample[parent].append((ft_theta, ft_phi))
            T_parent_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, parent, joint_info, q_opened)
            fp_quats[parent].append(tf.quaternion_from_matrix(T_parent_to_fingertip)) # xyzw
        else:
            # Chain has main joint: compute closest joints and theta_phi with initial q_opened, then vary points and quats over offset_list
            offset_list = joint_type_info[main_joint]["offset_list"]
            
            # Collect fingerprint points and find closest joints with initial q_opened
            relevant_joints = [
                j for j in finger_chain[1:]
                if (j in finger_print_dict.keys()) & (finger_print_dict.get(j, dict()).get('points', np.array([])).size > 0)
            ]
            if not relevant_joints:
                print(f"No fingerprint points for chain {finger_chain}, skipping.")
                continue
            
            all_points_fingertip = []
            all_joints = []
            all_thetas = []
            all_phis = []
            all_points_original = []  # Included for completeness, though not used here
            for j in relevant_joints:
                T_j_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, j, joint_info, q_opened)
                T_ft_to_j = np.linalg.inv(T_j_to_fingertip)
                phis_j = finger_print_dict[j]['phi']
                points_j = finger_print_dict[j]['points']
                thetas_j = finger_print_dict[j]['theta']
                points_j_h = np.hstack((points_j, np.ones((points_j.shape[0], 1))))
                points_fingertip_h = (T_ft_to_j @ points_j_h.T).T
                points_fingertip = points_fingertip_h[:, :3]
                all_points_fingertip.append(points_fingertip)
                all_points_original.append(points_j)
                all_joints.extend([j] * points_j.shape[0])
                all_thetas.append(thetas_j)
                all_phis.append(phis_j)
            
            all_points_fingertip = np.vstack(all_points_fingertip)
            all_thetas = np.concatenate(all_thetas)
            all_phis = np.concatenate(all_phis)
            
            T_root_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q_opened)
            T_ft_to_root = np.linalg.inv(T_root_to_fingertip)
            p_root_in_fingertip = T_ft_to_root[:3, 3]
            v = p_root_in_fingertip - np.array([0, 0, 0])
            pos_1_3 = np.array([0, 0, 0]) + (1/3) * v
            pos_2_3 = np.array([0, 0, 0]) + (2/3) * v
            
            # Find closest for 2/3
            distances = np.linalg.norm(all_points_fingertip - pos_2_3, axis=1)
            idx_closest = np.argmin(distances)
            closest_joint23 = all_joints[idx_closest]
            
            # Find closest for 1/3
            distances = np.linalg.norm(all_points_fingertip - pos_1_3, axis=1)
            idx_closest = np.argmin(distances)
            closest_joint13 = all_joints[idx_closest]
            
            # Compute theta_phi for root (fixed)
            root_sample_idx = len(fp_point_sample[root]) - 1
            root_xyz = np.array(joint_info[root][0])
            root_quat = np.array(joint_info[root][1])
            T_base_to_root = tf.quaternion_matrix(root_quat)
            T_base_to_root[:3,3] = root_xyz
            q_val = q_opened.get(root, 0.0)
            R_z = tf.rotation_matrix(q_val, [0, 0, 1])
            T_base_to_root = T_base_to_root @ R_z
            root_xyz_in_sphere = (T_sphere_to_base @ np.append(root_xyz, 1).T).T[:3]
            root_xyz_dir = root_xyz_in_sphere / np.linalg.norm(root_xyz_in_sphere)
            root_phi = np.arccos(root_xyz_dir[2])
            root_theta = np.arctan2(root_xyz_dir[1], root_xyz_dir[0])
            fp_theta_phi_sample[root].append((root_theta, root_phi))
            fp_point_sample[root].append([])  # Initialize list for variations
            fp_quats[root].append([])  # Initialize list for variations
            
            # Compute theta_phi for p23 with initial q_opened
            p23_sample_idx = len(fp_point_sample[closest_joint23]) - 1
            pos_2_3_h = np.append(pos_2_3, 1.0)
            T_closest_to_ft = compute_transform_from_joint_to_fingertip(finger_chain, closest_joint23, joint_info, q_opened)
            p23 = (T_closest_to_ft @ pos_2_3_h.T).T[:3]
            T_base_to_cj = compute_transform_to_joint(closest_joint23, finger_chains, joint_info, q_sphere, base_link)
            T_sphere_to_cj = T_sphere_to_base @ T_base_to_cj
            p23_in_sphere = (T_sphere_to_cj @ np.append(p23, 1).T).T[:3]
            p23_dir = p23_in_sphere / np.linalg.norm(p23_in_sphere)
            p23_phi = np.arccos(p23_dir[2])
            p23_theta = np.arctan2(p23_dir[1], p23_dir[0])
            fp_theta_phi_sample[closest_joint23].append((p23_theta, p23_phi))
            fp_point_sample[closest_joint23].append([])  # Initialize list for variations
            fp_quats[closest_joint23].append([])  # Initialize list for variations
            
            # Compute theta_phi for p13 with initial q_opened
            p13_sample_idx = len(fp_point_sample[closest_joint13]) - 1
            pos_1_3_h = np.append(pos_1_3, 1.0)
            T_closest_to_ft = compute_transform_from_joint_to_fingertip(finger_chain, closest_joint13, joint_info, q_opened)
            p13 = (T_closest_to_ft @ pos_1_3_h.T).T[:3]
            T_base_to_cj = compute_transform_to_joint(closest_joint13, finger_chains, joint_info, q_sphere, base_link)
            T_sphere_to_cj = T_sphere_to_base @ T_base_to_cj
            p13_in_sphere = (T_sphere_to_cj @ np.append(p13, 1).T).T[:3]
            p13_dir = p13_in_sphere / np.linalg.norm(p13_in_sphere)
            p13_phi = np.arccos(p13_dir[2])
            p13_theta = np.arctan2(p13_dir[1], p13_dir[0])
            fp_theta_phi_sample[closest_joint13].append((p13_theta, p13_phi))
            fp_point_sample[closest_joint13].append([])  # Initialize list for variations
            fp_quats[closest_joint13].append([])  # Initialize list for variations
            
            # Compute theta_phi for ft (fixed)
            ft_sample_idx = len(fp_point_sample[parent]) - 1
            xyz_fingertip = joint_info[fingertip][0]
            T_base_to_parent = compute_transform_to_joint(parent, finger_chains, joint_info, q_sphere, base_link)
            p_fingertip_h = np.append(xyz_fingertip, 1.0)
            T_sphere_to_parent = T_sphere_to_base @ T_base_to_parent
            ft_xyz_in_sphere = (T_sphere_to_parent @ p_fingertip_h.T).T[:3]
            ft_xyz_dir = ft_xyz_in_sphere / np.linalg.norm(ft_xyz_in_sphere)
            ft_phi = np.arccos(ft_xyz_dir[2])
            ft_theta = np.arctan2(ft_xyz_dir[1], ft_xyz_dir[0])
            fp_theta_phi_sample[parent].append((ft_theta, ft_phi))
            fp_point_sample[parent].append([])  # Initialize list for variations
            fp_quats[parent].append([])  # Initialize list for variations
            
            # Now vary over offset_list for points and quats
            for q_val in offset_list:
                q_temp = q_opened.copy()
                q_temp[main_joint] = q_val
                
                # Root sample
                fp_point_sample[root][root_sample_idx].append([0,0,0])
                T_root_to_ft = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q_temp)
                fp_quats[root][root_sample_idx].append(tf.quaternion_from_matrix(T_root_to_ft))

                # p23 sample
                T_root_to_ft = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q_temp)
                T_ft_to_root = np.linalg.inv(T_root_to_ft)
                p_root_in_fingertip = T_ft_to_root[:3, 3]
                v = p_root_in_fingertip - np.array([0, 0, 0])
                pos_2_3 = np.array([0,0,0]) + (2/3) * v
                pos_2_3_h = np.append(pos_2_3, 1.0)
                T_closest_to_ft = compute_transform_from_joint_to_fingertip(finger_chain, closest_joint23, joint_info, q_temp)
                p23 = (T_closest_to_ft @ pos_2_3_h.T).T[:3]
                fp_point_sample[closest_joint23][p23_sample_idx].append(p23)
                fp_quats[closest_joint23][p23_sample_idx].append(tf.quaternion_from_matrix(T_closest_to_ft))
                
                # p13 sample
                pos_1_3 = np.array([0,0,0]) + (1/3) * v
                pos_1_3_h = np.append(pos_1_3, 1.0)
                T_closest_to_ft = compute_transform_from_joint_to_fingertip(finger_chain, closest_joint13, joint_info, q_temp)
                p13 = (T_closest_to_ft @ pos_1_3_h.T).T[:3]
                fp_point_sample[closest_joint13][p13_sample_idx].append(p13)
                fp_quats[closest_joint13][p13_sample_idx].append(tf.quaternion_from_matrix(T_closest_to_ft))
                
                # ft sample
                fp_point_sample[parent][ft_sample_idx].append(xyz_fingertip)
                T_parent_to_ft = compute_transform_from_joint_to_fingertip(finger_chain, parent, joint_info, q_temp)
                fp_quats[parent][ft_sample_idx].append(tf.quaternion_from_matrix(T_parent_to_ft))
        
    print(f"Total of {len(fp_point_sample[base_link])} points for base_link")
    for finger_chain in finger_chains:
        total_points = sum(len(sublist) for j in finger_chain[1:] if j in fp_point_sample.keys() for sublist in fp_point_sample[j])
        print(f"Total of {total_points} points for finger chain with tip '{finger_chain[-1]}'")
    
    
    return fp_point_sample, fp_theta_phi_sample, fp_quats

def find_q_A_for_theta_mid(joint, theta_mid, q_0, q_max, fingertip, joint_info, kinematic_chains, T_sphere_to_base, T_palm_to_base, joint_type_info, N=1000, base_link = "base_link"):
    """Find the q_A value that places the fingertip at theta_mid."""
    lower, upper = joint_info[joint][4], joint_info[joint][5]
    q_A_values = np.linspace(lower, upper, N)
    best_q_A = None
    min_diff = float('inf')
    if joint_type_info[joint]["type"] == "C":
        finger_chain = kinematic_chains[next((i for i, sublist in enumerate(kinematic_chains) if joint in sublist), -1)] 
        if joint == finger_chain[1]:
            q = q_max.copy() 
        else:
            q = q_0.copy() 
            idx = finger_chain.index(joint)
            n_j = finger_chain[idx+1]
            q[n_j]= joint_info[n_j][5] # Max range of next joint (assumes positive closes the hand)
    else:
        q = q_0.copy()
    for q_A in q_A_values:
        theta = compute_theta_for_q_A(joint, q_A, q, fingertip, joint_info, joint_type_info, kinematic_chains, T_sphere_to_base, T_palm_to_base, base_link)
        if theta is not None:
            diff = abs(theta - theta_mid)
            if diff < min_diff:
                min_diff = diff
                best_q_A = q_A
    if best_q_A is None:
        raise ValueError(f"Can't find the q_A of joint {joint} for theta_mid {theta_mid}")
    return best_q_A

def build_lookup_dict(joint, theta_mid, q_0, q_max, fingertip, joint_info, kinematic_chains, T_sphere_to_base, T_palm_to_base, joint_type_info, resolution=0.01, N=10000, base_link="base_link"):
    """Build a lookup dictionary for a Type A joint."""
    lower, upper = joint_info[joint][4], joint_info[joint][5]
    q_A_values = np.linspace(lower, upper, N)
    theta_values = []
    valid_q_A = []

    if joint_type_info[joint]["type"] == "C":
        finger_chain = kinematic_chains[next((i for i, sublist in enumerate(kinematic_chains) if joint in sublist), -1)] 
        if joint == finger_chain[1]:
            q = q_max.copy() 
        else:
            q = q_0.copy()
            idx = finger_chain.index(joint)
            n_j = finger_chain[idx+1]
            q[n_j]= joint_info[n_j][5] # Max range of next joint (assumes positive closes the hand)
    else:
        q = q_0
    
    # Probe the joint extensively
    for q_A in q_A_values:
        theta = compute_theta_for_q_A(joint, q_A, q, fingertip, joint_info, joint_type_info, kinematic_chains, T_sphere_to_base, T_palm_to_base, base_link)
        if theta is not None:
            theta_values.append(theta)
            valid_q_A.append(q_A)
    
    if not theta_values:
        raise ValueError(f"No valid theta values for joint {joint}")
    
    theta_values = np.array(theta_values)
    valid_q_A = np.array(valid_q_A)

    # Update anchor and zero_idx with no_offset
    anchor = theta_mid
    
    # Compute theta offsets from theta_mid and wrap to [-pi, pi]
    theta_offsets = theta_values - anchor
    theta_offsets = (theta_offsets + np.pi) % (2 * np.pi) - np.pi
    
    # Determine min and max offsets
    min_offset = np.min(theta_offsets)
    max_offset = np.max(theta_offsets)
    
    # Generate offset steps
    n_offsets = np.arange(min_offset, 0, resolution)[::-1]  # Negative offsets, descending
    p_offsets = np.arange(0, max_offset, resolution)       # Positive offsets, ascending
    
    # print(theta_offsets)
    # Need to sort theta values and valid_q_A for np.search sorted to work
    if(theta_offsets[0]>theta_offsets[-1]):
        theta_offsets = theta_offsets[::-1]
        valid_q_A = valid_q_A [::-1]

    # Interpolate q_A values for negative offsets
    n_offset_list = []
    for offset in n_offsets:
        idx = np.searchsorted(theta_offsets, offset)
        # print("idx", idx, offset, min_offset)
        if idx == 0 or idx == len(theta_offsets):
            continue
        q_A_left = valid_q_A[idx - 1]
        q_A_right = valid_q_A[idx]
        theta_left = theta_offsets[idx - 1]
        theta_right = theta_offsets[idx]
        q_A = q_A_left + (offset - theta_left) * (q_A_right - q_A_left) / (theta_right - theta_left)
        n_offset_list.append(q_A)
    
    # Interpolate q_A values for positive offsets
    p_offset_list = []
    for offset in p_offsets:
        idx = np.searchsorted(theta_offsets, offset)
        if idx == 0 or idx == len(theta_offsets):
            continue
        q_A_left = valid_q_A[idx - 1]
        q_A_right = valid_q_A[idx]
        theta_left = theta_offsets[idx - 1]
        theta_right = theta_offsets[idx]
        q_A = q_A_left + (offset - theta_left) * (q_A_right - q_A_left) / (theta_right - theta_left)
        p_offset_list.append(q_A)
    
    n_offset_list = n_offset_list[::-1] # Reverse the lookup list
    zero_idx = len(n_offset_list)
    n_offset_list.append(q_0[joint])
    offset_list = n_offset_list + p_offset_list

    # Compute total range
    # theta_range = max_offset - min_offset

    return {
        "type": "A",
        "anchor": anchor,
        "resolution": resolution,
        "upper": max_offset,
        "lower": min_offset,
        "offset_list": offset_list,
        "zero_idx": zero_idx,
        "og_type": joint_type_info[joint]["type"]
    }

def allign_joint_x_axis(joint, desired_x, joint_info, chain, vertices_dict, meshes_dict):
    """
    Allign the joint's x-axis to a desired direction, update vertices, transformations, and compute bounding box.

    Parameters:
    - joint (str): Joint name.
    - desired_x (np.ndarray): 2D vector in x-y plane to align x-axis to.
    - joint_info (dict): Joint information dictionary.
    - chain (list): Kinematic chain of joint.
    - vertices_dict (dict): Dictionary of joint vertices.

    Returns:
    - tuple: (box_min, box_max) or (None, None) if no vertices.
    """
    import numpy as np

    # Normalize desired x-axis direction
    if np.linalg.norm(desired_x) < 1e-6:
        raise ValueError(f"Desired x-axis projection is zero for joint {joint}")
    desired_x = desired_x / np.linalg.norm(desired_x)

    # Compute the angle of the desired vector in the x-y plane
    angle = np.arctan2(desired_x[1], desired_x[0])

    # Create rotation matrix around z-axis by the computed angle
    R_align = tf.rotation_matrix(angle, [0, 0, 1])[:3, :3]
    T_align = np.eye(4)
    T_align[:3, :3] = R_align

    # Update Vertices
    if joint in vertices_dict and vertices_dict[joint].size > 0:
        vertices = vertices_dict[joint]
        homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        aligned_vertices = (np.linalg.inv(T_align) @ homogeneous_vertices.T).T[:, :3]
        vertices_dict[joint] = aligned_vertices
    
    # Update meshes
    if joint in meshes_dict:
        new_meshes = []
        for mesh in meshes_dict[joint]:
            new_mesh = mesh.copy()
            new_mesh = new_mesh.apply_transform(np.linalg.inv(T_align))
            new_meshes.append(new_mesh)
        meshes_dict[joint] = new_meshes

    # Update joint transformation: T_new = T_old @ T_align
    xyz = joint_info[joint][0]
    quat = joint_info[joint][1]
    T = tf.quaternion_matrix(quat)
    T[:3, 3] = xyz
    T_joint_new = T @ T_align
    xyz_new = T_joint_new[:3, 3]
    quat_new = tf.quaternion_from_matrix(T_joint_new)
    joint_info[joint] = (
        xyz_new,
        quat_new,
        joint_info[joint][2],
        joint_info[joint][3],
        joint_info[joint][4],
        joint_info[joint][5],
        joint_info[joint][6]
    )

    # Update child joint transformation
    idx = chain.index(joint)
    if idx + 1 < len(chain):
        child = chain[idx + 1]
        xyz = joint_info[child][0]
        quat = joint_info[child][1]
        T = tf.quaternion_matrix(quat)
        T[:3, 3] = xyz
        T_base_to_child_new = np.linalg.inv(T_align) @ T
        xyz_child_new = T_base_to_child_new[:3, 3]
        quat_child_new = tf.quaternion_from_matrix(T_base_to_child_new)
        joint_info[child] = (
            xyz_child_new,
            quat_child_new,
            joint_info[child][2],
            joint_info[child][3],
            joint_info[child][4],
            joint_info[child][5],
            joint_info[child][6]
        )

    return angle

def process_bounding_boxes_complete_finger(finger_chains, vertices_dict, joint_info, joint_type_info, q_0_dict):
    """Process fingertips as Type FT."""

    for chain in finger_chains:
        # Process box_min, box_max of all joints
        update_info = dict()
        for joint in chain[1:-1]:
            remaining_joints = chain[chain.index(joint):-1] # Don't include fingertip
            T_joint_to_ft = compute_transform_from_joint_to_fingertip(chain, joint, joint_info, q_0_dict)

            ft_normal = T_joint_to_ft[:3, 2]
            p_ft_joint = (T_joint_to_ft)[:3, 3]

            T_parent_to_child = np.eye(4)
            T_q = np.eye(4)
            vertices = []

            for rj in remaining_joints:
                if rj in vertices_dict and vertices_dict[rj].size > 0:
                    # Get vertices
                    new_vertices = vertices_dict[rj]
                    homogeneous_vertices = np.hstack((new_vertices, np.ones((new_vertices.shape[0], 1))))
                    transformed_vertices = (T_parent_to_child @ homogeneous_vertices.T).T[:, :3]
                    vertices.append(transformed_vertices)
                    
                # Use open palm joint transform
                cos_q = np.cos(q_0_dict[rj])
                sin_q = np.sin(q_0_dict[rj])
                R_z = np.array([
                    [cos_q, -sin_q, 0],
                    [sin_q, cos_q, 0],
                    [0, 0, 1]
                ])  # Shape: (n_spheres, 3, 3)
                T_q[:3,:3] = R_z

                # Perform transform to next joint
                if rj != remaining_joints[-1]:
                    nj = remaining_joints[remaining_joints.index(rj) + 1]
                    xyz = joint_info[nj][0]
                    quat = joint_info[nj][1]
                    T_nj = tf.quaternion_matrix(quat)
                    T_nj[:3, 3] = xyz
                    T_parent_to_child = T_parent_to_child @ T_nj
            
            vertices = np.vstack(vertices) if len(vertices)>0 else np.array([[0,0,0],[0,0,0]])

            # Save y axis as z- axis 
            box_min = np.min(vertices, axis=0)
            box_max = np.max(vertices, axis=0)
            update_info["ft"] = p_ft_joint.tolist() # Change to update
            update_info["ft_normal"] = ft_normal.tolist() 
            update_info["box_min"] =  box_min if box_min is not None else None 
            update_info["box_max"] =  box_max if box_max is not None else None
            joint_type_info[joint].update(update_info)
            

        # Add fingertip information to joint_type_info
        ft_update_info = dict()
        fingertip = chain[-1]
        parent_map = {chain[i]: chain[i - 1] for i in range(1, len(chain))} 
        fingertip_parent = parent_map[fingertip]

        xyz = joint_info[fingertip][0]
        quat = joint_info[fingertip][1]
        T_parent_to_fingertip = tf.quaternion_matrix(quat)
        T_parent_to_fingertip[:3, 3] = xyz
        T_fingertip_to_parent = np.linalg.inv(T_parent_to_fingertip)
        
        box_min, box_max = None, None
        if fingertip_parent in vertices_dict and vertices_dict[fingertip_parent].size > 0:
            vertices = vertices_dict[fingertip_parent]
            homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
            transformed_vertices = (T_fingertip_to_parent @ homogeneous_vertices.T).T[:, :3]
            
            # Save y axis as z- axis 
            box_min = np.min(transformed_vertices, axis=0)
            box_max = np.max(transformed_vertices, axis=0)

        ft_update_info = {
            "type": "FT",
            "box_min": box_min if box_min is not None else None,
            "box_max": box_max if box_max is not None else None
        }
        joint_type_info[fingertip].update(ft_update_info)
    return joint_type_info

