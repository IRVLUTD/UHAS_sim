#!/usr/bin/env python
import numpy as np
import json
import os 
import trimesh
from urdf_parser_py.urdf import URDF
import urdf_parser_py
import tf_ros as tf
import matplotlib.pyplot as plt
from utils_archive import *
import pyvista as pv
import h5py
import colorsys

import numpy.ma as ma

import collections
try:
    collections.Iterable
except AttributeError:
    import collections.abc
    collections.Iterable = collections.abc.Iterable

def load_link_meshes(robot_dir, robot):
    """
    Load meshes from URDF links into trimesh objects.
    
    Parameters:
    - robot_dir (str): Directory containing the URDF file (used to resolve relative paths).
    - robot (URDF): Parsed URDF object from urdf_parser_py.
    
    Returns:
    - dict: Mapping of link names to their visual and collision meshes as trimesh.Trimesh objects.
    """
    # Base directory for resolving relative paths
    base_dir = os.path.dirname(os.path.abspath(robot_dir))
    
    # Dictionary to store meshes for each link
    link_meshes = {}
    
    # Iterate over all links in the URDF
    for link in robot.links:
        link_meshes[link.name] = {"visual": [], "collision": []}
        
        # Handle visual geometries
        if link.visuals:
            for visual in link.visuals:
                if isinstance(visual.geometry, urdf_parser_py.urdf.Mesh):
                    mesh_path = visual.geometry.filename
                    # Get scale from URDF, default to [1.0, 1.0, 1.0] if not specified
                    scale = visual.geometry.scale
                    if scale is None:
                        scale = [1.0, 1.0, 1.0]
                    else:
                        # Pad scale to three elements if necessary
                        if len(scale) == 1:
                            scale = [scale[0]] * 3
                        elif len(scale) == 2:
                            scale = [scale[0], scale[1], 1.0]
                        elif len(scale) != 3:
                            raise ValueError(f"Invalid scale length: {len(scale)} for link {link.name}")
                    # Compute origin transformation
                    origin = visual.origin
                    if origin is not None:
                        translation = np.array(origin.xyz)
                        rotation = tf.euler_matrix(*origin.rpy, axes='sxyz')
                        origin_transform = np.eye(4)
                        origin_transform[:3, :3] = rotation[:3, :3]
                        origin_transform[:3, 3] = translation
                    else:
                        origin_transform = np.eye(4)
                    
                    # Resolve mesh path
                    if mesh_path.startswith("package://"):
                        mesh_path = os.path.join(os.path.dirname(base_dir), mesh_path.replace("package://", ""))
                    else:
                        mesh_path = os.path.join(base_dir, mesh_path)

                    # Load mesh with trimesh
                    try:
                        mesh = trimesh.load(mesh_path)
                        if isinstance(mesh, trimesh.Scene):
                            # If loaded as a Scene, extract the first Trimesh object
                            mesh = mesh.geometry[list(mesh.geometry.keys())[0]]
                        # Store mesh and scale as a tuple
                        link_meshes[link.name]["visual"].append((mesh, scale, origin_transform))
                    except Exception as e:
                        print(f"Failed to load visual mesh {mesh_path} for link {link.name}: {e}")
                        
        # Handle collision geometries
        if link.collisions:
            for collision in link.collisions:
                if isinstance(collision.geometry, urdf_parser_py.urdf.Mesh):
                    mesh_path = collision.geometry.filename
                    # Get scale from URDF, default to [1.0, 1.0, 1.0] if not specified
                    scale = collision.geometry.scale
                    if scale is None:
                        scale = [1.0, 1.0, 1.0]
                    else:
                        # Pad scale to three elements if necessary
                        if len(scale) == 1:
                            scale = [scale[0]] * 3
                        elif len(scale) == 2:
                            scale = [scale[0], scale[1], 1.0]
                        elif len(scale) != 3:
                            raise ValueError(f"Invalid scale length: {len(scale)} for link {link.name}")
                    
                    # Compute origin transformation
                    origin = collision.origin
                    if origin is not None:
                        translation = np.array(origin.xyz)
                        rotation = tf.euler_matrix(*origin.rpy, axes='sxyz')
                        origin_transform = np.eye(4)
                        origin_transform[:3, :3] = rotation[:3, :3]
                        origin_transform[:3, 3] = translation
                    else:
                        origin_transform = np.eye(4)
                    
                    # Resolve mesh path
                    if mesh_path.startswith("package://"):
                        mesh_path = os.path.join(os.path.dirname(base_dir), mesh_path.replace("package://", ""))
                    else:
                        mesh_path = os.path.join(base_dir, mesh_path)

                    # Load mesh with trimesh
                    try:
                        mesh = trimesh.load(mesh_path)
                        if isinstance(mesh, trimesh.Scene):
                            # If loaded as a Scene, extract the first Trimesh object
                            mesh = mesh.geometry[list(mesh.geometry.keys())[0]]
                        # Store mesh and scale as a tuple
                        link_meshes[link.name]["collision"].append((mesh, scale, origin_transform))
                    except Exception as e:
                        print(f"Failed to load collision mesh {mesh_path} for link {link.name}: {e}")
    return link_meshes


def detect_base_link(robot):
    """Detect the URDF kinematic root used as the palm / base frame.

    Floating links that are never a joint parent (e.g. unused dummy links) are
    ignored. If several connected roots exist, the tree with the most actuated
    joints is chosen.
    """
    child_links = {joint.child for joint in robot.joints}
    parent_links = {joint.parent for joint in robot.joints}
    roots = [link.name for link in robot.links if link.name not in child_links]
    candidates = [name for name in roots if name in parent_links] or roots
    if not candidates:
        raise ValueError("Could not detect base_link: URDF has no links.")
    if len(candidates) == 1:
        return candidates[0]

    children_map = {}
    joint_type_by_child = {}
    for joint in robot.joints:
        children_map.setdefault(joint.parent, []).append(joint.child)
        joint_type_by_child[joint.child] = joint.type
    actuated = {"revolute", "continuous", "prismatic"}

    def n_actuated(root):
        count = 0
        stack = [root]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for child in children_map.get(node, []):
                if joint_type_by_child.get(child) in actuated:
                    count += 1
                stack.append(child)
        return count

    return max(candidates, key=n_actuated)


def get_unfixed_kin_chains(robot_dir, link_meshes, base_link="base_link", correct_axis = True):
    # Load URDF
    robot = load_urdf(robot_dir, False)

    # Init Vars
    joint_info = dict()
    kin_chains = []
    vertices_dict = {}
    meshes_dict = {}
    transform = np.eye(4)

    # Include base_link to every kin_chain
    kc = [base_link]

    def get_transformed_meshes(link_name, T):
        """ Collect meshes of hand """
        meshes = []
        if link_name in link_meshes:
            for mesh_type in ["visual", "collision"]:
                for mesh, scale, origin_transform in link_meshes[link_name][mesh_type]:
                    transformed_mesh = mesh.copy()
                    transformed_mesh.vertices *= np.array(scale)
                    T_total = T @ origin_transform
                    transformed_mesh.apply_transform(T_total)
                    meshes.append(transformed_mesh)
        return meshes

    def get_transformed_vertices_with_face_points(link_name, T):
        """
        Retrieve and transform mesh vertices for a given link, applying scale and transformation,
        and include sampled points on mesh faces.

        Parameters:
        - link_name (str): Name of the link.
        - T (np.ndarray): 4x4 transformation matrix to apply to vertices.

        Returns:
        - np.ndarray: Array of transformed vertices and sampled points (Nx3), or empty array if no meshes.
        """
        # Initialize list to store all points (vertices and sampled points)
        vertices = []
        if link_name in link_meshes:
            for mesh_type in ["visual", "collision"]:
                for mesh, scale, origin_transform in link_meshes[link_name][mesh_type]:
                    # Get original vertices and faces from the mesh
                    original_vertices = mesh.vertices
                    faces = mesh.faces
                    
                    # Sample points on each face of the mesh
                    sampled_points = []
                    for face in faces:
                        A, B, C = original_vertices[face]  # Extract vertices A, B, C of the triangle
                        for _ in range(0):  # Generate 10 sample points per face
                            r1, r2 = np.random.rand(2)  # Random values for barycentric coordinates
                            sqrt_r1 = np.sqrt(r1)
                            u = 1 - sqrt_r1  # Barycentric coordinate u
                            v = sqrt_r1 * (1 - r2)  # Barycentric coordinate v
                            w = sqrt_r1 * r2  # Barycentric coordinate w
                            P = u * A + v * B + w * C  # Compute sampled point within triangle
                            sampled_points.append(P)
                    sampled_points = np.array(sampled_points) if sampled_points else np.array([])
                    
                    # Combine original vertices and sampled points
                    all_points = np.vstack((original_vertices, sampled_points)) if sampled_points.size > 0 else original_vertices
                    
                    # Scale all points according to URDF scale
                    scaled_points = all_points * np.array(scale)
                    
                    # Transform to the specified frame by applying T (homogeneous coordinates)
                    homogeneous_points = np.hstack((scaled_points, np.ones((scaled_points.shape[0], 1))))
                    T_total = T @ origin_transform

                    transformed_points = (T_total @ homogeneous_points.T).T[:, :3]
                    vertices.append(transformed_points)
        # Return stacked transformed points or empty array if no meshes
        return np.vstack(vertices) if vertices else np.array([])

    # Process base link and its fixed joints
    def process_base_and_fixed(parent_name, transform):
        """
        Collect vertices from parent link and its fixed joint descendants.

        Parameters:
        - parent_name (str): Current link name.
        - transform (np.ndarray): Accumulated transform.

        Returns:
        - list: List of vertex arrays.
        """
        vertices_list = []
        base_vertices = get_transformed_vertices_with_face_points(parent_name, transform) # Vertices with transform already implemented
        base_meshes = get_transformed_meshes(parent_name, transform)
        if base_vertices.size > 0:
            vertices_list.append(base_vertices)
        for joint in robot.joints:
            if joint.parent == parent_name and joint.type == "fixed":
                child_T = np.eye(4)
                try:
                    child_pose = joint.origin
                    translation = np.array(child_pose.xyz)
                    rotation = tf.euler_matrix(*child_pose.rpy, axes='sxyz')
                    child_T[:3, :3] = rotation[:3, :3]
                    child_T[:3, 3] = translation
                except: 
                    pass
                merged_T = transform @ child_T
                child_vertices, child_meshes = process_base_and_fixed(joint.child, merged_T)
                vertices_list.extend(child_vertices)
                base_meshes.extend(child_meshes)
        return vertices_list, base_meshes

    # Initialize base link vertices
    base_vertices, base_meshes = process_base_and_fixed(base_link, np.eye(4))
    vertices_dict[base_link] = np.vstack(base_vertices) if base_vertices else np.array([])
    meshes_dict[base_link] = base_meshes

    def get_child_joints(parent_name, transform, kin_chain):
        """
        Traverse URDF tree, accumulating vertices and tracking actuated joints.

        Parameters:
        - parent_name (str): Current parent link name.
        - transform (np.ndarray): Accumulated transform.
        - kin_chain (list): Current kinematic chain.
        - root_name (str): Name of the last actuated joint or base_link.
        """
        childs = []

        for joint in robot.joints:
            # Copy kin_chain to new memory
            wkc = kin_chain.copy() 

            if joint.parent == parent_name:
                # add to childs
                childs.append(joint.name) 
                
                # Get T matrix
                child_T = np.eye(4)
                try:
                    child_pose = joint.origin
                    translation = np.array(child_pose.xyz)
                    rotation = tf.euler_matrix(*child_pose.rpy, axes='sxyz')
                    child_T[:3, :3] = rotation[:3, :3]
                    child_T[:3, 3] = translation
                except: 
                    pass

                # Add transform from previous fixed joints to current transform
                merged_T = np.matmul(transform, child_T)
                
                # Add unfixed joints to kin chain
                if joint.type != "fixed": 
                    # Save joint as part of the kin_chain
                    wkc.append(joint.name)
                    
                    # Enforce axis to +z-axis and add to merged_T
                    T_z = axis_to_z_transform(joint.axis)
                    merged_T = np.matmul(merged_T, T_z)
                    
                    # axis = np.array(axis)
                    # p = np.linspace(0, 5, 20) * 0.01  
                    # points = p.reshape(20,1) * np.tile(axis,(20,1))
                    # print(f"joint {joint.name} axis {joint.axis}, {axis}, {points[-1]} to Z transform\n", T_z)

                    # Reset T for next
                    

                    # Collect vertices for this joint
                    vertices, meshes = process_base_and_fixed(joint.child, np.linalg.inv(T_z)) # Vertices in z alligned joint frame
                    vertices = np.vstack(vertices) if vertices else np.array([])
                    T_a = np.eye(4)
                    if (len(vertices)>0) & correct_axis:
                        z_max = np.max(vertices, axis=0)[2]
                        z_min = np.min(vertices, axis=0)[2]
                        z_mid = (z_max+z_min)/2
                        T_a[2,3] = z_mid
                        homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
                        vertices = (np.linalg.inv(T_a) @ homogeneous_vertices.T).T[:, :3]
                        # ax = np.array([0, 0, 1])
                        # p = np.linspace(0, 2, 20) * 0.01  
                        # points = p.reshape(20,1) * np.tile(ax,(20,1))
                        # vertices = np.vstack([vertices,points])

                        merged_T = np.matmul(merged_T,T_a)
                        for mesh in meshes:
                            mesh.apply_transform(np.linalg.inv(T_a))

                    vertices_dict[joint.name] = vertices
                    meshes_dict[joint.name] = meshes

                    # print("Tests:", vertices)
                    # rotation axis points
                    # Save joint info
                    xyz = merged_T[:3, 3]
                    quat = tf.quaternion_from_matrix(merged_T)

                    axis = np.array([0, 0, 1])

                    # Save joint info
                    joint_info[joint.name] = (
                        xyz,
                        quat,
                        joint.type,
                        axis,
                        joint.limit.lower,
                        joint.limit.upper,
                        np.linalg.norm(xyz) #l_link
                    )

                    merged_T = np.linalg.inv(T_z @ T_a)

                    # z = np.linspace(0, 5, 20) * 0.01  
                    # points = np.stack((np.zeros(z.flatten().shape), np.zeros(z.flatten().shape), z.flatten()), axis=1)
                    # vertices_dict[joint.name] = np.vstack((vertices_dict[joint.name], points)) if vertices else points
                                    

                c = get_child_joints(joint.child, merged_T, wkc)

                if c == []:
                    #Save finger only if it wasn't already saved (last if statement)
                    if joint.type == 'fixed':
                        wkc.append(joint.name)
                        xyz = merged_T[:3, 3]
                        rotation_matrix = merged_T[:3, :3]
                        quat = tf.quaternion_from_matrix(merged_T)
                        joint_info[joint.name] = (
                            xyz, # Translation
                            quat, # Rotation
                            joint.type, # Type of joint
                            joint.axis, # Axis (None)
                            None, # Lower Limit
                            None, # Upper Limit 
                            np.linalg.norm(xyz) #Length of link
                        )

                        merged_T = np.eye(4)

                    kin_chains.append(wkc)

        return childs
    
    T = np.eye(4)
    q = tf.quaternion_from_matrix(T)

    joint_info[base_link] = (
        np.array([0,0,0]),
        q,
        'fixed',
        None,
        None,
        None,
        None
    )
    get_child_joints(base_link, transform, kc)

    return joint_info, kin_chains, vertices_dict, meshes_dict

def process_type_ABC_joints(finger_chains, joint_info, kinematic_chains, vertices_dict, meshes_dict, q_0_dict):
    """Process joints as Type A B or C."""
    joint_type_info = {}

    for chain in finger_chains:
        joints = chain[1:-1]  # All joints from base to fingertip
        T = np.eye(4)
        for c, joint in enumerate(joints):
            # Compute fingertip position in joint frame at q_0
            T_joint_to_ft = compute_transform_from_joint_to_fingertip(chain, joint, joint_info, q_0_dict)
            p_ft_joint = (T_joint_to_ft)[:3, 3]
            ft_normal_joint = T_joint_to_ft[:3, 2]
            next_joint = chain[c+2]
            T_next_joint_to_ft = compute_transform_from_joint_to_fingertip(chain, next_joint, joint_info, q_0_dict)
            T_joint_to_next_joint = T_joint_to_ft @ np.linalg.inv(T_next_joint_to_ft)
            p_nj_joint = (T_joint_to_next_joint)[:3, 3]
            if np.linalg.norm(p_nj_joint) < 0.0025:
                next_joint = chain[c+3]
                T_next_joint_to_ft = compute_transform_from_joint_to_fingertip(chain, next_joint, joint_info, q_0_dict)
                T_joint_to_next_joint = T_joint_to_ft @ np.linalg.inv(T_next_joint_to_ft)
                p_nj_joint = (T_joint_to_next_joint)[:3, 3]
            
            # Determine joint type
            if abs(p_ft_joint[2]/2) > abs(p_ft_joint[0]) and abs(p_ft_joint[2]/2) > abs(p_ft_joint[1]):
                joint_type = "C"
                # Align x-axis to ft z-axis in joint frame 
                desired_x = ft_normal_joint[:2]
            elif abs(ft_normal_joint[2]) > abs(ft_normal_joint[0]) and abs(ft_normal_joint[2]) > abs(ft_normal_joint[1]):
                joint_type = "A"
                desired_x = p_ft_joint[:2]
                # desired_x = p_nj_joint[:2]
            else:
                joint_type = "B"
                desired_x = p_ft_joint[:2]
                # desired_x = p_nj_joint[:2]
            
            # Align x-axis
            angle = allign_joint_x_axis(joint, desired_x, joint_info, chain, vertices_dict, meshes_dict)
            
            joint_xyz = np.array(joint_info[joint][0])  # Translation from base_link to root 
            joint_quat = np.array(joint_info[joint][1])  # Translation from base_link to root 
            T_joint = tf.quaternion_matrix(joint_quat)
            T_joint[:3,3] = joint_xyz
            q_val = q_0_dict.get(joint, 0.0)  # Joint angle or default to 0
            R_z = tf.rotation_matrix(q_val, [0, 0, 1])  # Rotation around z-axis
            T = T @ T_joint @ R_z
            
            palm_xyz = joint_info["palm_normal"][0]
            palm_quat = joint_info["palm_normal"][1]
            T_base_to_palm = tf.quaternion_matrix(palm_quat)
            T_base_to_palm[:3,3] = palm_xyz
            T_palm_to_base = np.linalg.inv(T_base_to_palm)
            T_palm_to_joint = T_palm_to_base @ T
            T_joint_to_palm = np.linalg.inv(T_palm_to_joint)
            # palm_vertices = vertices_dict["palm_normal"]
            palm_center_local = np.array([0.0, 0.0, 0.05]) # Place holder for something better later on
            # palm_center_local = np.mean(palm_vertices, axis=0) if palm_vertices.size > 0 else np.array([0.0, 0.0, 0.0])
            ref_dir_homog = T_joint_to_palm @ np.append(palm_center_local, 1.0)
            ref_dir = ref_dir_homog[:3]

            # Save type-specific info
            joint_type_info[joint] = {
                "type": joint_type,
                "ref_dir": ref_dir[:3],
                "allign_angle": angle
            }

    return joint_type_info

def solve_inverse_kinematics(finger_chains, joint_info, kinematic_chains, vertices_dict, joint_type_info, q_opened):
    """Solve inverse kinematics to create q_max dictionary for all joints in finger chains."""
    all_joints = set()
    for chain in finger_chains:
        all_joints.update(chain[1:-1])
    joint_index_dict = {joint: idx for idx, joint in enumerate(all_joints)}
    q_0 = np.zeros(len(all_joints))

    # Get T_base_to_palm
    palm_xyz = joint_info["palm_normal"][0]
    palm_quat = joint_info["palm_normal"][1]
    T_base_to_palm = tf.quaternion_matrix(palm_quat)
    T_base_to_palm[:3, 3] = palm_xyz

    # Prepare palm vertices homogeneous
    palm_vertices = vertices_dict["palm_normal"]
    palm_vertices_h = np.hstack((palm_vertices, np.ones((palm_vertices.shape[0], 1))))


    for chain in finger_chains:
        a_count = 0
        for joint in chain[1:-1]:
            print("Solving for:", joint)
            T_base_to_joint = compute_transform_to_joint(joint, finger_chains, joint_info, dict(zip(all_joints, q_0)), base_link = kinematic_chains[0][0])
            T_joint_to_base = np.linalg.inv(T_base_to_joint)
            T_joint_to_palm = T_joint_to_base @ T_base_to_palm

            transformed_h = T_joint_to_palm @ palm_vertices_h.T
            joint_points = transformed_h.T[:, :3]
            centroid = np.mean(joint_points, axis=0)
            palm_point = joint_points[0,:]
            last_point = joint_points[-1,:]
            joint_points = np.tile(last_point,(20,1)) # Will be max z point

            joint_type = joint_type_info.get(joint, {}).get("type")
            if joint_type == "A":
                q_0[joint_index_dict[joint]] = q_opened[joint]
                continue
                q_0[joint_index_dict[joint]] = solve_for_proto_B_joint(
                    q_0, joint, joint_index_dict, joint_points, centroid, joint_info, joint_type_info
                )
            elif joint_type == "B":
                q_0[joint_index_dict[joint]] = solve_for_proto_B_joint(
                    q_0, joint, joint_index_dict, joint_points, centroid, joint_info, joint_type_info
                    )
            elif joint_type == "C":
                T_joint_to_ft = compute_transform_from_joint_to_fingertip(chain, joint, joint_info, q_opened)
                ft_pos = T_joint_to_ft[:3, 3]
                ft_normal = T_joint_to_ft[:3, 2]
                q_0[joint_index_dict[joint]] = solve_for_proto_C_joint(
                    q_0, joint, joint_index_dict, centroid, joint_info, ft_pos, ft_normal
                )

            print("Result", joint_type, q_0[joint_index_dict[joint]])
            print("bounds", joint_info[joint][4], joint_info[joint][5])
            # print(q_0)

    q_max = {joint: q_0[idx] for joint, idx in joint_index_dict.items()}
    return q_max

def construct_plane(vertices_dict, joint_info, kinematic_chains, base_link="base_link"):
    """Create a plane frame fitted to finger root positions with points along the normal.

    Parameters:
    - vertices_dict (dict): Maps joint names to vertex arrays.
    - joint_info (dict): Maps joint names to (xyz, quat, joint_type, axis, lower, upper, length).
    - kinematic_chains (list of lists): Defines the hierarchy of joints from base_link to end effectors.
    - base_link (str): Name of the base link frame.

    Returns:
    - np.ndarray: 4x4 transformation matrix from base_link to plane frame.
    """
    # Find the palm_normal chain
    palm_chain = next((chain for chain in kinematic_chains if "palm_normal" in chain), None)
    if palm_chain is None:
        raise ValueError("No kinematic chain contains 'palm_normal'")

    # Get transformation from base_link to palm_normal
    xyz_palm = joint_info["palm_normal"][0]  # Translation in base frame
    quat_palm = joint_info["palm_normal"][1]  # Quaternion in base frame
    R_palm = tf.quaternion_matrix(quat_palm)[:3, :3]  # Rotation matrix
    t_palm = np.array(xyz_palm)  # Ensure numpy array
    T_base_to_palm = np.eye(4)
    T_base_to_palm[:3, :3] = R_palm
    T_base_to_palm[:3, 3] = t_palm
    T_palm_to_base = np.linalg.inv(T_base_to_palm)

    # Identify finger chains and their roots
    finger_chains = [chain for chain in kinematic_chains if chain != palm_chain]
    finger_roots = [chain[1] for chain in finger_chains if len(chain) > 1]

    if not finger_roots:
        raise ValueError("No finger roots found in kinematic chains")

    # Compute finger root positions in base_link frame
    root_positions_base = [np.array(joint_info[root][0]) for root in finger_roots]

    root_positions_palm = [(T_palm_to_base @ np.append(root, 1))[:3] for root in root_positions_base]
    root_positions_base = np.array(root_positions_base)
    root_positions_palm = np.array(root_positions_palm)  # Ensure (N, 3) array
    # print(root_positions_palm)

    # Fit plane to finger root positions
    min_pos = np.min(root_positions_palm, axis=0)
    max_pos = np.max(root_positions_palm, axis=0)
    centroid = (min_pos + max_pos) / 2
    centroid[2] = 0 # No z movement
    # print(centroid)
    centered = root_positions_base - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1, :]
    normal /= np.linalg.norm(normal)


    # Choose sign of normal to align with palm z-axis direction
    palm_z_base = np.dot(R_palm, np.array([0, 0, 1]))
    if np.dot(normal, palm_z_base) < 0:
        normal = -normal

    # Set z_axis to the chosen normal
    z_axis = normal

    # Construct rotation matrix for the plane frame
    ref = np.array([1, 0, 0]) if np.abs(np.dot(z_axis, [1, 0, 0])) < 0.999 else np.array([0, 1, 0])
    y_axis = np.cross(z_axis, ref)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    R = np.stack((x_axis, y_axis, z_axis), axis=1)

    # Construct T_base_to_plane
    T_base_to_plane = np.eye(4)
    T_base_to_plane[:3, :3] = R # Sets palm as best plane that holds all roots
    T_base_to_plane[:3, :3] = R_palm
    T_base_to_plane[:3, 3] = (T_base_to_palm @ np.append(centroid,1))[:3]

    # Extract translation and quaternion
    xyz_plane = T_base_to_plane[:3, 3]
    quat_plane = tf.quaternion_from_matrix(T_base_to_plane)
    # t_palm_h = np.append(t_palm, 1)
    # original_palm = np.linalg.inv(T_base_to_plane) @ t_palm_h
    # z_shift = [0.0, 0.0, original_palm[2]]
    # xyz_plane = xyz_plane + T_base_to_plane[:3,:3] @ z_shift

    # Add to joint_info
    joint_info["palm_normal"] = (
        xyz_plane,           # Translation
        quat_plane,          # Quaternion
        "fixed",             # Type (fixed frame)
        None,                # Axis (not applicable)
        None,                # Lower limit
        None,                # Upper limit
        None                 # No length for plane
    )
    

    # Generate points along the normal (perpendicular) in local plane frame: along z from -1m to 1m
    num_points = 10000
    z_values = np.linspace(-1.0, 100.0, num_points)
    local_points = np.zeros((num_points, 3))
    local_points[:, 2] = z_values

    # Filter points based on non-negative z in palm normal frame
    filtered_points = []
    for p_local in local_points:
        p_base_h = np.append(p_local, 1)
        p_base = (T_base_to_plane @ p_base_h)[:3]
        p_palm_h = np.append(p_base, 1)
        p_palm = (T_palm_to_base @ p_palm_h)[:3]
        if p_palm[2] >= 0:
            filtered_points.append(p_local)

    filtered_local = np.array(filtered_points)

    # Add to vertices_dict (only the filtered perpendicular points)
    vertices_dict["palm_normal"] = filtered_local

    return T_base_to_plane

def construct_sphere_with_palm(vertices_dict, joint_info, kinematic_chains, q_0, q_max, base_link="base_link", sphere_factor = 4/(2.5*(np.pi))): #! Sphere factor was 4/3*pi
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

    # Ensure the palm_normal joint is properly identified
    if palm_chain is None:
        raise ValueError("No chain containing 'palm_normal' found in kinematic_chains")

    # Identify finger chains and their roots
    finger_chains = [chain for chain in kinematic_chains if chain != palm_chain]
    finger_roots = [chain[1] for chain in finger_chains if len(chain) > 1]

    if not finger_roots:
        raise ValueError("No finger roots found in kinematic chains")

    # Identify finger chains (excluding palm_normal chain)
    fingertips = [chain[-1] for chain in finger_chains]  # Last joint in each chain
    roots = [chain[1] for chain in finger_chains] 

    # Compute transformation from base_link to palm_normal at q_max
    xyz_palm = joint_info["palm_normal"][0]  # Translation in base frame
    quat_palm = joint_info["palm_normal"][1]  # Quaternion in base frame
    R_palm = tf.quaternion_matrix(quat_palm)[:3, :3]  # Rotation matrix
    T_base_to_palm = np.eye(4)
    T_base_to_palm[:3, :3] = R_palm
    T_base_to_palm[:3, 3] = xyz_palm
    T_palm_to_base = np.linalg.inv(T_base_to_palm)

    # Compute fingertip positions in base_link frame at q_max
    fingertip_positions_base = []
    root_positions_base = []
    l_all = []
    for fingertip in fingertips:
        # Compute transformation from base_link to fingertip at q_max
        T_base_to_fingertip = compute_transform_to_joint(fingertip, kinematic_chains, joint_info, q=q_0, base_link=base_link)
        T_base_to_root = compute_transform_to_joint(fingertip, kinematic_chains, joint_info, q=q_0, base_link=base_link)
        # Extract position in base frame
        p_fingertip_base = T_base_to_fingertip[:3, 3]
        p_root_base = T_base_to_root[:3, 3]
        fingertip_positions_base.append(p_fingertip_base)
        root_positions_base.append(p_root_base)
        tip_in_palm = (T_palm_to_base @ T_base_to_fingertip)[:3,3]
        l = np.linalg.norm(tip_in_palm)
        l_all.append(l)

    # Compute root positions in palm_normal frame and extract z-coordinates
    z_values = []
    for p_root_base in root_positions_base:
        p_root_palm = R_palm.T @ (p_root_base - xyz_palm)
        z_values.append(p_root_palm[2])
    mean_z = np.mean(z_values)
    max_z = np.max(z_values)

    # l_all = []
    # for chain in finger_chains:
    #     # Finger chain length
    #     root = chain[1]
    #     root_xyz = np.array(joint_info[root][0], dtype = float)
    #     l = np.linalg.norm(root_xyz - t_palm)
    #     print("root", l)
    #     for j in chain[2:]:
    #         xyz = np.array(joint_info[j][0])
    #         l+=np.linalg.norm(xyz)
    #         print(j , np.linalg.norm(xyz))
    #     print("total", l)
    #     l_all.append(l)


    mean_l = np.mean(l_all)
    # Assuming mean_z is positive; if negative, the palm_normal orientation may need review
    # if mean_z < 0:
    #     raise ValueError("Mean z-coordinate of fingertips in palm_normal frame is negative; check orientation")
    
    # Compute radius as half the mean z (approximating the reach along the palm normal direction)
    # radius = (mean_z) * sphere_factor 
    radius = (mean_l) * sphere_factor 

    # Compute sphere origin in base_link frame: translated along palm_normal z-axis by half mean_z
    # origin = T_base_to_palm @ np.array([0, 0, 0.05*radius, 1])
    origin = T_base_to_palm @ np.array([0, 0, radius, 1])
    origin = origin[:3]

    # Define rotation matrix for pi rotation around x-axis to flip y and z (making sphere z point back towards palm_normal)
    rot_flip = np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]])

    # Compute sphere rotation matrix: palm_normal rotation composed with the flip
    R = R_palm @ rot_flip
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print(f"Sphere Radius: {radius}")
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print()
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
    p = sample_fibonacci_points(10000)
    theta, phi = p[:, 0], p[:, 1]
    radius = joint_info["sphere_frame"][6]  # Assuming radius is the 7th element
    phi_breaks = [0, np.pi/2, 3 * np.pi / 4, np.pi]
    radius_values = [1.0, 1.0, 1.0, 1.0]
    sphere_points = compute_axisymmetric_deformed_points(phi_breaks, radius_values, p, base_radius=radius)
    
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

def compute_main_joints(joint_info, kinematic_chains, q_0, q_max, joint_type_info, base_link = "base_link"):
    """Identify Type 'main' joints and their theta ranges for each finger.

    Parameters:
    - joint_info (dict): Maps joint names to (xyz, quat, joint_type, axis, lower, upper, length).
    - kinematic_chains (list of lists): Defines joint hierarchy.
    - q_0 (dict): Initial joint angles.
    - q_max (dict): .
    - joint_type_info:

    Returns:
    - tuple: (main_joints, theta_ranges) where main_joints is a list of the fingers main joints
             (None if none), and theta_ranges is a list of (min_theta, max_theta) tuples
             (None if no Type "A" joint), with ranges computed around the circular midpoint.
    """
    # Initialize lists
    main_joints = []
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

        # Track best joint candidate
        max_theta_range = 0
        main_candidate = None
        min_theta_candidate = None
        max_theta_candidate = None

        # Check if finger has an azimuthal joint -> A >> C >> B
        azimuthal_joint_flag = False
        A_joints = []
        for joint in joints:
            if joint_type_info[joint]["type"] == "A":
                azimuthal_joint_flag = True
                A_joints.append(joint)

        # Check for roll joint 
        roll_joint_flag = False
        C_joints = []
        for joint in joints:
            if joint_type_info[joint]["type"] == "C":
                roll_joint_flag = True
                C_joints.append(joint)

        if azimuthal_joint_flag: 
            print(f"Finger Chain {finger_chain} has type A joints {A_joints}")
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
                    main_candidate = joint
                    min_theta_candidate = min_theta
                    max_theta_candidate = max_theta
            
            # If multiple type A's, deactivate all but the main
            if len(A_joints)> 1:
                for joint in A_joints:
                    if joint!= main_candidate:
                        joint_type_info[joint]["type"] = "B"
                        joint_type_info[joint]['og_type'] = "A"
            joint_type_info[main_candidate]["azimuthal_flag"] = True
        elif roll_joint_flag: 
            print(f"Finger Chain {finger_chain} has type C joints {C_joints}")
            for joint in C_joints:
                lower = joint_info[joint][4]
                upper = joint_info[joint][5]
                angles = np.linspace(lower, upper, 100)

                theta_values = []
                xy_projections = []

                # Check if root joint
                if joint == finger_chain[1]:
                    q = q_max.copy() 
                else:
                    q = q_0.copy() 
                    idx = finger_chain.index(joint)
                    n_j = finger_chain[idx+1]
                    q[n_j]= joint_info[n_j][5] # Max range of next joint (assumes positive closes the hand)

                # Sample all joint angles
                for angle in angles:
                    q[joint] = angle
                    
                    # Get transform to fingertip with the q changed
                    T_base_to_fingertip = compute_transform_to_joint(
                        fingertip, kinematic_chains, joint_info, q, base_link
                    )
                    p_fingertip_base = T_base_to_fingertip[:3, 3]

                    # Transform to sphere frame
                    p_fingertip_sphere_h = T_sphere_to_base @ np.append(p_fingertip_base, 1)
                    p_fingertip_sphere = p_fingertip_sphere_h[:3]

                    # Get Azimuthal angle of every q value
                    theta = np.arctan2(p_fingertip_sphere[1], p_fingertip_sphere[0])

                    theta_values.append(theta)

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
                    main_candidate = joint
                    min_theta_candidate = min_theta
                    max_theta_candidate = max_theta
            joint_type_info[main_candidate]["azimuthal_flag"] = False
        else:
            print(f"Finger Chain {finger_chain} has no type A or type C joints, it will be considered azimuthally bound")

            # # If there is no joint, evaluate the other BC joints and get the best candidate.
            # for joint in joints:
            #     lower = joint_info[joint][4]
            #     upper = joint_info[joint][5]
            #     angles = np.linspace(lower, upper, 100)

            #     theta_values = []
            #     xy_projections = []


            #     # Sample all joint angles
            #     for angle in angles:
            #         q = q_0.copy() 
            #         q[joint] = angle

            #         # Get transform to fingertip with the q changed
            #         T_base_to_fingertip = compute_transform_to_joint(
            #             fingertip, kinematic_chains, joint_info, q, base_link
            #         )
            #         p_fingertip_base = T_base_to_fingertip[:3, 3]

            #         # Check z in palm frame, eliminate any below palm normal
            #         p_fingertip_palm_h = T_palm_to_base @ np.append(p_fingertip_base, 1)
            #         if p_fingertip_palm_h[2] <= 0:
            #             continue

            #         # Transform to sphere frame
            #         p_fingertip_sphere_h = T_sphere_to_base @ np.append(p_fingertip_base, 1)
            #         p_fingertip_sphere = p_fingertip_sphere_h[:3]

            #         # Get Azimuthal angle of every q value
            #         theta = np.arctan2(p_fingertip_sphere[1], p_fingertip_sphere[0])

            #         theta_values.append(theta)
            #         xy_projections.append(p_fingertip_sphere[:2]) # For exclusion condition

            #     if not theta_values:
            #         continue

            #     # Compute circular midpoint and range
            #     theta_values = np.array(theta_values)
            #     theta_mid = np.arctan2(np.mean(np.sin(theta_values)), np.mean(np.cos(theta_values)))
            #     shifted_theta = np.mod(theta_values - theta_mid + np.pi, 2 * np.pi) - np.pi
            #     min_shift = np.min(shifted_theta)
            #     max_shift = np.max(shifted_theta)

            #     # Answer can now be from -3π to 3π, accounts for circularity
            #     min_theta = theta_mid + min_shift
            #     max_theta = theta_mid + max_shift
            #     theta_range = max_theta - min_theta
            #     print(f"{joint} theta range {theta_range}")

            #     # Update candidate
            #     xy_projections = np.array(xy_projections)
            #     if len(xy_projections) > 1: # Check that the finger joint, doesn't pass through the sphere center directly. 
            #         centroid = np.mean(xy_projections, axis=0)
            #         U, S, Vt = np.linalg.svd(xy_projections - centroid)
            #         direction = Vt[0]
            #         t = -np.dot(centroid, direction)
            #         closest_point = centroid + t * direction
            #         distance = np.linalg.norm(closest_point)
            #         print(f"{joint} distance {distance} {radius}")
            #         if distance < 0.15 * radius: # Threshold
            #             continue
                    
            #     if theta_range > max_theta_range:
            #         max_theta_range = theta_range
            #         main_candidate = joint
            #         min_theta_candidate = min_theta
            #         max_theta_candidate = max_theta
            # joint_type_info[main_candidate]["azimuthal_flag"] = False
            

        # Assign Type "A" joint if conditions are met
        if main_candidate and max_theta_range > np.deg2rad(10):
            print(f"Found main joint {main_candidate}")
            main_joints.append(main_candidate)
            theta_ranges.append((min_theta_candidate, max_theta_candidate))
        else:
            main_joints.append(None)
            theta_ranges.append(None)

    return main_joints, theta_ranges

def compute_finger_chain_order(joint_info, kinematic_chains, theta_ranges, q_0, left=False):
    """
    Compute finger chain order based on spherical median and assign indices.
    If left=False, aligns to mean of median and closest smaller theta; if left=True, aligns to mean of median and closest larger theta for 5 fingers.

    Parameters:
    - joint_info (dict): Joint information dictionary with sphere_frame and chain root positions.
    - kinematic_chains (list of lists): List of kinematic chains for fingers.
    - theta_ranges (list): List of theta ranges for each chain.
    - q_0 (dict): Joint angles for the initial configuration.
    - left (bool): If False, use median and smaller theta; if True, use median and larger theta (default: False).

    Returns:
    - tuple: (theta_alignment, indices)
        - theta_alignment (float): Adjusted alignment angle in radians.
        - indices (list): Ordered indices for each finger chain (e.g., [2] for 1 chain, [1, 2, 3] for 3 chains).
    """
    # Extract sphere frame transformation
    sphere_xyz = joint_info["sphere_frame"][0]
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3, 3] = sphere_xyz
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)

    palm_xyz = joint_info["palm_normal"][0]
    palm_quat = joint_info["palm_normal"][1]
    T_base_to_palm = tf.quaternion_matrix(palm_quat)
    T_base_to_palm[:3, 3] = palm_xyz
    T_palm_to_base = np.linalg.inv(T_base_to_palm)

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
    print()
    print("theta_values")
    print(theta_values)
    print(theta_alignment)
    print()
    print()

    # Find spherical median (chain closest to theta_alignment)
    min_diff = float('inf')
    median_idx = 0
    for idx, theta in enumerate(theta_values):
        diff = abs((theta - theta_alignment + np.pi) % (2 * np.pi) - np.pi)
        if diff < min_diff:
            min_diff = diff
            median_idx = idx
    
    alignment_idx = median_idx


    # For 5 fingers, adjust theta_alignment to mean of median and closest smaller/larger theta based on left
    # if num_chains == 5:
    #     theta_median = theta_values[median_idx]
    #     theta_diffs = [(theta - theta_median + np.pi) % (2 * np.pi) - np.pi for theta in theta_values]
        
    #     if left:
    #         # Find closest theta that is larger than median
    #         valid_diffs = [d if d > 0 else float('inf') for d in theta_diffs]
    #     else:
    #         # Find closest theta that is smaller than median
    #         valid_diffs = [d if d < 0 else float('inf') for d in theta_diffs]
        
    #     # Find index of closest valid theta
    #     closest_idx = np.argmin([abs(d) for d in valid_diffs])
    #     if valid_diffs[closest_idx] != float('inf'):
    #         # Compute mean of median theta and closest valid theta
    #         theta_closest = theta_values[closest_idx]
    #         sum_sin = np.sin(theta_median) + np.sin(theta_closest)
    #         sum_cos = np.cos(theta_median) + np.cos(theta_closest)
    #         theta_alignment = np.arctan2(sum_sin, sum_cos)
    #         alignment_idx = median_idx  # Keep alignment_idx as median for indexing
    #     else:
    #         # Fallback to median if no valid theta found
    #         theta_alignment = theta_median
    # else:
    #     theta_alignment = theta_values[median_idx]
    
    theta_alignment = theta_values[median_idx]

    c = 0
    for theta_range in theta_ranges:  # Align directly to any joint without azimuthal joints if any
        if theta_range is None:
            root_xyz = joint_info[finger_chains[c][1]][0]  # Root position of the chain
            root_xyz_h = np.append(root_xyz, 1)  # Homogeneous coordinate
            root_in_sphere = T_sphere_to_base @ root_xyz_h
            theta_alignment = np.arctan2(root_in_sphere[1], root_in_sphere[0])
            alignment_idx = c
        c += 1

    print("Alligning with chain:", finger_chains[alignment_idx] )

    # Update sphere frame with rotation around z-axis using final theta_alignment
    R = tf.rotation_matrix(theta_alignment, [0, 0, 1])
    T_new = T_base_to_sphere @ R
    xyz_sphere = T_new[:3, 3]
    quat_sphere = tf.quaternion_from_matrix(T_new)
    joint_info["sphere_frame"] = (
        xyz_sphere,
        quat_sphere,
        "fixed",
        None,
        None,
        None,
        joint_info["sphere_frame"][6]  # Preserve radius
    )

    # Allign palm normal x-axis to sphere x-axis
    sphere_point = np.array([1.0, 0.0, 0.0, 1.0])
    point_in_palm = T_palm_to_base @ T_new @ sphere_point
    angle = np.arctan2(point_in_palm[1], point_in_palm[0])
    R_z_palm = tf.rotation_matrix(angle, [0, 0, 1])
    T_new_palm = T_base_to_palm @ R_z_palm

    # Update joint_info for palm_normal
    xyz_palm_new = T_new_palm[:3, 3]
    quat_palm_new = tf.quaternion_from_matrix(T_new_palm)
    joint_info["palm_normal"] = (
        xyz_palm_new,
        quat_palm_new,
        "fixed",
        None,
        None,
        None,
        None  # No length for plane
    )

    # Compute shifted theta values relative to theta_alignment
    theta_shifted = (np.array(theta_values) - theta_alignment + np.pi) % (2 * np.pi) - np.pi

    # Assign indices dynamically based on number of chains and their positions
    indices = [None] * num_chains
    print("Theta shifted", theta_shifted)
    sorted_indices = np.argsort(theta_shifted)
    
    # Assign index 2 to the spherical median (midpoint)
    indices[alignment_idx] = 2
    alignment_pos = sorted_indices.tolist().index(alignment_idx)

    # Assign indices to chains on the negative side (if any)
    j = 0 
    for i in sorted_indices[:alignment_pos].tolist():
        indices[i] = j  # 0, 1, etc., for negative side
        j += 1

    j = 0
    # Assign indices to chains on the positive side (if any)
    for i in np.flip(sorted_indices[(alignment_pos+1):]).tolist():
        indices[i] = 4 - j  # 3, 4, etc., for positive side
        j += 1

    return theta_alignment, indices

def compute_finger_chain_order_translation(joint_info, kinematic_chains, theta_ranges, q_0, left=False, base_link="base_link", joint_type_info=None):
    """
    Compute finger chain order based on spherical median and assign indices.
    If left=False, aligns to mean of median and closest smaller theta; if left=True, aligns to mean of median and closest larger theta for 5 fingers.
    Parameters:
    - joint_info (dict): Joint information dictionary with sphere_frame and chain root positions.
    - kinematic_chains (list of lists): List of kinematic chains for fingers.
    - theta_ranges (list): List of theta ranges for each chain.
    - q_0 (dict): Joint angles for the initial configuration.
    - left (bool): If False, use median and smaller theta; if True, use median and larger theta (default: False).
    - base_link (str): The base link for transformations (default: "base_link").
    - joint_type_info (dict): Joint type information.
    Returns:
    - tuple: (theta_alignment, indices)
        - theta_alignment (float): Adjusted alignment angle in radians.
        - indices (list): Ordered indices for each finger chain (e.g., [2] for 1 chain, [1, 2, 3] for 3 chains).
    """
    # Extract and compute transformations for sphere and palm frames
    sphere_xyz = joint_info["sphere_frame"][0]
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3, 3] = sphere_xyz
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)
    palm_xyz = joint_info["palm_normal"][0]
    palm_quat = joint_info["palm_normal"][1]
    T_base_to_palm = tf.quaternion_matrix(palm_quat)
    T_base_to_palm[:3, 3] = palm_xyz
    T_palm_to_base = np.linalg.inv(T_base_to_palm)
    
    # Filter to get only finger chains
    finger_chains = [
        chain for chain in kinematic_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]
    num_chains = len(finger_chains)
    
    # Compute azimuthal angles (theta) for each finger root in sphere frame
    theta_values = []
    for chain in finger_chains:
        root_joint = chain[1]
        T_base_to_root = compute_transform_to_joint(root_joint, kinematic_chains, joint_info, q=q_0, base_link=base_link)
        root_xyz = T_base_to_root[:3, 3]
        root_xyz_h = np.append(root_xyz, 1)
        root_in_sphere = T_sphere_to_base @ root_xyz_h
        theta = np.arctan2(root_in_sphere[1], root_in_sphere[0])
        theta_values.append(theta)
    
    # Compute circular mean of theta values
    sum_sin = sum(np.sin(theta) for theta in theta_values)
    sum_cos = sum(np.cos(theta) for theta in theta_values)
    theta_alignment = np.arctan2(sum_sin, sum_cos)
    print()
    print("theta_values")
    print(theta_values)
    print(theta_alignment)
    print()
    print()
    
    # Find index of spherical median (closest to circular mean)
    min_diff = float('inf')
    median_idx = 0
    for idx, theta in enumerate(theta_values):
        diff = abs((theta - theta_alignment + np.pi) % (2 * np.pi) - np.pi)
        if diff < min_diff:
            min_diff = diff
            median_idx = idx
    alignment_idx = median_idx
    
    # Set theta_alignment to median theta (commented code for 5-finger adjustment not active)
    theta_alignment = theta_values[median_idx]
    
    # Override alignment for chains without theta range (no azimuthal joint)
    c = 0
    for theta_range in theta_ranges:
        if theta_range is None:
            root_joint = finger_chains[c][1]
            T_base_to_root = compute_transform_to_joint(root_joint, kinematic_chains, joint_info, q=q_0, base_link=base_link)
            root_xyz = T_base_to_root[:3, 3]
            root_xyz_h = np.append(root_xyz, 1)
            root_in_sphere = T_sphere_to_base @ root_xyz_h
            theta_alignment = np.arctan2(root_in_sphere[1], root_in_sphere[0])
            alignment_idx = c
        c += 1
    print("Alligning with chain:", finger_chains[alignment_idx] )
    
    # Align sphere frame so median finger's root-to-tip line has y=0 in sphere frame
    median_chain = finger_chains[alignment_idx]
    root_joint = median_chain[1]
    fingertip = median_chain[-1]
    T_base_to_root = compute_transform_to_joint(root_joint, kinematic_chains, joint_info, q=q_0, base_link=base_link)
    p_root_base = T_base_to_root[:3, 3]
    T_base_to_tip = compute_transform_to_joint(fingertip, kinematic_chains, joint_info, q=q_0, base_link=base_link)
    p_tip_base = T_base_to_tip[:3, 3]
    D_base = p_tip_base - p_root_base
    
    # Compute positions and direction in current sphere frame
    pos = sphere_xyz
    Rot = T_base_to_sphere[:3, :3]
    p_root_s = Rot.T @ (p_root_base - pos)
    p_tip_s = Rot.T @ (p_tip_base - pos)
    D_s = p_tip_s - p_root_s
    
    # Compute rotation angle phi and translation v to align line to x-axis (y=0)
    D_sx, D_sy = D_s[0], D_s[1]
    r_xy = np.linalg.norm([D_sx, D_sy])
    if r_xy < 1e-10:
        phi = np.arctan2(p_root_s[1], p_root_s[0])
        v = np.zeros(3)
    else:
        phi = np.arctan2(D_sy, D_sx)
        n = np.array([-np.sin(phi), np.cos(phi), 0.0])
        s = np.dot(n, p_root_s)
        v = s * n
    
    # Apply translation to sphere position
    delta_base = Rot @ v
    new_pos = pos + delta_base
    T_base_to_sphere_trans = np.copy(T_base_to_sphere)
    T_base_to_sphere_trans[:3, 3] = new_pos
    
    # Apply rotation around z-axis by phi
    R = tf.rotation_matrix(phi, [0, 0, 1])
    T_new = T_base_to_sphere_trans @ R
    xyz_sphere = T_new[:3, 3]
    quat_sphere = tf.quaternion_from_matrix(T_new)
    joint_info["sphere_frame"] = (
        xyz_sphere,
        quat_sphere,
        "fixed",
        None,
        None,
        None,
        joint_info["sphere_frame"][6]  # Preserve radius
    )
    
    # Align palm normal x-axis to new sphere x-axis
    sphere_point = np.array([1.0, 0.0, 0.0, 1.0])
    point_in_palm = T_palm_to_base @ T_new @ sphere_point
    angle = np.arctan2(point_in_palm[1], point_in_palm[0])
    R_z_palm = tf.rotation_matrix(angle, [0, 0, 1])
    T_new_palm = T_base_to_palm @ R_z_palm
    xyz_palm_new = T_new_palm[:3, 3]
    quat_palm_new = tf.quaternion_from_matrix(T_new_palm)
    joint_info["palm_normal"] = (
        xyz_palm_new,
        quat_palm_new,
        "fixed",
        None,
        None,
        None,
        None  # No length for plane
    )
    
    # Compute relative transform from palm to sphere
    T_palm_to_sphere = np.linalg.inv(T_new_palm) @ T_new
    
    # Find first "B" joints in each finger chain (after root)
    b_joints = []
    if joint_type_info is not None:
        for chain in finger_chains:
            found = False
            for j in chain[2:]:
                if joint_info.get(j, [None, None, None])[2] == "revolute" and joint_type_info.get(j, {}).get("type") == "A":
                    b_joints.append(j)
                    found = True
                    break
            if not found:
                pass  # No "B" joint found for this chain
    
    # Optimize y-rotation if "B" joints exist
    phi_opt = 0.0
    if joint_type_info is not None and len(b_joints) > 0:
        def cost(phi):
            R_y = tf.rotation_matrix(phi, [0, 1, 0])
            T_base_to_palm_phi = T_new_palm @ R_y
            T_palm_phi_to_base = np.linalg.inv(T_base_to_palm_phi)
            zs = []
            for j in b_joints:
                T_base_to_j = compute_transform_to_joint(j, kinematic_chains, joint_info, q=q_0, base_link=base_link)
                p_base = T_base_to_j[:3, 3]
                p_h = np.append(p_base, 1)
                p_palm = T_palm_phi_to_base @ p_h
                zs.append(p_palm[2])
            if any(z < 0 for z in zs):
                return np.inf
            return max(zs) if zs else 0.0
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(cost, bounds=(-np.pi/4, np.pi/4), method='bounded')

        if res.success and res.fun != np.inf:
            phi_opt = res.x
    
    # Apply optimal y-rotation to palm
    R_y_opt = tf.rotation_matrix(phi_opt, [0, 1, 0])
    T_base_to_palm_final = T_new_palm @ R_y_opt
    xyz_palm_final = T_base_to_palm_final[:3, 3]
    quat_palm_final = tf.quaternion_from_matrix(T_base_to_palm_final)
    joint_info["palm_normal"] = (
        xyz_palm_final,
        quat_palm_final,
        "fixed",
        None,
        None,
        None,
        None  # No length for plane
    )
    
    # Update sphere frame using relative transform
    T_base_to_sphere_final = T_base_to_palm_final @ T_palm_to_sphere
    xyz_sphere_final = T_base_to_sphere_final[:3, 3]
    quat_sphere_final = tf.quaternion_from_matrix(T_base_to_sphere_final)
    joint_info["sphere_frame"] = (
        xyz_sphere_final,
        quat_sphere_final,
        "fixed",
        None,
        None,
        None,
        joint_info["sphere_frame"][6]  # Preserve radius
    )
    
    # Recompute theta_values in updated sphere frame
    T_base_to_sphere = T_base_to_sphere_final
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)
    theta_values = []
    for chain in finger_chains:
        root_joint = chain[1]
        T_base_to_root = compute_transform_to_joint(root_joint, kinematic_chains, joint_info, q=q_0, base_link=base_link)
        root_xyz = T_base_to_root[:3, 3]
        root_xyz_h = np.append(root_xyz, 1)
        root_in_sphere = T_sphere_to_base @ root_xyz_h
        theta = np.arctan2(root_in_sphere[1], root_in_sphere[0])
        theta_values.append(theta)
    
    # Update theta_alignment to median theta in new frame
    theta_alignment = theta_values[alignment_idx]
    
    # Compute shifted thetas relative to alignment
    theta_shifted = (np.array(theta_values) - theta_alignment + np.pi) % (2 * np.pi) - np.pi
    
    # Assign indices: 2 to alignment chain, lower to negative side, higher to positive side
    indices = [None] * num_chains
    print("Theta shifted", theta_shifted)
    sorted_indices = np.argsort(theta_shifted)
    indices[alignment_idx] = 2
    alignment_pos = sorted_indices.tolist().index(alignment_idx)
    j = 0
    for i in sorted_indices[:alignment_pos].tolist():
        indices[i] = j  # 0, 1, etc., for negative side
        j += 1
    j = 0
    for i in np.flip(sorted_indices[(alignment_pos+1):]).tolist():
        indices[i] = 4 - j  # 3, 4, etc., for positive side
        j += 1
    return theta_alignment, indices

def process_type_D_joints(finger_chains, type_A_joints, joint_type_info):
    """Process Type D joints."""
    update_info = {}
    for chain in finger_chains:
        type_A_joint = find_type_A_joint_in_chain(chain, type_A_joints)
        if type_A_joint is None:
            continue
        idx_A = chain.index(type_A_joint)
        type_D_joints = chain[1:idx_A]  # Between base_link and Type A
        
        for joint in type_D_joints:
            if joint_type_info[joint]["type"] == "C":
                raise ValueError("Type D joint of og_type = C, Not yet implemented! ")
            # Save to joint_type_info
            update_info[joint] = {
                "og_type": joint_type_info[joint]["type"],
                "type": "D"
            }

    
    return update_info

def process_bounding_boxes(finger_chains, vertices_dict, joint_info, joint_type_info, q_0_dict, main_joints):
    """Process fingertips as Type FT."""
    for chain in finger_chains:
        # Process box_min, box_max of all joints
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
                    
                cos_q = np.cos(q_0_dict[rj])
                sin_q = np.sin(q_0_dict[rj])
                R_z = np.array([
                    [cos_q, -sin_q, 0],
                    [sin_q, cos_q, 0],
                    [0, 0, 1]
                ])  # Shape: (n_spheres, 3, 3)
                T_q[:3,:3] = R_z

                # Perform transform to next joint
                nj = chain[chain.index(rj) + 1]
                xyz = joint_info[nj][0]
                quat = joint_info[nj][1]
                T_nj = tf.quaternion_matrix(quat)
                T_nj[:3, 3] = xyz
                T_parent_to_child = T_parent_to_child @ T_q @ T_nj

                if (len(vertices) > 0) & (nj not in main_joints): # If more than 1 DOF joint 
                    break

            # Save point of next joint in joint frame
            p_to_nj = T_parent_to_child[:3, 3]
            vertices = np.vstack(vertices) 

            # Save y axis as z- axis 
            box_min = np.min(vertices, axis=0)
            box_max = np.max(vertices, axis=0)

            joint_type_info[joint]["nj"] = p_to_nj.tolist() 
            joint_type_info[joint]["ft"] = p_ft_joint.tolist() 
            joint_type_info[joint]["ft_normal"] = ft_normal.tolist() 
            joint_type_info[joint]["box_min"] =  box_min if box_min is not None else None 
            joint_type_info[joint]["box_max"] =  box_max if box_max is not None else None

        # Add fingertip information to joint_type_info 
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

        joint_type_info[fingertip] = {
            "type": "FT",
            "box_min": box_min if box_min is not None else None,
            "box_max": box_max if box_max is not None else None
        }

    return joint_type_info

def spherical_raycasts(mesh_dict, joint_info, kinematic_chains, q_0, num_rays=2000, base_link="base_link", ray_origin = None):
    """
    Perform spherical raycasting from the origin of the sphere frame, intersecting with meshes at max_sphere grasping.
    Returns hits in joint frames with their linked spherical coordinates.

    Parameters:
    - mesh_dict (dict): Dictionary mapping joint names to lists of trimesh.Trimesh objects in their joint frames.
    - joint_info (dict): Dictionary containing transformation information for each joint.
    - kinematic_chains (list of lists): List of kinematic chains defining joint hierarchy.
    - q_0 (dict): Joint angles for the initial configuration.
    - num_rays (int): Number of rays to cast (default: 2000).

    Returns:
    - dict: Dictionary with joint names as keys, each mapped to a list of hits.
            Each hit is a dict with:
                'position': Hit position in the joint's local frame (np.ndarray, shape: (3,)).
                'theta': Azimuthal angle of the ray (float, radians).
                'phi': Polar angle of the ray (float, radians).
    """
    # Get transformation from base to sphere frame
    xyz_sphere = joint_info["sphere_frame"][0]
    quat_sphere = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(quat_sphere)
    T_base_to_sphere[:3, 3] = xyz_sphere
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)

    finger_chains = [ # Finger chains 
        chain for chain in kinematic_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]

    roots = []
    for chain in finger_chains:
        root = chain[1]
        root_xyz = joint_info[root][0]
        roots.append(root_xyz)
    roots = np.array(roots)
    roots_h =  np.concatenate(
        [roots, np.ones((roots.shape[0],1))], 
        axis=1
    )
    roots_in_sphere = (T_sphere_to_base @ roots_h.T).T
    r_max = np.max(roots_in_sphere, axis=0)
    print("roots max", r_max)
    root_max_z = r_max[2]

    # Generate ray directions using Fibonacci sampling
    samples = sample_fibonacci_points(num_rays)
    theta = samples[:, 0]  # Azimuthal angle
    phi = samples[:, 1]    # Polar angle
    directions_h = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
        np.ones(samples.shape[0])
    ], axis=1)

    # directions = (T_base_to_sphere @ directions_h.T).T[:,:3]
    directions = directions_h[:,:3]
    
    print("Samples finger print directions shape", directions.shape)
    

    # Initialize output dictionary
    finger_print_dict = dict()
    if ray_origin == None:
        ray_origins = np.tile([0.0, 0.0, 0.0], (num_rays, 1))
    else:
        ray_origins = np.tile(ray_origin, (num_rays, 1))

    # Process each joint and its meshes
    for joint_name in mesh_dict.keys():
        if joint_name not in joint_info or not mesh_dict[joint_name]:
            continue

        # Compute transformation from base to joint
        T_base_to_joint = compute_transform_to_joint(joint_name, kinematic_chains, joint_info, q_0, base_link)

        T_joint_to_base = np.linalg.inv(T_base_to_joint)
        T_joint_to_sphere = T_joint_to_base @ T_base_to_sphere
        T_sphere_to_joint = np.linalg.inv(T_joint_to_sphere)

        # Transform meshes to sphere frame
        meshes_in_base = []
        for mesh in mesh_dict[joint_name]:
            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(T_sphere_to_joint) # Mesh in base
            meshes_in_base.append(mesh_copy)
        
        # Perform raycasting in sphere frame
        for mesh in meshes_in_base:
            locations, indices , _ = mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=directions,
                multiple_hits=False
            )
            print(f"{len(locations)} finger print hits for {joint_name}")

            if len(locations) > 0:
                # Transform hit positions back to joint frame (from base)
                # Convert to homogeneous coordinates
                loc_h = np.hstack((locations, np.ones((locations.shape[0], 1))))
                # loc_sphere = (T_sphere_to_base @ loc_h.T).T[:, :3]
                loc_sphere = locations[:, :3]

                # Filter points with z < root_max_z
                z_filter = loc_sphere[:, 2] < root_max_z
                filtered_locations = locations[z_filter]
                filtered_indices = indices[z_filter]

                # Recalculate spherical coordinates from actual hit positions in sphere frame
                hit_sphere = filtered_locations
                r = np.linalg.norm(hit_sphere, axis=1)
                theta_hit = np.arctan2(hit_sphere[:, 1], hit_sphere[:, 0])
                phi_hit = np.arccos(hit_sphere[:, 2] / np.maximum(r, 1e-8))

                # Apply inverse transformation
                loc_h = np.hstack((filtered_locations, np.ones((filtered_locations.shape[0], 1))))
                loc_joint = (T_joint_to_sphere @ loc_h.T).T[:, :3]

                # Collect hit information
                # print(finger_print_dict.keys())
                if joint_name in finger_print_dict:

                    # Append new points, theta, and phi to existing values if the key already exists
                    finger_print_dict[joint_name]["points"] = np.vstack((finger_print_dict[joint_name]["points"], loc_joint))
                    finger_print_dict[joint_name]["theta"] = np.concatenate((finger_print_dict[joint_name]["theta"], theta_hit)) #!
                    finger_print_dict[joint_name]["phi"] = np.concatenate((finger_print_dict[joint_name]["phi"], phi_hit)) #!
                    finger_print_dict[joint_name]["indices"] = np.concatenate((finger_print_dict[joint_name]["indices"], filtered_indices))
                else:
                    # Initialize the entry if the key does not exist
                    finger_print_dict[joint_name] = {
                        "points": loc_joint,
                        "theta": theta_hit,
                        "phi": phi_hit,
                        "indices": filtered_indices
                    }
                print(f"Max-Min Theta for {np.max(theta[filtered_indices])} - {np.min(theta[filtered_indices])}")
                print(f"Max-Min Phi for {np.max(phi[filtered_indices])} - {np.min(phi[filtered_indices])}")
            

    return finger_print_dict

def compute_7_point_sample_skeleton(finger_print_dict, joint_info, kin_chains, q):
    """
    Compute a 7-point sample for each finger chain using a six-level split at 1/6, 2/6, 3/6, 4/6, 5/6 positions.
    Also returns a second rich dictionary with inherited theta/phi for visualization.

    Parameters:
    - finger_print_dict (dict): Maps joint names to {'points': array, 'theta': array, 'phi': array}.
    - joint_info (dict): Maps joint names to (xyz, quat, type, axis, lower, upper, length).
    - kin_chains (list): List of kinematic chains (lists of joint names).
    - q (dict): Joint angles (e.g. q_0_dict or q_opened).

    Returns:
    - fp_point_sample (dict): Maps joint names to lists of selected points in their local frames (original behavior).
    - fp_7_sample_dict (dict): Maps joint names to {'points': np.ndarray, 'theta': np.ndarray, 'phi': np.ndarray}
      with theta/phi inherited from the closest fingerprint point used for that sample. Ready for visualize_meshes_and_fingerprints.
    """

    # Initialize output dictionaries
    fp_point_sample = {joint: [] for joint in joint_info.keys()}
    fp_7_sample_dict = {joint: {'points': [], 'theta': [], 'phi': []} for joint in joint_info.keys()}

    # Filter finger chains (exclude palm_normal and sphere_frame)
    finger_chains = [
        chain for chain in kin_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]

    sphere_xyz = joint_info["sphere_frame"][0]
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3, 3] = sphere_xyz
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)

    # === 9-point sample for base_link (with inherited theta/phi) ===
    base_link = kin_chains[0][0]
    if base_link in finger_print_dict and finger_print_dict[base_link].get('points', np.array([])).size > 0:
        orig_points = finger_print_dict[base_link]['points']
        orig_thetas = finger_print_dict[base_link]['theta']
        orig_phis = finger_print_dict[base_link]['phi']

        points_h = np.hstack((orig_points, np.ones((orig_points.shape[0], 1))))
        points_base = (T_sphere_to_base @ points_h.T).T[:, :3]

        radius = joint_info["sphere_frame"][6]
        n_pole_h = np.array([0.0, 0.0, 0.0, 1.0])
        n_pole_base = (T_base_to_sphere @ n_pole_h.T).T[:3]

        # Pole - inherit from closest
        if len(points_base) > 0:
            dists = np.linalg.norm(points_base - n_pole_base, axis=1)
            idx_closest = np.argmin(dists)
            # fp_point_sample[base_link].append(n_pole_base.tolist())
            fp_7_sample_dict[base_link]['points'].append(n_pole_base.tolist())
            fp_7_sample_dict[base_link]['theta'].append(float(orig_thetas[idx_closest]))
            fp_7_sample_dict[base_link]['phi'].append(float(orig_phis[idx_closest]))

        # Statistics in base frame
        min_x, min_y, min_z = np.min(points_base, axis=0)
        max_x, max_y, max_z = np.max(points_base, axis=0)
        avg_x, avg_y, avg_z = np.mean(points_base, axis=0)

        targets_base_h = np.array([
            [min_x, min_y, 0.0, 1.0],
            [min_x, max_y, 0.0, 1.0],
            [max_x, min_y, 0.0, 1.0],
            [max_x, max_y, 0.0, 1.0],
            [min_x, avg_y, 0.0, 1.0],
            [max_x, avg_y, 0.0, 1.0],
            [avg_x, min_y, 0.0, 1.0],
            [avg_x, max_y, 0.0, 1.0]
        ])
        targets_base = targets_base_h[:, :3]
        targets_transformed = (T_base_to_sphere @ targets_base_h.T).T[:, :3]  # keep original behavior

        for i, point in enumerate(targets_transformed.tolist()):
            target_b = targets_base[i]
            dists = np.linalg.norm(points_base - target_b, axis=1)
            idx_closest = np.argmin(dists)
            # fp_point_sample[base_link].append(point)
            fp_7_sample_dict[base_link]['points'].append(point)
            fp_7_sample_dict[base_link]['theta'].append(float(orig_thetas[idx_closest]))
            fp_7_sample_dict[base_link]['phi'].append(float(orig_phis[idx_closest]))

    # === Per-finger 7-point samples ===
    for finger_chain in finger_chains:
        fingertip = finger_chain[-1]
        parent = finger_chain[-2]
        root = finger_chain[1]
        base_link_chain = finger_chain[0]  # not used for 7pt

        root_pos_local = [0.0, 0.0, 0.0]
        fp_point_sample[root].append(root_pos_local)

        # Collect relevant joints that have fingerprint data
        relevant_joints = [
            j for j in finger_chain[1:]
            if (j in finger_print_dict.keys()) and (finger_print_dict.get(j, dict()).get('points', np.array([])).size > 0)
        ]

        if not relevant_joints:
            print(f"No fingerprint points for chain {finger_chain}, skipping 7pt surface samples.")
            fp_7_sample_dict[root]['points'].append(root_pos_local)
            fp_7_sample_dict[root]['theta'].append(0.0)
            fp_7_sample_dict[root]['phi'].append(0.0)
            continue

        # Collect all fingerprint points + theta/phi in fingertip frame
        all_points_fingertip = []
        all_joints = []
        all_thetas = []
        all_phis = []

        for j in relevant_joints:
            T_j_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, j, joint_info, q)
            T_ft_to_j = np.linalg.inv(T_j_to_fingertip)

            points_j = finger_print_dict[j]['points']
            thetas_j = finger_print_dict[j]['theta']
            phis_j = finger_print_dict[j]['phi']

            points_j_h = np.hstack((points_j, np.ones((points_j.shape[0], 1))))
            points_fingertip_h = (T_ft_to_j @ points_j_h.T).T
            points_fingertip = points_fingertip_h[:, :3]

            all_points_fingertip.append(points_fingertip)
            all_joints.extend([j] * points_j.shape[0])
            all_thetas.extend(thetas_j.tolist() if hasattr(thetas_j, 'tolist') else list(thetas_j))
            all_phis.extend(phis_j.tolist() if hasattr(phis_j, 'tolist') else list(phis_j))

        all_points_fingertip = np.vstack(all_points_fingertip)
        all_thetas = np.array(all_thetas)
        all_phis = np.array(all_phis)
        # all_joints stays as list for simplicity

        # Root position in fingertip frame (for v and closest lookup)
        T_root_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, root, joint_info, q)
        T_ft_to_root = np.linalg.inv(T_root_to_fingertip)
        p_root_in_fingertip = T_ft_to_root[:3, 3]
        p_fingertip = np.array([0.0, 0.0, 0.0])

        # Inherit theta/phi for root origin from closest fingerprint point
        distances_root = np.linalg.norm(all_points_fingertip - p_root_in_fingertip, axis=1)
        idx_closest_root = np.argmin(distances_root)
        fp_7_sample_dict[root]['points'].append(root_pos_local)
        fp_7_sample_dict[root]['theta'].append(float(all_thetas[idx_closest_root]))
        fp_7_sample_dict[root]['phi'].append(float(all_phis[idx_closest_root]))

        # 5 intermediate samples (1/6 → 5/6) - inherit from closest
        v = p_root_in_fingertip - p_fingertip
        fractions = [5/6, 4/6, 3/6, 2/6, 1/6]
        for frac in fractions:
            pos = p_fingertip + frac * v
            distances = np.linalg.norm(all_points_fingertip - pos, axis=1)
            idx_closest = np.argmin(distances)
            closest_joint = all_joints[idx_closest]
            theta_c = float(all_thetas[idx_closest])
            phi_c = float(all_phis[idx_closest])

            T_closest_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, closest_joint, joint_info, q)
            pos_h = np.append(pos, 1.0)
            local_pos_h = T_closest_to_fingertip @ pos_h
            local_pos = local_pos_h[:3]

            fp_point_sample[closest_joint].append(local_pos)
            fp_7_sample_dict[closest_joint]['points'].append(local_pos.tolist() if isinstance(local_pos, np.ndarray) else local_pos)
            fp_7_sample_dict[closest_joint]['theta'].append(theta_c)
            fp_7_sample_dict[closest_joint]['phi'].append(phi_c)

        # Fingertip position on parent - inherit from closest to fingertip origin
        xyz_fingertip = joint_info[fingertip][0]
        fp_point_sample[parent].append(xyz_fingertip)

        distances_ft = np.linalg.norm(all_points_fingertip - p_fingertip, axis=1)
        idx_closest_ft = np.argmin(distances_ft)
        fp_7_sample_dict[parent]['points'].append(np.asarray(xyz_fingertip).tolist())
        fp_7_sample_dict[parent]['theta'].append(float(all_thetas[idx_closest_ft]))
        fp_7_sample_dict[parent]['phi'].append(float(all_phis[idx_closest_ft]))

    # Convert lists → numpy arrays in the rich dict (consistent with finger_print_dict)
    for joint_name in fp_7_sample_dict:
        entry = fp_7_sample_dict[joint_name]
        if len(entry['points']) > 0:
            entry['points'] = np.asarray(entry['points'])
            entry['theta'] = np.asarray(entry['theta'])
            entry['phi'] = np.asarray(entry['phi'])
        else:
            entry['points'] = np.empty((0, 3))
            entry['theta'] = np.array([])
            entry['phi'] = np.array([])

    # Summary prints (unchanged)
    print(f"Total of {len(fp_point_sample[base_link])} points for base_link")
    for finger_chain in finger_chains:
        total_points = sum(len(fp_point_sample[j]) for j in finger_chain[1:] if j in fp_point_sample)
        print(f"Total of {total_points} points for finger chain with tip '{finger_chain[-1]}'")

    return fp_point_sample

def compute_axisymmetric_deformed_points(phi_breaks, radius_values, points, base_radius=1.0):
    """
    Compute (x, y, z) coordinates for points on an axisymmetric deformed sphere.
    Parameters:
    - phi_breaks: list or array, sorted phi values including 0 and π.
    - radius_values: list or array, radius offsets at each phi_break.
    - points: 2D array, (n_points, 2), theta and phi.
    - base_radius: float, base radius of the sphere.
    Returns:
    - xyz: 2D array, (n_points, 3), (x, y, z) coordinates.
    """
    theta = points[:, 0]
    phi = points[:, 1]
    phi_breaks = np.array(phi_breaks)
    radius_values = np.array(radius_values)
    idx = np.searchsorted(phi_breaks, phi, side='right') - 1
    idx = np.clip(idx, 0, len(phi_breaks) - 2)
    phi_a = phi_breaks[idx]
    phi_b = phi_breaks[idx + 1]
    offset_a = radius_values[idx]
    offset_b = radius_values[idx + 1]
    weight = np.where(phi_b != phi_a, (phi - phi_a) / (phi_b - phi_a), 0)
    radius_offset = offset_a + weight * (offset_b - offset_a)
    r = base_radius * (radius_offset)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    xyz = np.stack([x, y, z], axis=1)
    return xyz

def sample_fibonacci_points(n_points):
    """ Create a list of n_points using the Fibonacci Sampling Algorithm"""
    indices = np.arange(n_points)
    points = np.zeros((n_points,2))
    phi = np.arccos(1 - 2 * indices / n_points)  # Polar angle from 0 to ~π
    theta = 2 * np.pi * indices / ((1 + np.sqrt(5)) / 2) % (2 * np.pi)  # Azimuthal angle from 0 to 2π
    points[:,0]= theta
    points[:,1]= phi
    points[-1] = np.array([np.pi,0]) # Always include south pole
    return points

def process_anchors(order, anchors):
    """
    Create a new list by mapping each integer in 'order' to the corresponding value in 'anchors',
    with adjustments for 0 and 4 if their neighboring indices (1 and 3 respectively) are missing in 'ints'.
    """
    # Check if 1 is present in ints
    has_1 = 1 in order
    # Check if 3 is present in ints
    has_3 = 3 in order
    # Initialize the result list
    result = []
    # Iterate over each integer in ints
    for i in order:
        # If i is 0 and 1 is not present, use average of anchors[0] and anchors[1]
        if i == 0 and not has_1:
            value = (anchors[0] + anchors[1]) / 2
        # If i is 4 and 3 is not present, use average of anchors[3] and anchors[4]
        elif i == 4 and not has_3:
            value = (anchors[3] + anchors[4]) / 2
        # Otherwise, use the direct value from anchors
        else:
            value = anchors[i]
        # Append the value to the result list
        result.append(value)
    # Return the result list
    return result

def solve_for_B_joint_max_phi(q_0_all, joint_tag, joint_index_dict,
                              joint_points, joint_info, point_mask, joint_type_info, points_phi):
    """
    Compute updated joint angles for a B joint across all spheres by selecting the angle to the point with maximum phi among filtered points.
    If no valid points, close the joint by setting to the limit based on ref_dir.
    Parameters:
    - q_0_all (np.ndarray): Shape (n_spheres, n_joints), current joint angles.
    - joint_tag (str): Name of the B joint to solve for.
    - joint_index_dict (dict): Maps joint names to column indices in q_0_all.
    - joint_points (np.ndarray): Shape (n_points, n_spheres, 3), points in the joint frame (x, y, z).
    - joint_info (dict): General joint information.
    - point_mask: Prefiltered points to follow (n_points, n_spheres)
    - joint_type_info (dict): Type-specific joint information.
    - points_phi (np.ndarray): Shape (n_points,), polar angles in sphere frame.
    Returns:
    - np.ndarray: Shape (n_spheres,), updated joint angles for the B joint.
    """
    # Get the column index of the B joint
    B_joint_idx = joint_index_dict[joint_tag]
    n_points_orig = points_phi.shape[0]
    # Compute arctan2 angles for all points
    angles = np.arctan2(joint_points[:n_points_orig, ..., 1], joint_points[:n_points_orig, ..., 0]) # Shape: (n_points, n_spheres)
    # Initialize result array
    n_spheres = q_0_all.shape[0]
    result = np.zeros(n_spheres)
    # Extract joint limits from joint_info
    lower_limit = joint_info[joint_tag][4]
    upper_limit = joint_info[joint_tag][5]
    # Valid mask (limited to original points)
    valid_mask = point_mask # Shape: (n_points, n_spheres)
    # Masked phi for original points
    masked_phi = np.ma.masked_array(np.tile(points_phi[:, np.newaxis], (1, n_spheres)), mask=~valid_mask)
    # Compute amax to determine valid spheres
    masked_max = np.ma.amax(masked_phi, axis=0)
    valid_spheres = ~masked_max.mask
    # Find indices of max phi per sphere
    max_phi_indices = np.ma.argmax(masked_phi, axis=0) # Shape: (n_spheres,)
    # Gather angles for valid spheres
    sphere_indices = np.arange(n_spheres)
    result[valid_spheres] = angles[max_phi_indices[valid_spheres], sphere_indices[valid_spheres]]
    # Set default for invalid spheres based on ref_dir
    ref_dir = joint_type_info[joint_tag]["ref_dir"]
    if ref_dir[1] > 0:
        default_result = upper_limit
    else:
        default_result = lower_limit
    q_0 = q_0_all[:, B_joint_idx]
    result[~valid_spheres] = default_result - q_0[~valid_spheres]
    # Update joint angles
    new_q = q_0 + result
    # Clip the values in the specified joint column to the [lower, upper] range
    new_q = np.clip(new_q, lower_limit, upper_limit)
    return new_q


def build_lookup_dict_with_fp_and_phi(
        fp_sample,
        joint,
        q_0,
        joint_info,
        kinematic_chains,
        joint_type_info,
        main_joints,
        chain_idx,
        merge_anchor = None,
        anchor_override = None,
        realign_sphere = False,
        resolution=0.005,
        N=200,
        base_link="base_link",
        meshes_dict = None):
    """Build a lookup dictionary for a Type A joint including FP and phi lookups."""

    lower, upper = joint_info[joint][4], joint_info[joint][5]
    # print(joint)
    print("Building Lookup Dicts for ", joint)

    # Identify the finger chain for this joint
    chain = next((c for c in kinematic_chains if joint in c), None)
    if chain is None:
        raise ValueError(f"No kinematic chain found for joint {joint}")
    fingertip = chain[-1]
    parent = chain[-2]
    root = chain[1]

    # Sphere transform
    xyz_sphere = joint_info["sphere_frame"][0]
    quat_sphere = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(quat_sphere)
    T_base_to_sphere[:3, 3] = xyz_sphere
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)

    # Compute fixed position of root joint in sphere frame using initial configuration
    T_base_to_root = compute_transform_to_joint(root, kinematic_chains, joint_info, q_0, base_link=base_link)
    p_base_root = T_base_to_root[:3, 3]
    p_sphere_root = (T_sphere_to_base @ np.append(p_base_root, 1))[:3]
    r_root = np.linalg.norm(p_sphere_root)
    theta_root = np.arctan2(p_sphere_root[1], p_sphere_root[0])

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
    p = sample_fibonacci_points(2500)
    theta, phi = p[:, 0], p[:, 1]
    radius = joint_info["sphere_frame"][6]  # Assuming radius is the 7th element
    phi_breaks = [0, np.pi/2, 3 * np.pi / 4, np.pi]
    # radius_values = [0, 0, np.sqrt(2), 2]
    radius_values = [1.0, 1.0, 1.0, 1.0]
    points = compute_axisymmetric_deformed_points(phi_breaks, radius_values, p, base_radius=radius)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    points_h_t = points_h.T  # (4, n_points)
    print("points", points_h_t.shape)
        

    if meshes_dict:
        # Create finger_print_dict
        viz_finger_print_dict = {
            "sphere_frame": {
                "points": points,
                "theta": theta,
                "phi": phi
            }
        }
    
    # Compute the radial distance for each point
    r = np.sqrt(x**2 + y**2 + z**2)  # radial distance, shape (n_points,)

    # Compute phi (polar angle) for each point
    points_phi = np.arccos(z / r)  # phi in radians, shape (n_points,)

    # Combined q_main_joint from lower to upper
    q_main_joint = np.linspace(lower, upper, N + 1)
    q_all_joints = np.tile(base_q_array, (N + 1, 1))
    q_all_joints[:, main_idx] = q_main_joint
    # print("q", q_all_joints)

    # Solve IK for all B joints across all q_all_joints
    for i in range(len(chain) - 2):
        j = chain[i + 1]
        if (joint_type_info[j]["type"] == "B") | (joint_type_info[j]["type"] == "D"):
            if ("og_type" in joint_type_info[j].keys()) :
                if joint_type_info[j]["og_type"] == "A":
                    continue
            T_sphere_to_j = compute_sphere_to_single_joint_transforms(j, q_all_joints, joint_index_dict, joint_info, chain)
            T_j_to_sphere = np.linalg.inv(T_sphere_to_j)
            transformed_points_h = np.matmul(T_j_to_sphere, points_h_t).transpose(2, 0, 1)  # (n_points, n_spheres, 4)
            
            # Extract the position of the joint origin in the sphere frame
            joint_pos_in_sphere = T_sphere_to_j[:, :3, 3]  # Shape: (n_spheres, 3)
            r_joint = np.linalg.norm(joint_pos_in_sphere, axis=1)  # Shape: (n_spheres,)
            joint_phi = np.arccos(joint_pos_in_sphere[:, 2] / r_joint)  # Shape: (n_spheres,)

            joint_points = transformed_points_h[:, :, :3]
            # print("transformed_points_h", transformed_points_h.shape)
            centroids = np.mean(joint_points, axis=0)
            # print("centroids", centroids.shape)
            # print("joint_points", joint_points.shape)
            box_min = np.array(joint_type_info[j]["box_min"])
            box_max = np.array(joint_type_info[j]["box_max"])
            z_values = joint_points[:, :, 2]
            # z_mask = (z_values >= box_min[2]) & (z_values <= box_max[2])
            z_mask = np.abs(z_values)<= 0.10 * radius
            # Add phi mask
            l = np.linalg.norm(joint_type_info[j]["nj"][:2])
            points_norm = np.linalg.norm(joint_points[:,:,:3], axis = 2)
            x_values = joint_points[:, :, 0] 
            if j == root:
                mask = (points_phi[:, np.newaxis] >= joint_phi[np.newaxis, :] )
                # range_mask = (mask & (x_values >= l))
                range_mask = (mask & (points_norm >= l))
            elif j == chain[-2]:
                range_mask = (points_norm >= l)
            else:
                # range_mask = ((x_values > 0.0) & (x_values < l)) 
                range_mask = (points_norm >= l) 
            point_mask = z_mask & range_mask
            # point_mask = (z_values >= -1000) 
            l_ft = np.full(N + 1, l) 
            # print("point_mask", point_mask.shape)
            # print("l_ft", l_ft.shape)
            # new_q = solve_for_B_joint(q_all_joints, j, joint_index_dict, centroids, joint_points, joint_info, point_mask, l_ft, joint_type_info)
            new_q = solve_for_B_joint(q_all_joints, j, joint_index_dict, centroids, joint_points, joint_info, point_mask, l, joint_type_info)
            # new_q = solve_for_B_joint_max_phi(
            #     q_all_joints,
            #     j, 
            #     joint_index_dict,
            #     joint_points, 
            #     joint_info, 
            #     point_mask, 
            #     joint_type_info, 
            #     points_phi
            #     )
            q_all_joints[:, joint_index_dict[j]] = new_q   

    # Compute p_ft_sphere_all
    T_sphere_to_ft_all = compute_sphere_to_single_joint_transforms(fingertip, q_all_joints, joint_index_dict, joint_info, chain)
    p_ft_sphere_all = T_sphere_to_ft_all[:, :3, 3]
    r_all = np.linalg.norm(p_ft_sphere_all, axis=1)
    phi_all = np.arccos(p_ft_sphere_all[:, 2] / r_all)

    # Compute theta_all
    theta_all = np.arctan2(p_ft_sphere_all[:, 1]/ r_all, p_ft_sphere_all[:, 0]/ r_all)

    # Create mask
    # phi_mask =  phi_all >= np.deg2rad(120) #! Was pi/2
    phi_mask =  phi_all >= np.deg2rad(45) #! Was pi/2
    valid_mask = phi_mask 

    # Find q_mid_idx
    q_mid = q_0[joint]
    q_mid_idx = np.argmin(np.abs(q_main_joint - q_mid))

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
    valid_phi = phi_all[valid_idx]
    
    if not valid_q_main_joint.size:
        raise ValueError(f"No valid theta values for joint {joint}")

    # Compute anchor as center of theta range
    min_theta = np.min(valid_theta)
    max_theta = np.max(valid_theta)
    min_q = valid_q_main_joint[np.argmin(valid_theta)]
    max_q = valid_q_main_joint[np.argmax(valid_theta)]

    # Find index closest to anchor
    if anchor_override is None:
        mean_q = (max_q + min_q) / 2
        mid_idx = np.argmin(np.abs(valid_q_main_joint - mean_q))
        idx_anchor = mid_idx
        anchor = valid_theta[mid_idx]

        # Mid theta
        anchor = (max_theta + min_theta)/2
        print("max_theta",max_theta)
        print("min_theta",min_theta)

    else:
        # anchor = np.mean(anchor_override)
        anchor = anchor_override[np.argmin(np.abs(anchor_override))]
        mid_idx = np.argmin(np.abs(valid_theta - anchor))
    if realign_sphere:
        anchor = min_theta + (max_theta - min_theta) / 2

    q_anchor = valid_q_main_joint[mid_idx]
    idx_anchor = np.argmin(np.abs(valid_theta - anchor))
    min_anchor_distance = valid_theta[idx_anchor] - anchor
    
    if realign_sphere:
        print("min_anchor_distance", min_anchor_distance)
        print("theta min & max", min_theta, max_theta)

        angle = anchor
        c = np.cos(angle)
        s = np.sin(angle)
        T = np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        T_base_to_sphere = np.matmul(T_base_to_sphere, T)
        T_sphere_to_base = np.linalg.inv(T_base_to_sphere)
        new_quat = tf.quaternion_from_matrix(T_base_to_sphere)
        
        # Recalculate affected values with the updated sphere transformation
        T = np.eye(4)
        joint_info["sphere_frame"] = (
            xyz_sphere,  # Translation
            new_quat,  # Quaternion
            "fixed",  # Type (fixed frame)
            None,  # Axis (not applicable)
            None,  # Lower limit
            None,  # Upper limit
            radius  # Radius in length field
        )
        print(f"Realligned Sphere to middle finger anchor{anchor}. Joint: {joint}")
        print()
        
        return
    
    # Place valid closest to open palm position #!
    if 4 not in chain_idx:
        mean_q = (max_q + min_q) / 2
        # idx_anchor = np.argmin(np.abs(valid_q_main_joint - q_0[joint]))
        mid_idx = np.argmin(np.abs(valid_q_main_joint - mean_q))
        idx_anchor = mid_idx
        q_anchor = valid_q_main_joint[idx_anchor]
        anchor = valid_theta[idx_anchor] 
        # min_anchor_distance = 0.0 # New anchor will be were it is
        
        # if anchor_override.shape[0] > 1:
        #     for i in range(anchor_override.shape[0]):
        #         anchor_override[i]= anchor
    min_anchor_distance = valid_theta[idx_anchor] - anchor

    print("anchor", anchor)
    print("q_anchor", q_anchor)
    print("idx_anchor", idx_anchor)

    if merge_anchor:
        merge_anchor_idx = np.argmin(np.abs(valid_theta - (merge_anchor + anchor)/2))
        q_merge = valid_q_all_joints[merge_anchor_idx]
        merge_offset = valid_theta[merge_anchor_idx] - anchor
        q_merge_dict = {}
        for j in chain[1:-1]:
            q_merge_dict[j] = q_merge[joint_index_dict[j]]

    if anchor_override is not None:
        if anchor_override.shape[0] > 1:
            anchor_0_idx = np.argmin(np.abs(valid_theta - anchor_override[0]))
            q_anchor_0 = valid_q_all_joints[anchor_0_idx]
            anchor_0_offset = valid_theta[anchor_0_idx] - anchor
            q_anchor_0_dict = {}
            for j in chain[1:-1]:
                q_anchor_0_dict[j] = q_anchor_0[joint_index_dict[j]]

            anchor_1_idx = np.argmin(np.abs(valid_theta - anchor_override[1]))
            q_anchor_1 = valid_q_all_joints[anchor_1_idx]
            anchor_1_offset = valid_theta[anchor_1_idx] - anchor
            q_anchor_1_dict = {}
            for j in chain[1:-1]:
                q_anchor_1_dict[j] = q_anchor_1[joint_index_dict[j]]

    # Compute theta offsets from anchor and wrap to [-pi, pi]
    theta_offsets = valid_theta - anchor
    theta_offsets = (theta_offsets + np.pi) % (2 * np.pi) - np.pi

    # Find indices for min and max q values
    min_idx = np.argmin(valid_q_main_joint)
    max_idx = np.argmax(valid_q_main_joint)

    # Find actual min and max offsets
    min_offset = theta_offsets[min_idx]
    max_offset = theta_offsets[max_idx]

    # Find index of theta_offset closest to 0 (closest to anchor)
    idx_zero_theta = np.argmin(np.abs(theta_offsets))
    zero_q = valid_q_main_joint[idx_zero_theta]  # q value closest to the anchor
    print("offsets ", min_offset, max_offset)

    # Generate offset steps (full range)
    if min_offset < max_offset:
        offsets = np.arange(min_offset, max_offset + resolution / 2, resolution)
    else: 
        offsets = np.arange(max_offset, min_offset + resolution / 2, resolution)
        min_offset = theta_offsets[max_idx]
        max_offset = theta_offsets[min_idx]

    # Find zero_idx in offsets (index where offset is closest to 0)
    zero_idx = np.argmin(np.abs(offsets))

    # Get min and max q values
    q_min = valid_q_main_joint[min_idx]
    q_max = valid_q_main_joint[max_idx]

    # Determine the overall trend (slope of linear fit)
    p = np.polyfit(theta_offsets, valid_q_main_joint, 1)
    slope = p[0]

    # Initialize q_list
    q_list = np.zeros(len(offsets))

    # Split into two segments: min_offset to zero_offset and zero_offset to max_offset
    if slope > 0:
        # Ascending: q_min -> zero_q -> q_max
        q_list[:zero_idx + 1] = np.linspace(q_min, zero_q, zero_idx + 1)
        q_list[zero_idx:] = np.linspace(zero_q, q_max, len(offsets) - zero_idx)
    else:
        # Descending: q_max -> zero_q -> q_min
        q_list[:zero_idx + 1] = np.linspace(q_max, zero_q, zero_idx + 1)
        q_list[zero_idx:] = np.linspace(zero_q, q_min, len(offsets) - zero_idx)

    # Convert q_list to numpy array (already is, but explicit)
    q_list = np.array(q_list)
    # print("theta offsets", offsets)
    print("q min & max", q_min, q_max)
    print("theta at q min & max", min_offset + anchor, max_offset + anchor)

    new_fp_sample = {j: [] for j in chain}

    # Use reference configuration at anchor for computations
    q_ref_all = valid_q_all_joints[idx_anchor]
    q = {joint_list[i]: q_ref_all[i] for i in range(len(joint_list))}

    # Compute fingertip position and theta at anchor
    T_base_to_ft = compute_transform_to_joint(fingertip, kinematic_chains, joint_info, q, base_link=base_link)

    p_ft_base = T_base_to_ft[:3, 3]
    p_ft_sphere = (T_sphere_to_base @ np.append(p_ft_base, 1))[:3]

    # Project input fp_sample points to the perfect sphere surface (using phi_fp, theta_fp, r=radius)
    for joint_s in chain[1:]:
        if joint_s not in fp_sample or not fp_sample[joint_s]:
            continue
        T_base_to_joint_s = compute_transform_to_joint(joint_s, kinematic_chains, joint_info, q, base_link=base_link)
        T_joint_to_base = np.linalg.inv(T_base_to_joint_s)

        for local_p in fp_sample[joint_s]:
            p_base = (T_base_to_joint_s @ np.append(local_p, 1))[:3]
            p_sphere = (T_sphere_to_base @ np.append(p_base, 1))[:3]
            r_current = np.linalg.norm(p_sphere)
            phi = np.arccos(p_sphere[2] / r_current) if r_current > 0 else 0.0
            theta = np.arctan2(p_sphere[1], p_sphere[0])
            new_p_sphere = radius * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            p_base_new = (T_base_to_sphere @ np.append(new_p_sphere, 1))[:3]
            p_local_new = (T_joint_to_base @ np.append(p_base_new, 1))[:3]
            new_fp_sample[joint_s].append(p_local_new)
    
    # Compute joint phis at anchor configuration
    chain_joints = chain[1:-1]
    joint_phis = []

    for j in chain_joints:
        T_base_to_j = compute_transform_to_joint(j, kinematic_chains, joint_info, q, base_link=base_link)
        p_base = T_base_to_j[:3, 3]
        p_sphere = (T_sphere_to_base @ np.append(p_base, 1))[:3]
        r = np.linalg.norm(p_sphere)
        phi_j = np.arccos(p_sphere[2] / r) if r > 0 else 0.0
        joint_phis.append(phi_j)

    # Build joints_phi dict
    joints_phi = {chain_joints[i]: joint_phis[i] for i in range(len(chain_joints))}

    # Add fingertip phi
    r_ft = np.linalg.norm(p_ft_sphere)
    phi_ft = np.arccos(p_ft_sphere[2] / r_ft) if r_ft > 0 else 0.0
    joints_phi[chain[-1]] = phi_ft


    ft_in_sphere = p_ft_sphere
    T_base_to_joint = compute_transform_to_joint(joint, kinematic_chains, joint_info, q, base_link=base_link)
    ft_in_joint = (np.linalg.inv(T_sphere_to_base @ T_base_to_joint) @ np.append(ft_in_sphere, 1))[:3] # in m
    print()
    print(" FT INFO: ")

    print(ft_in_sphere, ft_in_joint)
    print("Phi:",  phi_ft)
    # Compute next_joint_dict using phis at anchor
    next_joint_dict = {}
    for j in chain_joints:
        j_idx = chain.index(j)
        next_joint = chain[j_idx + 1]
        next_joint_dict[j] = next_joint
        nj_i = j_idx + 1
        if joint_type_info[j]["type"] == "D": #!
            # nj_i = j_idx + 1
            next_joint = chain[nj_i]

        while ((next_joint != chain[-1]) & ((abs(joints_phi[j] - joints_phi[next_joint]) < 0.0872665) | (next_joint in main_joints))): #!
            nj_i += 1
            next_joint = chain[nj_i]
        next_joint_dict[j] = next_joint

    # Save  sphere control
    sphere_control_dict = {
        joint: {
            "next_joint": next_joint_dict,
            "joint_phis": joints_phi,
            }
    }
    
    chain_order = chain[1:1]
    q_chains = np.zeros(len(chain_order))

    # Save Main Joint Lookup Dictionary
    a_dict = { 
        "type": "A",
        "anchor": anchor,
        "q_anchor": q_anchor,
        "resolution": resolution,
        "upper": max_offset,
        "lower": min_offset,
        "q_min": q_min,
        "q_max": q_max,
        "q_list": q_list,
        "zero_idx": zero_idx,
        "anchor_dist": min_anchor_distance,
        "ft_sphere": ft_in_sphere,
        "ft_joint": ft_in_joint,
        "og_type": joint_type_info[joint]["type"],
    }
    if merge_anchor:
        a_dict["q_merge_dict"] = q_merge_dict,
        a_dict["merge_offset"] = merge_offset
        print("Joint values for Merge", q_merge_dict)
    if anchor_override is not None:
        if anchor_override.shape[0] > 1:
            a_dict["q_anchor_0_dict"] = q_anchor_0_dict,
            a_dict["anchor_0_offset"] = anchor_0_offset
            print("Joint values for anchor_0", q_anchor_0_dict)
            print("Offset values for anchor_0", anchor_0_offset)
            a_dict["q_anchor_1_dict"] = q_anchor_1_dict,
            a_dict["anchor_1_offset"] = anchor_1_offset
            print("Joint values for anchor_1", q_anchor_1_dict)
            print("Offset values for anchor_1", anchor_1_offset)

    print()
    print()
    joint_type_info[joint].update(a_dict)
    
    # Get the full q_ref_all for this index 
    q_ref_all = valid_q_all_joints[idx_anchor]
    q_ref_dict = {joint_list[i]: q_ref_all[i] for i in range(len(joint_list))}

    # Save max phi q
    max_phi_idx = np.argmax(valid_theta)
    q_max_phi = valid_q_all_joints[max_phi_idx]
    
    # max_phi_idx = np.argmin(np.abs(valid_q_main_joint - q_0[joint]))
    # q_max_phi = valid_q_all_joints[max_phi_idx]
    q_max_phi_dict = {joint_list[i]: q_max_phi[i] for i in range(len(joint_list))}

    # Visualize min_offset, anchor, and max_offset points
    # Find indices for min_offset, anchor, and max_offset in valid_theta
    min_offset_idx = np.argmin(np.abs(theta_offsets - min_offset))
    max_offset_idx = np.argmin(np.abs(theta_offsets - max_offset))
    anchor_idx = np.argmin(np.abs(valid_theta - anchor))  # Already computed as idx_anchor

    # Call visualize_meshes_and_fingerprints
    if meshes_dict:
        # Get corresponding q values and compute fingertip positions
        offset_indices = [min_offset_idx, anchor_idx, max_offset_idx]
        offset_labels = ['min_offset', 'anchor', 'max_offset']
        offsets = [min_offset, 0.0, max_offset]


        for idx in offset_indices:
            q_config = valid_q_all_joints[idx]
            q_dict = {joint_list[i]: q_config[i] for i in range(len(joint_list))}
            # thetas= [viz_finger_print_dict["sphere_frame"]["theta"].copy() - offsets[c] for c in rnage]

        v_dict = viz_finger_print_dict.copy()
        c=0
        import copy
        for idx in offset_indices:
            q_config = valid_q_all_joints[idx]
            q_dict = {joint_list[i]: q_config[i] for i in range(len(joint_list))}
            # Deep copy the original fingerprint dict to avoid mutation across iterations
            v_dict = copy.deepcopy(viz_finger_print_dict)
            v_dict["sphere_frame"]["theta"] = v_dict["sphere_frame"]["theta"] - offsets[c]
            visualize_meshes_and_fingerprints_screenshot(
                meshes_dict=meshes_dict,  # Assuming meshes_dict is available; replace with your actual meshes_dict
                joint_info=joint_info,
                kinematic_chains=kinematic_chains,
                finger_print_dict=v_dict,
                q=q_dict,
                title=" ",
                base_link=base_link,
                camera_position=(0.5, 0.5, 0.5),  # Camera at (10, 10, 10)
                focal_point=(0, 0, 0),        # Looking at origin
                view_up=(0, 0, 1),            # Up direction along z-axis
                screenshot_path = f"/home/felipe/Downloads/{joint}_{c}.png"
            )
            c+=1

    return new_fp_sample, sphere_control_dict, q_ref_dict, q_max_phi_dict, min_offset, max_offset

def compute_transform_to_joint(target_joint, kinematic_chains, joint_info, q=None, base_link="base_link"):
    """
    Compute the 4x4 transformation matrix from base_link to the target joint, considering joint angles in q.

    Parameters:
    - target_joint (str): The joint to compute the transformation to.
    - kinematic_chains (list of lists): Defines the hierarchy of joints.
    - joint_info (dict): Maps joint names to (xyz, quat, joint_type, axis, lower, upper, length).
    - q (dict, optional): Joint angles for revolute joints; if None, assumes zero configuration.

    Returns:
    - np.ndarray: 4x4 transformation matrix from base_link to target_joint.
    """
    # Find the chain containing the target_joint
    for chain in kinematic_chains:
        if target_joint in chain:
            break
    else:
        raise ValueError(f"Target joint '{target_joint}' not found in any kinematic chain")

    # Build parent map for the chain
    parent_map = {chain[i]: chain[i - 1] for i in range(1, len(chain))}
    # Collect transformations from base_link to target_joint
    transforms = []
    current_joint = target_joint
    while current_joint != base_link:
        xyz = joint_info[current_joint][0]
        quat = joint_info[current_joint][1]
        joint_type = joint_info[current_joint][2]
        T = tf.quaternion_matrix(quat)
        T[:3, 3] = xyz
        if q is not None and joint_type == "revolute":
            new_q = q.copy()
            q_val = new_q.get(current_joint, 0.0)
            R_z = tf.rotation_matrix(q_val, [0, 0, 1])  # Assuming z-axis rotation
            T = T @ R_z
        transforms.append(T)
        current_joint = parent_map[current_joint]

    # Multiply transformations in reverse order
    T_base_to_joint = np.eye(4)
    for T in reversed(transforms):
        T_base_to_joint = T_base_to_joint @ T
    return T_base_to_joint

def get_q_A_from_offset(theta_offset, joint, lookup_dicts):
    """Retrieve q_A for a given theta offset and finger index."""
    lookup = lookup_dicts[joint]
    if lookup is None:
        raise ValueError(f"No lookup dictionary for type A joint {joint}")
    
    resolution = lookup["resolution"]
    zero_idx = lookup["zero_idx"]
    steps = int(theta_offset / resolution) + zero_idx
    steps = np.clip(steps, 0, len(lookup["q_list"])-1)
    
    return lookup["q_list"][steps]
    

def y_axis_palm_rotation(joint_info, kinematic_chains, q_max, base_link="base_link"):
    """
    Rotates the palm_normal frame around its y-axis so that its z-axis points toward 
    the xz-midpoint of all fingertips in the current palm_normal frame.
    
    Parameters:
    - joint_info (dict): Maps joint names to (xyz, quat, joint_type, axis, lower, upper, length).
    - kinematic_chains (list of lists): Defines the hierarchy of joints from base_link to end effectors.
    - q_max (dict): Joint angles at maximum reach configuration.
    - base_link (str): Name of the base link frame.
    
    Returns:
    - None: Updates joint_info["palm_normal"] in place.
    """
    # Find palm_normal chain
    palm_chain = next((chain for chain in kinematic_chains if "palm_normal" in chain), None)
    if palm_chain is None:
        raise ValueError("No chain containing 'palm_normal' found in kinematic_chains")
    
    palm_xyz_init = np.array(joint_info["palm_normal"][0])
    palm_quat_init = joint_info["palm_normal"][1]
    T_base_to_palm_init = tf.quaternion_matrix(palm_quat_init)
    T_base_to_palm_init[:3, 3] = palm_xyz_init

    sphere_xyz = np.array(joint_info["sphere_frame"][0])
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3, 3] = sphere_xyz

    T_palm_to_sphere_init = np.linalg.inv(T_base_to_palm_init) @ T_base_to_sphere
    
    # Identify finger chains and their fingertips
    finger_chains = [chain for chain in kinematic_chains if "palm_normal" not in chain and "sphere_frame" not in chain]
    if not finger_chains:
        raise ValueError("No finger chains found")
    
    fingertips = [chain[-1] for chain in finger_chains]
    
    # Get current palm_normal transformation
    palm_xyz = np.array(joint_info["palm_normal"][0])
    palm_quat = joint_info["palm_normal"][1]
    T_base_to_palm = tf.quaternion_matrix(palm_quat)
    T_base_to_palm[:3, 3] = palm_xyz
    T_palm_to_base = np.linalg.inv(T_base_to_palm)
    
    # Compute fingertip positions in base frame at q_max
    fingertip_positions_base = []
    for fingertip in fingertips:
        T_base_to_tip = compute_transform_to_joint(
            fingertip, kinematic_chains, joint_info, q_max, base_link
        )
        pos_base = T_base_to_tip[:3, 3]
        fingertip_positions_base.append(pos_base)
    fingertip_positions_base = np.array(fingertip_positions_base)
    
    # Transform fingertip positions to current palm_normal frame
    fingertip_positions_palm = []
    for pos_base in fingertip_positions_base:
        pos_h = np.append(pos_base, 1)
        pos_palm_h = T_palm_to_base @ pos_h
        fingertip_positions_palm.append(pos_palm_h[:3])
    fingertip_positions_palm = np.array(fingertip_positions_palm)
    
    # Zero out y-coordinates and compute xz-midpoint
    fingertip_xz = fingertip_positions_palm.copy()
    fingertip_xz[:, 1] = 0  # Project to xz-plane
    min_pos = np.min(fingertip_xz, axis=0)
    max_pos = np.max(fingertip_xz, axis=0)
    xz_midpoint = (min_pos + max_pos) / 2
    
    # Desired z-axis direction: from palm origin to xz_midpoint (in palm frame)
    desired_z = xz_midpoint 
    if np.linalg.norm(desired_z) < 1e-8:
        raise ValueError("Error: Fingertip xz-midpoint is at palm origin. No rotation applied.")
    
    desired_z = desired_z / np.linalg.norm(desired_z)
    
    # Current palm z-axis
    current_z = np.array([0, 0, 1])
    
    # Compute rotation around y-axis to align current z with desired z
    # Rotation axis is y = [0,1,0], but we compute angle between current_z and desired_z
    cos_angle = np.dot(current_z, desired_z)
    if abs(cos_angle) > 0.9999:
        rotation_angle = 0.0
    else:
        # Cross product gives rotation axis (should be along y or -y)
        cross = np.cross(current_z, desired_z)
        rotation_angle = np.arcsin(cross[1])  # Since axis is y, sin(angle) = cross[1]
        # Correct sign using dot product
        if cos_angle < 0:
            rotation_angle = np.pi - rotation_angle if rotation_angle > 0 else -np.pi - rotation_angle
    
    # Build rotation matrix around y-axis
    R_y = tf.rotation_matrix(rotation_angle, [0, 1, 0])[:3, :3]
    
    # Apply rotation to current palm rotation
    R_palm_current = T_base_to_palm[:3, :3]
    R_palm_new = R_palm_current @ R_y
    
    # Construct new transformation matrix
    T_base_to_palm_new = np.eye(4)
    T_base_to_palm_new[:3, :3] = R_palm_new
    T_base_to_palm_new[:3, 3] = palm_xyz  # Keep position unchanged
    
    # Update joint_info
    quat_palm_new = tf.quaternion_from_matrix(T_base_to_palm_new)
    joint_info["palm_normal"] = (
        palm_xyz.tolist(),           # xyz (unchanged)
        quat_palm_new,               # updated quaternion
        "fixed",                     # type
        None,                        # axis
        None,                        # lower
        None,                        # upper
        None                         # length
    )
    
    print(f"y_axis_palm_rotation: Applied {np.rad2deg(rotation_angle):.2f}° rotation around y-axis")
    print(f"   xz-midpoint in palm frame: {xz_midpoint}")
    print(f"   New z-axis direction: {R_palm_new @ np.array([0,0,1])}")
    print()


def compute_n_point_sample_skeleton(num_points_per_finger, num_fingertip_extra, finger_print_dict, joint_info, kin_chains, q, use_x_axis = True):
    """
    Compute an n-point sample skeleton for each finger chain by tracing equally spaced points along the kinematic chain from root to fingertip, with extra points extending the fingertip.
    Parameters:
    - num_points_per_finger (int): Number of points along the main finger chain.
    - num_fingertip_extra (int): Number of extra points to add beyond the fingertip.
    - finger_print_dict (dict): Maps joint names to {'points': array, 'theta': array, 'phi': array}.
    - joint_info (dict): Maps joint names to (xyz, quat, type, axis, lower, upper, length).
    - kin_chains (list): List of kinematic chains (lists of joint names).
    - q (dict): Joint angles.
    - use_x_axis (bool): If True, extend extra points along fingertip's x-axis; else, along the line from parent to fingertip.
    Returns:
    - dict: Maps joint names to lists of selected points in their local frames.
    - dict: Maps joint names to lists of phi values corresponding to the points.
    """
    # Initialize output dictionaries
    skeleton_point_dict = {joint: [] for joint in joint_info.keys()}
    skeleton_phi_dict = {joint: [] for joint in joint_info.keys()}
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
    # Add points for base_link
    base_link = kin_chains[0][0] # base_link is the first joint in all chains
    if base_link in finger_print_dict and finger_print_dict[base_link].get('points', np.array([])).size > 0:
        radius = joint_info["sphere_frame"][6]
        n_pole_base = (T_base_to_sphere @ np.array([0.0, 0.0, radius, 1.0])).T[:3]
        skeleton_point_dict[base_link].append(n_pole_base.tolist()) # Palm center point
        n_pole_sphere = (T_sphere_to_base @ np.append(n_pole_base, 1.0))[:3]
        skeleton_phi_dict[base_link].append(np.arctan2(n_pole_sphere[1], n_pole_sphere[0]))
        # Compute points along lines from n_pole to each finger root
        lines = []
        for finger_chain in finger_chains:
            root = finger_chain[1]
            root_pos = joint_info[root][0]
            root_quat = joint_info[root][1]
            T_base_to_root = tf.quaternion_matrix(root_quat)
            T_base_to_root[:3,3] = root_pos
            v = root_pos - n_pole_base
            L = np.linalg.norm(v)
            if L > 1e-6:
                lines.append((root_pos, L, v))
        total_length = sum(L for _, L, _ in lines)
        total_points = 10 * len(finger_chains)
        remaining_points = total_points - 1  # Excluding n_pole
        for root_pos, L, v in lines:
            if total_length > 0:
                num_add = round((L / total_length) * remaining_points)
            else:
                num_add = 0
            if num_add > 0:
                for k in range(1, num_add + 1):
                    frac = k / num_add
                    pos_base = n_pole_base + frac * v
                    skeleton_point_dict[base_link].append(pos_base.tolist())
                    pos_sphere = (T_sphere_to_base @ np.append(pos_base, 1.0))[:3]
                    skeleton_phi_dict[base_link].append(np.arctan2(pos_sphere[1], pos_sphere[0]))
    
    for finger_chain in finger_chains:
        fingertip = finger_chain[-1] # Last joint in the chain
        parent = finger_chain[-2] # Parent of fingertip
        root = finger_chain[1] # First joint after base_link
        joint_list = finger_chain[1:] # root to fingertip
        # Compute transforms and positions in root frame
        root_pos = joint_info[root][0]
        root_quat = joint_info[root][1]
        T_base_to_root = tf.quaternion_matrix(root_quat)
        T_base_to_root[:3,3] = root_pos
        T_joints = {}
        T_cumulative = tf.rotation_matrix(q[root], [0, 0, 1])  # Rotation around z-axis (common convention)

        positions = {}
        positions[root] = np.array([0.0, 0.0, 0.0])
        # print("joint_pos")
        for j in joint_list[1:]:
            j_pos = joint_info[j][0]
            j_quat = joint_info[j][1]
            T_to_j = tf.quaternion_matrix(j_quat)
            T_to_j[:3,3] = j_pos
            T_cumulative = T_cumulative @ T_to_j
            T_root_to_j = T_cumulative
            positions[j] = T_root_to_j[:3, 3]
            # print(j, T_to_j)
            # print(j, j_pos)
            if j != finger_chain[-1]:
                T_z = tf.rotation_matrix(q[j], [0, 0, 1])  # Rotation around z-axis (common convention)
                T_cumulative = T_cumulative @ T_z
                T_joints[j] = T_cumulative.copy()
        
        T_root_to_fingertip = T_cumulative

        pos_list = [positions[j] for j in joint_list]
        # Compute cumulative lengths along chain
        cum_len = [0.0]
        for i in range(1, len(pos_list)):
            dist = np.linalg.norm(pos_list[i] - pos_list[i - 1])
            cum_len.append(cum_len[-1] + dist)
        L_main = cum_len[-1]
        num_main = num_points_per_finger
        d = L_main / (num_main - 1) if num_main > 1 and L_main > 0 else 0.0
        # Sample equally spaced points along main chain
        for k in range(num_main):
            target = k * d
            # Handle floating point precision for last point
            if target > L_main - 1e-6:
                target = L_main
            for s in range(len(cum_len) - 1):
                if cum_len[s] <= target <= cum_len[s + 1]:
                    seg_len = cum_len[s + 1] - cum_len[s]
                    frac = (target - cum_len[s]) / seg_len if seg_len > 0 else 0.0
                    p_root = pos_list[s] + frac * (pos_list[s + 1] - pos_list[s])
                    key = joint_list[s]
                    # Compute T_root_to_key
                    if key == root:
                        T_root_to_key = tf.rotation_matrix(q[root], [0, 0, 1])
                    else:
                        T_root_to_key = T_joints[key]
                    p_root_h = np.append(p_root, 1.0)
                    local_h = np.linalg.inv(T_root_to_key) @ p_root_h
                    local = local_h[:3].tolist()
                    skeleton_point_dict[key].append(local)
                    # Compute position in base frame
                    p_base_h = T_base_to_root @ T_root_to_key @ local_h
                    p_base = p_base_h[:3]
                    # Compute position in sphere frame
                    p_sphere_h = T_sphere_to_base @ T_base_to_root @ T_root_to_key @ local_h
                    p_sphere = p_sphere_h[:3]
                    phi = np.arccos(p_sphere[2] / np.linalg.norm(p_sphere))
                    skeleton_phi_dict[key].append(phi)
                    # print("phis", key)
                    # print( phi)
                    break
        # Add extra fingertip points, direction depends on use_x_axis
        if num_fingertip_extra > 0:
            if use_x_axis:
                vec = np.array([1.0, 0.0, 0.0])
                dir_root = T_root_to_fingertip[:3, :3] @ vec
                dir_root = dir_root / np.linalg.norm(dir_root) if np.linalg.norm(dir_root) > 0 else np.array([1.0, 0.0, 0.0])
            else:
                dir_root = positions[fingertip] - positions[parent]
                dir_root = dir_root / np.linalg.norm(dir_root) if np.linalg.norm(dir_root) > 0 else np.array([0.0, 0.0, 0.0])
            T_parent_to_fingertip = compute_transform_from_joint_to_fingertip(finger_chain, parent, joint_info, q)
            T_root_to_parent = T_root_to_fingertip @ np.linalg.inv(T_parent_to_fingertip)
            T_parent_to_root = np.linalg.inv(T_root_to_parent)
            for m in range(1, num_fingertip_extra + 1):
                p_extra_root = positions[fingertip] + m * d * dir_root
                p_extra_root_h = np.append(p_extra_root, 1.0)
                local_h = T_parent_to_root @ p_extra_root_h
                local = local_h[:3].tolist()
                skeleton_point_dict[parent].append(local)

                # Compute position in sphere frame
                p_sphere_h = T_sphere_to_base @ T_base_to_root @ T_root_to_parent @ local_h
                p_sphere = p_sphere_h[:3]
                phi = np.arccos(p_sphere[2] / np.linalg.norm(p_sphere))
                skeleton_phi_dict[parent].append(phi)
    print(f"Total of {len(skeleton_point_dict[base_link])} points for base_link")
    for finger_chain in finger_chains:
        total_points = sum(len(skeleton_point_dict[j]) for j in finger_chain[1:] if j in skeleton_point_dict.keys())
        print(f"Total of {total_points} points for finger chain with tip '{finger_chain[-1]}'")
    return skeleton_point_dict, skeleton_phi_dict

def y_axis_palm_rotation_iterative(
    joint_info, kinematic_chains, vertices_dict, joint_type_info, q_opened,
    q_max_initial, base_link="base_link", max_iters=10, angle_tol_deg=2.0
):
    """
    Iteratively aligns palm_normal z-axis to xz-midpoint of fingertips.
    - Regenerates filtered palm_normal line points
    - Updates q_max via solve_inverse_kinematics
    - Rotates palm around y-axis
    - Stops when rotation < 2° or after 10 iterations
    """
    # Finger chains and fingertips
    finger_chains = [
        chain for chain in kinematic_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]
    if not finger_chains:
        raise ValueError("No finger chains found")
    fingertips = [chain[-1] for chain in finger_chains]

    # === Save initial T_palm_to_sphere (before any palm rotation) ===
    palm_xyz_init = np.array(joint_info["palm_normal"][0])
    palm_quat_init = joint_info["palm_normal"][1]
    T_base_to_palm_init = tf.quaternion_matrix(palm_quat_init)
    T_base_to_palm_init[:3, 3] = palm_xyz_init

    sphere_xyz = np.array(joint_info["sphere_frame"][0])
    sphere_quat = joint_info["sphere_frame"][1]
    T_base_to_sphere = tf.quaternion_matrix(sphere_quat)
    T_base_to_sphere[:3, 3] = sphere_xyz

    T_palm_to_sphere_init = np.linalg.inv(T_base_to_palm_init) @ T_base_to_sphere

    # Start with initial q_max
    q_max = q_max_initial.copy()

    for iteration in range(max_iters):
        # === 1. Update palm_normal line points ===
        palm_xyz = np.array(joint_info["palm_normal"][0])
        palm_quat = joint_info["palm_normal"][1]
        T_base_to_palm = tf.quaternion_matrix(palm_quat)
        T_base_to_palm[:3, 3] = palm_xyz
        T_palm_to_base = np.linalg.inv(T_base_to_palm)

        num_points = 1000
        z_values = np.linspace(-1.0, 10.0, num_points)
        local_points = np.zeros((num_points, 3))
        local_points[:, 2] = z_values

        filtered_points = []
        for p_local in local_points:
            p_base_h = np.append(p_local, 1)
            p_base = (T_base_to_palm @ p_base_h)[:3]
            p_palm_h = np.append(p_base, 1)
            p_palm = (T_palm_to_base @ p_palm_h)[:3]
            if p_palm[2] >= 0:
                filtered_points.append(p_local)
        vertices_dict["palm_normal"] = np.array(filtered_points)

        # === 2. Solve IK with updated palm points ===
        q_max = solve_inverse_kinematics(
            finger_chains, joint_info, kinematic_chains,
            vertices_dict, joint_type_info, q_opened
        )

        # === 3. Compute fingertip positions in palm frame ===
        fingertip_positions_base = []
        for fingertip in fingertips:
            T = compute_transform_to_joint(
                fingertip, kinematic_chains, joint_info, q_max, base_link
            )
            fingertip_positions_base.append(T[:3, 3])
        fingertip_positions_base = np.array(fingertip_positions_base)

        fingertip_positions_palm = []
        for pos_base in fingertip_positions_base:
            pos_h = np.append(pos_base, 1)
            pos_palm_h = T_palm_to_base @ pos_h
            fingertip_positions_palm.append(pos_palm_h[:3])
        fingertip_positions_palm = np.array(fingertip_positions_palm)

        # === 4. Compute xz-midpoint using (min + max)/2 ===
        fingertip_xz = fingertip_positions_palm.copy()
        fingertip_xz[:, 1] = 0  # Project to xz-plane
        min_pos = np.min(fingertip_xz, axis=0)
        max_pos = np.max(fingertip_xz, axis=0)
        xz_midpoint = (min_pos + max_pos) / 2

        # === 5. Desired z direction ===
        norm = np.linalg.norm(xz_midpoint)
        if norm < 1e-8:
            print("Warning: xz-midpoint at palm origin. Stopping.")
            break
        desired_z = xz_midpoint / norm

        # === 6. Compute rotation angle around y-axis ===
        current_z = np.array([0, 0, 1])
        cos_angle = np.dot(current_z, desired_z)

        if abs(cos_angle) > 0.9999:
            rotation_angle = 0.0
        else:
            cross = np.cross(current_z, desired_z)
            rotation_angle = np.arcsin(cross[1])
            if cos_angle < 0:
                rotation_angle = np.pi - rotation_angle if rotation_angle > 0 else -np.pi - rotation_angle

        # === 7. Check convergence ===
        if abs(rotation_angle) < np.deg2rad(angle_tol_deg):
            print(f"Converged after {iteration + 1} iteration(s). "
                  f"Angle: {np.rad2deg(abs(rotation_angle)):.3f}° < {angle_tol_deg}°")
            break

        # === 8. Apply rotation to palm frame ===
        R_y = tf.rotation_matrix(rotation_angle, [0, 1, 0])[:3, :3]
        R_palm_new = T_base_to_palm[:3, :3] @ R_y

        T_base_to_palm_new = np.eye(4)
        T_base_to_palm_new[:3, :3] = R_palm_new
        T_base_to_palm_new[:3, 3] = palm_xyz

        quat_palm_new = tf.quaternion_from_matrix(T_base_to_palm_new)
        joint_info["palm_normal"] = (
            palm_xyz.tolist(),
            quat_palm_new,
            "fixed", None, None, None, None
        )

        print(f"Iter {iteration + 1}: {np.rad2deg(abs(rotation_angle)):.3f}° rotation applied")
        print(f"   xz-midpoint: {xz_midpoint}")

    else:
        print(f"Max iterations ({max_iters}) reached. "
              f"Final angle: {np.rad2deg(abs(rotation_angle)):.3f}°")
        
    # === 9. After convergence: Update sphere_frame using saved T_palm_to_sphere_init ===
    palm_xyz_final = np.array(joint_info["palm_normal"][0])
    palm_quat_final = joint_info["palm_normal"][1]
    T_base_to_palm_final = tf.quaternion_matrix(palm_quat_final)
    T_base_to_palm_final[:3, 3] = palm_xyz_final

    # Reconstruct sphere frame: preserve original offset from palm
    T_base_to_sphere_final = T_base_to_palm_final @ T_palm_to_sphere_init
    sphere_xyz_final = T_base_to_sphere_final[:3, 3]
    sphere_quat_final = tf.quaternion_from_matrix(T_base_to_sphere_final)
    radius = joint_info["sphere_frame"][6]  # Preserve radius

    joint_info["sphere_frame"] = (
        sphere_xyz_final.tolist(),
        sphere_quat_final,
        "fixed", None, None, None,
        radius
    )

    print()
    return q_max